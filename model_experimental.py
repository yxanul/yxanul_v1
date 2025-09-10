#!/usr/bin/env python3
"""
Minimal BF16 Transformer with MoE (top-1) for fast experiments.

Targets:
- 10–12 layers, d_model 512–1024
- SwiGLU FFN (≈2.67x expansion, i.e., 8/3)
- MoE with top-1 routing, optional dropless routing
- Load-balancing auxiliary loss (0.01–0.1)
- Capacity factor 1.0–1.25 (used only if dropless=False)
- AdamW, RMSNorm, BF16 forward
- Vocab size 32768 by default (matches your setup)

Notes on expert size (SwiGLU MoE): params_per_expert ≈ 8*d^2.
- d=512  -> ~2.1M
- d=768  -> ~4.7M
- d=1024 -> ~8.4M

Default config here aims <~125M params for quick iteration:
- d=512, n_layer=10, n_head=8, n_experts=4 (MoE in every block)
- Rough total ≈ 110–115M with tied embeddings.
"""

from __future__ import annotations

import math
import time
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch._dynamo as _dynamo
    _dynamo_disable = _dynamo.disable
except Exception:
    def _dynamo_disable(fn):
        return fn

try:
    from wandb_logger import WandBLogger
except Exception:
    # Fallback no-op logger
    class WandBLogger:
        def __init__(self, *args, **kwargs): pass
        def watch(self, *args, **kwargs): pass
        def log_metrics(self, *args, **kwargs): pass
        def log_eval(self, *args, **kwargs): pass
        def set_summary(self, *args, **kwargs): pass
        def finish(self, *args, **kwargs): pass


# ----------------------------- Utilities -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [*, D]
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x = (x * rms).to(dtype)
        # Return a fresh tensor to avoid potential cudagraph aliasing with torch.compile
        out = x * self.weight
        return out


def swiglu(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # SwiGLU: silu(u) * v
    return F.silu(u) * v


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class RoPE:
    """Rotary position embeddings (applied to q, k)."""
    @staticmethod
    def create_cos_sin_cache(seq_len: int, n_elem: int, base: float = 10000.0, device: str = 'cpu', dtype: torch.dtype = torch.float32):
        # n_elem is head_dim
        half = n_elem // 2
        theta = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
        seq_idx = torch.arange(seq_len, device=device, dtype=torch.float32)
        idx_theta = torch.outer(seq_idx, theta)  # [T, half]
        cos = torch.cos(idx_theta).to(dtype)
        sin = torch.sin(idx_theta).to(dtype)
        return cos, sin  # each [T, half]

    @staticmethod
    def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, D]; cos/sin: [T, D/2]
        B, H, T, D = x.shape
        half = D // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        cos_t = cos[:T].to(dtype=x.dtype, device=x.device)[None, None, :, :]
        sin_t = sin[:T].to(dtype=x.dtype, device=x.device)[None, None, :, :]
        xr1 = x1 * cos_t - x2 * sin_t
        xr2 = x1 * sin_t + x2 * cos_t
        return torch.cat([xr1, xr2], dim=-1)


# ------------------------------ MoE parts ----------------------------

class ExpertSwiGLU(nn.Module):
    """One SwiGLU expert with expansion ~8/3.

    in_features -> hidden (8/3 * d) using two projections for SwiGLU, then down to in_features.
    """
    def __init__(self, d_model: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        hidden = int(round(d_model * (8.0 / 3.0)))
        # round to multiples of 64 for better kernels
        hidden = (hidden + 63) // 64 * 64
        self.up_u = nn.Linear(d_model, hidden, bias=bias)
        self.up_v = nn.Linear(d_model, hidden, bias=bias)
        self.down = nn.Linear(hidden, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.up_u(x)
        v = self.up_v(x)
        y = swiglu(u, v)
        y = self.down(y)
        return self.dropout(y)


class Top1Router(nn.Module):
    """Top-1 routing with optional dropless dispatch and load-balancing aux loss.

    When dropless=True: routes all tokens, no capacity truncation.
    When dropless=False: enforces per-expert capacity; extra tokens are dropped (masked to zero contribution).
    """
    def __init__(self, d_model: int, n_experts: int, capacity_factor: float = 1.25, dropless: bool = True,
                 load_balance_alpha: float = 0.05, router_z_loss_coef: float = 0.0,
                 temperature: float = 1.0, noise_std: float = 0.0, noise_type: str = 'gumbel'):
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.dropless = dropless
        self.load_balance_alpha = load_balance_alpha
        self.router_z_loss_coef = router_z_loss_coef
        self.w_gating = nn.Linear(d_model, n_experts, bias=True)
        # routing dynamics
        self.temperature = float(temperature)
        self.noise_std = float(noise_std)
        self.noise_type = str(noise_type)

    @torch.no_grad()
    def set_router_state(self, temperature: Optional[float] = None, noise_std: Optional[float] = None):
        if temperature is not None:
            self.temperature = float(temperature)
        if noise_std is not None:
            self.noise_std = float(noise_std)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing decisions.

        Returns
        - probs: [N, E] softmax probabilities
        - top1_idx: [N] selected expert indices
        - top1_prob: [N] selected expert probabilities
        - aux_loss: scalar tensor for load balancing (and router z-loss if enabled)
        - me: [E] fraction of tokens per expert (no grad)
        - ce: [E] mean gate prob per expert (no grad)
        - entropy_mean: [1] mean token entropy of router softmax (no grad)
        """
        logits = self.w_gating(x)  # [N, E]
        temp = max(1e-5, float(self.temperature))
        logits_base = logits / temp
        probs = F.softmax(logits_base, dim=-1)
        # Noisy top-1 selection for exploration (does not affect aux stats)
        logits_sel = logits_base
        if self.training and self.noise_std > 1e-8:
            if self.noise_type == 'gumbel':
                u = torch.rand_like(logits_sel).clamp_(1e-6, 1 - 1e-6)
                g = -torch.log(-torch.log(u))
                logits_sel = logits_sel + self.noise_std * g
            else:  # gaussian
                logits_sel = logits_sel + self.noise_std * torch.randn_like(logits_sel)
        top1_idx = logits_sel.argmax(dim=-1)
        # Use non-noisy probs to compute selected probability
        top1_prob = probs.gather(-1, top1_idx.unsqueeze(-1)).squeeze(-1)

        # Load balancing auxiliary loss (Switch-Transformer style)
        # fraction of tokens per expert (me) and mean probability per expert (ce)
        with torch.no_grad():
            N, E = probs.shape
            one_hot_assign = F.one_hot(top1_idx, num_classes=E).float()
            me = one_hot_assign.mean(dim=0)  # [E]
            ce = probs.mean(dim=0)           # [E]
            # Router entropy across tokens
            entropy_mean = (-(probs.clamp_min(1e-9).log() * probs).sum(dim=-1)).mean()
        aux = (self.n_experts * (me * ce).sum())
        aux = self.load_balance_alpha * aux

        # Optional z-loss on router logits (stabilizes softmax)
        if self.router_z_loss_coef > 0.0:
            z = torch.logsumexp(logits.float(), dim=-1)
            z_loss = (z.square()).mean() * self.router_z_loss_coef
            aux = aux + z_loss.to(aux.dtype)

        return probs, top1_idx, top1_prob, aux, me, ce, entropy_mean


class MoE(nn.Module):
    """Mixture-of-Experts with top-1 routing.

    Implementation emphasizes simplicity for small models and BF16 speed.
    Uses dropless routing by default: every token is processed by its selected expert.
    """
    def __init__(self, d_model: int, n_experts: int, bias: bool, dropout: float,
                 capacity_factor: float = 1.25, dropless: bool = True,
                 load_balance_alpha: float = 0.05, router_z_loss_coef: float = 0.0,
                 router_temperature: float = 1.0, router_noise_std: float = 0.0, router_noise_type: str = 'gumbel',
                 grouped: bool = False):
        super().__init__()
        self.n_experts = n_experts
        self.grouped = bool(grouped)
        self.router = Top1Router(
            d_model, n_experts, capacity_factor, dropless,
            load_balance_alpha, router_z_loss_coef,
            temperature=router_temperature, noise_std=router_noise_std, noise_type=router_noise_type,
        )
        self.experts = nn.ModuleList([
            ExpertSwiGLU(d_model, bias=bias, dropout=dropout) for _ in range(n_experts)
        ])
        self.dropout = nn.Dropout(dropout)

        # Fast path when n_experts=1 -> just dense SwiGLU
        self._dense_fallback = (n_experts == 1)
        if self._dense_fallback:
            self.dense = ExpertSwiGLU(d_model, bias=bias, dropout=dropout)

    @_dynamo_disable  # avoid torch.compile capturing highly dynamic routing code
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        if self._dense_fallback:
            return self.dense(x), x.new_zeros(())

        # Flatten tokens to route per-token
        x_flat = x.reshape(b * t, d)
        probs, top1_idx, top1_p, aux, me, ce, entropy_mean = self.router(x_flat)

        if self.router.dropless:
            # Dropless: process all tokens by their chosen expert and scatter back
            y_flat = torch.zeros_like(x_flat)
            for e in range(self.n_experts):
                mask = (top1_idx == e)
                if mask.any():
                    xe = x_flat[mask]
                    ye = self.experts[e](xe)
                    # Scale by gate probability (Switch-style top-1 weighting)
                    ye = ye * top1_p[mask].unsqueeze(-1)
                    y_flat[mask] = ye
            y = y_flat.reshape(b, t, d)
            # Stats (no drops in dropless)
            with torch.no_grad():
                max_frac = me.max()
                num_active = (me > 0).sum()
                drop_frac = torch.zeros((), dtype=torch.float32, device=x.device)
                top1_p_mean = top1_p.mean()
            # Store for logging
            self._last_stats = {
                'aux': aux.detach(),
                'me': me.detach(),
                'ce': ce.detach(),
                'entropy_mean': entropy_mean.detach(),
                'max_frac': max_frac.detach(),
                'num_active': num_active.detach(),
                'drop_frac': drop_frac.detach(),
                'top1_p_mean': top1_p_mean.detach(),
                'tokens': torch.tensor(b * t, device=x.device),
            }
            return self.dropout(y), aux
        else:
            # Capacity-limited variant (tokens beyond capacity are dropped to zero contribution)
            N = b * t
            cap = int(math.ceil(self.router.capacity_factor * (N / self.n_experts)))
            if not self.grouped:
                y_flat = torch.zeros_like(x_flat)
                processed = 0
                for e in range(self.n_experts):
                    # Select up to cap tokens with highest prob to expert e
                    p_e = probs[:, e]
                    routed = (top1_idx == e)
                    if routed.any():
                        idx_e = torch.nonzero(routed, as_tuple=False).squeeze(-1)
                        if idx_e.numel() > cap:
                            pe = p_e[idx_e]
                            topk = torch.topk(pe, cap, dim=0).indices
                            idx_e = idx_e[topk]
                        xe = x_flat[idx_e]
                        ye = self.experts[e](xe)
                        ye = ye * p_e[idx_e].unsqueeze(-1)
                        y_flat[idx_e] = ye
                        processed += int(idx_e.numel())
                y = y_flat.reshape(b, t, d)
                with torch.no_grad():
                    max_frac = me.max()
                    num_active = (me > 0).sum()
                    drop_frac = torch.tensor(1.0 - (processed / max(1, N)), device=x.device)
                    top1_p_mean = top1_p.mean()
                self._last_stats = {
                    'aux': aux.detach(),
                    'me': me.detach(),
                    'ce': ce.detach(),
                    'entropy_mean': entropy_mean.detach(),
                    'max_frac': max_frac.detach(),
                    'num_active': num_active.detach(),
                    'drop_frac': drop_frac.detach(),
                    'top1_p_mean': top1_p_mean.detach(),
                    'tokens': torch.tensor(b * t, device=x.device),
                    'cap': torch.tensor(cap, device=x.device),
                }
                return self.dropout(y), aux
            else:
                # Grouped/padded dispatch with batched GEMMs across experts
                device = x_flat.device
                dtype = x_flat.dtype
                d_model = d
                # Prepare per-expert indices up to capacity
                idx_lists = []
                counts = []
                processed = 0
                for e in range(self.n_experts):
                    p_e = probs[:, e]
                    routed = (top1_idx == e)
                    if routed.any():
                        idx_e = torch.nonzero(routed, as_tuple=False).squeeze(-1)
                        if idx_e.numel() > cap:
                            pe = p_e[idx_e]
                            topk = torch.topk(pe, cap, dim=0).indices
                            idx_e = idx_e[topk]
                    else:
                        idx_e = torch.empty(0, dtype=torch.long, device=device)
                    counts.append(int(idx_e.numel()))
                    processed += int(idx_e.numel())
                    # pad to cap
                    if idx_e.numel() < cap:
                        pad = torch.full((cap - idx_e.numel(),), -1, dtype=torch.long, device=device)
                        idx_e = torch.cat([idx_e, pad], dim=0)
                    else:
                        idx_e = idx_e[:cap]
                    idx_lists.append(idx_e)

                idx_mat = torch.stack(idx_lists, dim=0)  # [E, cap]
                valid_mask = (idx_mat >= 0)
                # Gather tokens with padding
                safe_idx = idx_mat.clamp_min(0)
                x_grouped = x_flat[safe_idx]            # [E, cap, d]
                gate_grouped = torch.zeros((self.n_experts, cap, 1), device=device, dtype=dtype)
                # selected prob per token for the chosen expert
                top1_p_grouped = top1_p[safe_idx]
                gate_grouped[valid_mask] = top1_p_grouped[valid_mask].unsqueeze(-1).to(dtype)

                # Stack weights for grouped GEMMs
                # up_u
                Wu = torch.stack([exp.up_u.weight for exp in self.experts], dim=0).to(dtype)  # [E, hidden, d]
                WuT = Wu.transpose(1, 2)  # [E, d, hidden]
                bu = torch.stack([exp.up_u.bias if exp.up_u.bias is not None else torch.zeros(exp.up_u.out_features, device=device, dtype=dtype) for exp in self.experts], dim=0).to(dtype)
                # up_v
                Wv = torch.stack([exp.up_v.weight for exp in self.experts], dim=0).to(dtype)
                WvT = Wv.transpose(1, 2)
                bv = torch.stack([exp.up_v.bias if exp.up_v.bias is not None else torch.zeros(exp.up_v.out_features, device=device, dtype=dtype) for exp in self.experts], dim=0).to(dtype)
                # down
                WdT = torch.stack([exp.down.weight.t() for exp in self.experts], dim=0).to(dtype)  # [E, hidden, d]
                bd = torch.stack([exp.down.bias if exp.down.bias is not None else torch.zeros(exp.down.out_features, device=device, dtype=dtype) for exp in self.experts], dim=0).to(dtype)

                # Batched GEMMs
                U = torch.bmm(x_grouped, WuT) + bu.unsqueeze(1)  # [E, cap, hidden]
                V = torch.bmm(x_grouped, WvT) + bv.unsqueeze(1)
                H = swiglu(U, V)
                Y = torch.bmm(H, WdT) + bd.unsqueeze(1)  # [E, cap, d]
                Y = Y * gate_grouped
                # Zero-out padding rows
                Y = Y.masked_fill(~valid_mask.unsqueeze(-1), 0)

                # Scatter back
                y_flat = torch.zeros_like(x_flat)
                for e in range(self.n_experts):
                    cnt = counts[e]
                    if cnt > 0:
                        idx_valid = idx_mat[e, :cnt]
                        y_flat.index_copy_(0, idx_valid, Y[e, :cnt])

                y = y_flat.reshape(b, t, d)
                with torch.no_grad():
                    max_frac = me.max()
                    num_active = (me > 0).sum()
                    drop_frac = torch.tensor(1.0 - (processed / max(1, N)), device=x.device)
                    top1_p_mean = top1_p.mean()
                self._last_stats = {
                    'aux': aux.detach(),
                    'me': me.detach(),
                    'ce': ce.detach(),
                    'entropy_mean': entropy_mean.detach(),
                    'max_frac': max_frac.detach(),
                    'num_active': num_active.detach(),
                    'drop_frac': drop_frac.detach(),
                    'top1_p_mean': top1_p_mean.detach(),
                    'tokens': torch.tensor(b * t, device=x.device),
                    'cap': torch.tensor(cap, device=x.device),
                }
                return self.dropout(y), aux


# --------------------------- Transformer blocks ---------------------------

class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout_p = float(dropout)

    def forward(self, x: torch.Tensor, rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        b, t, d = x.shape
        h = self.n_head
        q = self.q_proj(x).view(b, t, h, -1).transpose(1, 2)  # [b,h,t,d]
        k = self.k_proj(x).view(b, t, h, -1).transpose(1, 2)
        v = self.v_proj(x).view(b, t, h, -1).transpose(1, 2)
        # Optional RoPE
        if rope_cache is not None:
            cos, sin = rope_cache
            q = RoPE.apply_rope(q, cos, sin)
            k = RoPE.apply_rope(k, cos, sin)
        # Use PyTorch SDPA (enables Flash and mem-efficient kernels when available)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )  # [b,h,t,d]
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        y = self.o_proj(y)
        return y


class GatedMultiheadAttention(nn.Module):
    """SDPA attention with per-head, elementwise sigmoid gate (Qwen-style).

    Gate is computed from the input x via a linear projection with sigmoid.
    Then applied multiplicatively to the attention output per-head before o_proj.
    """
    def __init__(self, d_model: int, n_head: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        # Gate projection (head-specific via reshape)
        self.gate_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout_p = float(dropout)

    def forward(self, x: torch.Tensor, rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        b, t, d = x.shape
        h = self.n_head
        q = self.q_proj(x).view(b, t, h, -1).transpose(1, 2)  # [b,h,t,d_head]
        k = self.k_proj(x).view(b, t, h, -1).transpose(1, 2)
        v = self.v_proj(x).view(b, t, h, -1).transpose(1, 2)
        # Optional RoPE
        if rope_cache is not None:
            cos, sin = rope_cache
            q = RoPE.apply_rope(q, cos, sin)
            k = RoPE.apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )  # [b,h,t,d_head]

        # Elementwise, head-specific, sigmoid gate
        gate = torch.sigmoid(self.gate_proj(x))              # [b,t,d_model]
        gate = gate.view(b, t, h, -1).transpose(1, 2)        # [b,h,t,d_head]
        y = y * gate

        y = y.transpose(1, 2).contiguous().view(b, t, d)
        y = self.o_proj(y)
        return y


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, bias: bool, dropout: float,
                 n_experts: int, capacity_factor: float, dropless: bool,
                 load_balance_alpha: float, router_z_loss_coef: float,
                 attn_gate: str = 'none', use_rope: bool = True):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.use_rope = use_rope
        if attn_gate == 'sigmoid_head':
            self.attn = GatedMultiheadAttention(d_model, n_head, bias=bias, dropout=dropout)
        else:
            self.attn = MultiheadAttention(d_model, n_head, bias=bias, dropout=dropout)
        self.ln2 = RMSNorm(d_model)
        self.moe = MoE(
            d_model, n_experts, bias=bias, dropout=dropout,
            capacity_factor=capacity_factor, dropless=dropless,
            load_balance_alpha=load_balance_alpha, router_z_loss_coef=router_z_loss_coef,
            router_temperature=1.0, router_noise_std=0.0, router_noise_type='gumbel',
            grouped=False,
        )

    def forward(self, x: torch.Tensor, rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_in = self.ln1(x)
        attn_out = self.attn(attn_in, rope_cache if self.use_rope else None)
        x = x + attn_out
        y, aux = self.moe(self.ln2(x))
        x = x + y
        return x, aux


class TinyMoETransformer(nn.Module):
    def __init__(self,
                 vocab_size: int = 32768,
                 n_layer: int = 10,
                 n_head: int = 8,
                 d_model: int = 512,
                 block_size: int = 2048,
                 dropout: float = 0.0,
                 bias: bool = False,
                 n_experts: int = 4,
                 capacity_factor: float = 1.25,
                 dropless: bool = True,
                 load_balance_alpha: float = 0.05,
                 router_z_loss_coef: float = 0.0,
                 attn_gate: str = 'none',
                 router_temperature: float = 1.0,
                 router_noise_std: float = 0.0,
                 router_noise_type: str = 'gumbel',
                 use_rope: bool = True,
                 rope_theta: float = 10000.0,
                 moe_grouped: bool = False,
                 ):  # noqa: E501
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.use_rope = use_rope
        self.head_dim = d_model // n_head
        if self.use_rope:
            assert (self.head_dim % 2) == 0, "head_dim must be even for RoPE"

        self.wte = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList([
            TransformerBlock(
                d_model, n_head, bias, dropout,
                n_experts, capacity_factor, dropless,
                load_balance_alpha, router_z_loss_coef,
                attn_gate=attn_gate,
                use_rope=use_rope,
            )
            for _ in range(n_layer)
        ])
        # set grouped flag into each block's MoE
        if moe_grouped:
            for blk in self.h:
                if hasattr(blk, 'moe'):
                    blk.moe.grouped = True
        # initialize router dynamics state across blocks
        for blk in self.h:
            if hasattr(blk, 'moe'):
                blk.moe.router.set_router_state(router_temperature, router_noise_std)
                blk.moe.router.noise_type = router_noise_type
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.wte.weight

        # RoPE cache stored at model level
        if self.use_rope:
            cos, sin = RoPE.create_cos_sin_cache(self.block_size, self.head_dim, base=rope_theta)
            self.register_buffer('rope_cos', cos, persistent=False)
            self.register_buffer('rope_sin', sin, persistent=False)
        else:
            self.rope_cos = None
            self.rope_sin = None

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        b, t = idx.shape
        assert t <= self.block_size
        x = self.wte(idx)
        x = self.drop(x)
        aux_losses = []
        rope_cache = (self.rope_cos, self.rope_sin) if self.use_rope else None
        for blk in self.h:
            x, aux = blk(x, rope_cache=rope_cache)
            aux_losses.append(aux)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # compute CE in float32 for stability
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1), ignore_index=-1)
            # sum of aux losses (already scaled by alpha)
            if aux_losses:
                aux_total = torch.stack([a.float() for a in aux_losses]).mean()
            else:
                aux_total = logits.new_zeros(())
            loss = ce + aux_total
        return logits, loss

    @torch.no_grad()
    def num_parameters(self) -> int:
        return count_parameters(self)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(1e-8, temperature)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# --------------------------- Data + training ---------------------------

@dataclass
class TrainConfig:
    # Model
    vocab_size: int = 32768
    n_layer: int = 10
    n_head: int = 8
    d_model: int = 512
    n_experts: int = 4
    block_size: int = 2048
    dropout: float = 0.0
    bias: bool = False
    capacity_factor: float = 1.25
    dropless: bool = True
    load_balance_alpha: float = 0.05
    router_z_loss_coef: float = 0.0
    attn_gate: str = 'none'  # 'none' or 'sigmoid_head'
    use_rope: bool = True
    rope_theta: float = 10000.0
    # Router dynamics
    router_temp_init: float = 1.5
    router_temp_final: float = 1.0
    router_temp_anneal_iters: int = 1000
    router_noise_std_init: float = 0.5
    router_noise_decay_iters: int = 1000
    router_noise_type: str = 'gumbel'  # 'gumbel' or 'gaussian'
    moe_grouped: bool = False

    # Training
    device: str = 'cuda'
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_iters: int = 2000
    eval_interval: int = 200
    eval_iters: int = 50
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_iters: int = 200
    lr_decay_iters: int = 2000
    grad_clip: float = 1.0
    compile: bool = False
    seed: int = 1337

    # IO
    data_path: str = 'train.bin'  # can be a dir containing train.bin/val.bin or a single .bin file
    checkpoint_dir: str = 'checkpoints_moe_bf16'
    log_interval: int = 10

    # Logging
    wandb_project: str = 'moe-bf16-experiments'
    wandb_run_name: Optional[str] = None


class BinDataLoader:
    """Memory-mapped uint16 tokens. Accepts path to .bin or directory.

    If only train.bin is present, creates val.bin as the last 1% of the tokens
    (keeps train.bin intact, does not rewrite).
    """
    def __init__(self, data_path: str, block_size: int, device: str):
        self.block_size = block_size
        self.device = device
        p = Path(data_path)
        if p.is_dir():
            train_path = p / 'train.bin'
            val_path = p / 'val.bin'
        else:
            train_path = p
            val_path = p.parent / 'val.bin'
        if not train_path.exists():
            raise FileNotFoundError(f"train.bin not found at {train_path}")

        train_mm = np.memmap(train_path, dtype=np.uint16, mode='r')
        if not val_path.exists():
            # Create last 1% as validation split
            n = train_mm.shape[0]
            cut = max(int(0.99 * n), self.block_size + 1)
            cut = min(cut, n - (self.block_size + 1))
            val_tokens = train_mm[cut:]
            print(f"[data] Creating {val_path} with {len(val_tokens):,} tokens (1% tail)")
            with open(val_path, 'wb') as f:
                val_tokens.tofile(f)

        self.train_mm = train_mm
        self.val_mm = np.memmap(val_path, dtype=np.uint16, mode='r')

    def _get_batch(self, mm: np.memmap, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n = mm.shape[0] - (self.block_size + 1)
        ix = torch.randint(0, max(1, n), (batch_size,))
        x = torch.stack([torch.from_numpy((mm[i:i + self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((mm[i + 1:i + 1 + self.block_size]).astype(np.int64)) for i in ix])
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def get_batch(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if split == 'train':
            return self._get_batch(self.train_mm, batch_size)
        else:
            return self._get_batch(self.val_mm, batch_size)


def cosine_lr(it: int, base_lr: float, min_lr: float, warmup: int, total: int) -> float:
    if it < warmup:
        return base_lr * (it + 1) / max(1, warmup)
    if it >= total:
        return min_lr
    progress = (it - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def _anneal_linear(it: int, init_v: float, final_v: float, total: int) -> float:
    if total <= 0:
        return final_v
    if it >= total:
        return final_v
    a = it / float(total)
    return (1 - a) * init_v + a * final_v


def evaluate(model: TinyMoETransformer, data: BinDataLoader, cfg: TrainConfig, eval_iters: int) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = data.get_batch('val', cfg.batch_size)
            logits, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def train(cfg: TrainConfig):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    data = BinDataLoader(cfg.data_path, cfg.block_size, device=str(device))

    model = TinyMoETransformer(
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        d_model=cfg.d_model,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
        bias=cfg.bias,
        n_experts=cfg.n_experts,
        capacity_factor=cfg.capacity_factor,
        dropless=cfg.dropless,
        load_balance_alpha=cfg.load_balance_alpha,
        router_z_loss_coef=cfg.router_z_loss_coef,
        attn_gate=cfg.attn_gate,
        router_temperature=cfg.router_temp_init,
        router_noise_std=cfg.router_noise_std_init,
        router_noise_type=cfg.router_noise_type,
        use_rope=cfg.use_rope,
        rope_theta=cfg.rope_theta,
        moe_grouped=cfg.moe_grouped,
    ).to(device=device, dtype=torch.bfloat16)

    total_params = model.num_parameters()
    print(f"Model params: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Config: layers={cfg.n_layer}, d_model={cfg.d_model}, heads={cfg.n_head}, experts={cfg.n_experts}")
    compiled_enabled = False
    if cfg.compile and hasattr(torch, 'compile'):
        # Configure inductor to skip cudagraphs for dynamic shapes (MoE token counts vary per step)
        try:
            import torch._inductor.config as inductor_config
            # Prefer skipping cudagraph capture on dynamic graphs to avoid aliasing/overwrites
            if hasattr(inductor_config.triton, 'cudagraph_skip_dynamic_graphs'):
                inductor_config.triton.cudagraph_skip_dynamic_graphs = True
            if hasattr(inductor_config.triton, 'cudagraph_dynamic_shape_warn_limit'):
                # Silence or limit warnings (set to None or 0 to silence)
                inductor_config.triton.cudagraph_dynamic_shape_warn_limit = None
            # As a stronger fallback, try disabling cudagraphs entirely if available
            if hasattr(inductor_config.triton, 'cudagraphs'):
                # Disable cudagraphs to avoid aliasing with gradient accumulation
                inductor_config.triton.cudagraphs = False
        except Exception:
            pass

        model = torch.compile(model, mode='max-autotune')
        compiled_enabled = True

    # cudagraph step marker for torch.compile
    _mark_step = None
    if compiled_enabled:
        try:
            from torch.compiler import cudagraph_mark_step_begin as _mark_step
        except Exception:
            try:
                from torch._inductor import cudagraph_mark_step_begin as _mark_step
            except Exception:
                _mark_step = None

    # Optimizer
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if p.dim() >= 2 and 'wte' not in n:  # don't decay embeddings' shared weights
                decay_params.append(p)
            else:
                no_decay_params.append(p)
    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))

    # Logger
    logger = WandBLogger(
        enabled=True,
        project=cfg.wandb_project,
        run_name=cfg.wandb_run_name,
        config=asdict(cfg),
    )
    logger.watch(model)

    # Training loop
    model.train()
    best_val = float('inf')
    # If a previous best exists, seed best_val so we don't overwrite with worse
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / 'best_moe_bf16.pt'
    if best_path.exists():
        try:
            prev = torch.load(best_path, map_location='cpu')
            if isinstance(prev, dict) and 'val_loss' in prev:
                best_val = float(prev['val_loss'])
                print(f"Found existing best checkpoint with val_loss={best_val:.4f} at {best_path}")
        except Exception as _e:
            print(f"Warning: could not read existing best checkpoint: {_e}")
    t0 = time.time()
    tokens_seen = 0
    clip_cum = 0
    for it in range(cfg.max_iters):
        lr = cosine_lr(it, cfg.learning_rate, cfg.min_lr, cfg.warmup_iters, cfg.lr_decay_iters)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0  # sum of raw micro losses (for averaging only)
        last_router_metrics = None
        for micro in range(cfg.gradient_accumulation_steps):
            x, y = data.get_batch('train', cfg.batch_size)
            if _mark_step is not None:
                # Helps avoid cudagraph output aliasing across iterations
                _mark_step()
            # Update router schedules (temperature and noise) per-step
            curr_temp = _anneal_linear(it, cfg.router_temp_init, cfg.router_temp_final, cfg.router_temp_anneal_iters)
            curr_noise = _anneal_linear(it, cfg.router_noise_std_init, 0.0, cfg.router_noise_decay_iters)
            for blk in model.h:
                if hasattr(blk, 'moe'):
                    blk.moe.router.set_router_state(curr_temp, curr_noise)
            logits, loss = model(x, y)
            if compiled_enabled:
                # Ensure outputs are not aliased to cudagraph internal buffers
                logits = logits.clone()
                loss = loss.clone()
            (loss.float() / cfg.gradient_accumulation_steps).backward()
            total_loss += float(loss.detach())
            tokens_seen += (x.numel())

            # Gather router/expert stats from this micro-step (use the last one for logging)
            try:
                layers_stats = []
                for i, blk in enumerate(model.h):
                    moe = getattr(blk, 'moe', None)
                    if moe is None or getattr(moe, '_dense_fallback', False):
                        continue
                    st = getattr(moe, '_last_stats', None)
                    if st is None:
                        continue
                    # Move small scalars to CPU floats for logging
                    entry = {
                        'layer': i,
                        'aux': float(st['aux'].detach().float().cpu()),
                        'max_frac': float(st['max_frac'].detach().float().cpu()),
                        'num_active': int(st['num_active'].detach().long().cpu()),
                        'drop_frac': float(st['drop_frac'].detach().float().cpu()),
                        'top1_p_mean': float(st['top1_p_mean'].detach().float().cpu()),
                        'entropy_mean': float(st['entropy_mean'].detach().float().cpu()),
                    }
                    # Per-expert fractions summary
                    me = st['me'].detach().float().cpu()
                    entry['me_mean'] = float(me.mean())
                    entry['me_max'] = float(me.max())
                    entry['me_min'] = float(me.min())
                    layers_stats.append(entry)
                if layers_stats:
                    # Aggregate across layers
                    max_fracs = [e['max_frac'] for e in layers_stats]
                    num_active = [e['num_active'] for e in layers_stats]
                    auxs = [e['aux'] for e in layers_stats]
                    drops = [e['drop_frac'] for e in layers_stats]
                    tpm = [e['top1_p_mean'] for e in layers_stats]
                    ent = [e['entropy_mean'] for e in layers_stats]
                    last_router_metrics = {
                        'router/aux_mean': sum(auxs) / len(auxs),
                        'router/max_frac_mean': sum(max_fracs) / len(max_fracs),
                        'router/max_frac_max': max(max_fracs),
                        'router/active_min': min(num_active),
                        'router/active_mean': sum(num_active) / len(num_active),
                        'router/drop_frac_mean': sum(drops) / len(drops),
                        'router/top1_p_mean': sum(tpm) / len(tpm),
                        'router/entropy_mean': sum(ent) / len(ent),
                    }
                    last_router_metrics['router/collapsed'] = 1 if last_router_metrics['router/max_frac_max'] >= 0.90 or last_router_metrics['router/active_min'] <= 1 else 0
            except Exception:
                last_router_metrics = None

        grad_total_norm = None
        clipped_flag = None
        if cfg.grad_clip > 0:
            try:
                grad_total_norm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
                if math.isfinite(grad_total_norm):
                    clipped_flag = 1 if grad_total_norm > (cfg.grad_clip + 1e-12) else 0
                    if clipped_flag:
                        clip_cum += 1
            except Exception:
                grad_total_norm = None
                clipped_flag = None
        optimizer.step()

        # Logging
        if (it % cfg.log_interval) == 0:
            dt = max(1e-6, time.time() - t0)
            tps = tokens_seen / dt
            avg_loss = total_loss / max(1, cfg.gradient_accumulation_steps)
            metrics = {
                'train/loss': avg_loss,
                'train/lr': lr,
                'speed/tokens_per_s': tps,
            }
            # Router schedule metrics
            metrics['router/temp'] = curr_temp
            metrics['router/noise_std'] = curr_noise
            if grad_total_norm is not None and math.isfinite(grad_total_norm):
                metrics['grad/global_norm'] = grad_total_norm
                metrics['grad/clip_threshold'] = float(cfg.grad_clip)
            if clipped_flag is not None:
                metrics['grad/clipped'] = int(clipped_flag)
                metrics['grad/clipped_cum'] = int(clip_cum)
            if last_router_metrics:
                metrics.update(last_router_metrics)
            logger.log_metrics(metrics, step=it)
            # Console summary with key router signals
            if last_router_metrics:
                rf = last_router_metrics['router/max_frac_max']
                na = last_router_metrics['router/active_min']
                col = last_router_metrics['router/collapsed']
                print(f"iter {it:6d} | loss {avg_loss:.4f} | lr {lr:.3e} | {tps:.0f} tok/s | r_max {rf:.2f} act_min {na} col {col}")
            else:
                print(f"iter {it:6d} | loss {avg_loss:.4f} | lr {lr:.3e} | {tps:.0f} tok/s")
            t0 = time.time(); tokens_seen = 0

        # Eval
        if (it % cfg.eval_interval) == 0:
            val_loss = evaluate(model, data, cfg, cfg.eval_iters)
            logger.log_metrics({'val/loss': val_loss, 'val/ppl': math.exp(min(20.0, val_loss))}, step=it)
            print(f"eval | val_loss {val_loss:.4f}")
            # Always save a rolling 'last' checkpoint for convenience
            try:
                torch.save({'model': model.state_dict(), 'cfg': asdict(cfg), 'val_loss': float(val_loss), 'iter': it}, ckpt_dir / 'last_moe_bf16.pt')
            except Exception:
                pass

            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = Path(cfg.checkpoint_dir) / 'best_moe_bf16.pt'
                torch.save({'model': model.state_dict(), 'cfg': asdict(cfg), 'val_loss': val_loss, 'iter': it}, ckpt_path)

    logger.set_summary(best_val_loss=best_val, params=total_params)
    logger.finish()


def main():
    import argparse
    p = argparse.ArgumentParser(description='BF16 Tiny MoE Transformer trainer')
    # Model
    p.add_argument('--vocab_size', type=int, default=32768)
    p.add_argument('--n_layer', type=int, default=10)
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--n_experts', type=int, default=4)
    p.add_argument('--block_size', type=int, default=2048)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--bias', action='store_true')
    p.add_argument('--capacity_factor', type=float, default=1.25)
    p.add_argument('--dropless', action='store_true')
    p.add_argument('--no-dropless', dest='dropless', action='store_false')
    p.set_defaults(dropless=True)
    p.add_argument('--load_balance_alpha', type=float, default=0.05)
    p.add_argument('--router_z_loss_coef', type=float, default=0.0)
    p.add_argument('--attn_gate', type=str, default='none', choices=['none', 'sigmoid_head'], help='Enable SDPA + elementwise head-specific sigmoid gate')
    p.add_argument('--use_rope', dest='use_rope', action='store_true')
    p.add_argument('--no-use_rope', dest='use_rope', action='store_false')
    p.set_defaults(use_rope=True)
    p.add_argument('--rope_theta', type=float, default=10000.0)
    # Router dynamics CLI
    p.add_argument('--router_temp_init', type=float, default=1.5)
    p.add_argument('--router_temp_final', type=float, default=1.0)
    p.add_argument('--router_temp_anneal_iters', type=int, default=1000)
    p.add_argument('--router_noise_std_init', type=float, default=0.5)
    p.add_argument('--router_noise_decay_iters', type=int, default=1000)
    p.add_argument('--router_noise_type', type=str, default='gumbel', choices=['gumbel', 'gaussian'])
    p.add_argument('--moe_grouped', action='store_true', help='Use grouped/padded MoE with batched GEMMs (capacity mode)')

    # Train
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--gradient_accumulation_steps', type=int, default=1)
    p.add_argument('--max_iters', type=int, default=2000)
    p.add_argument('--eval_interval', type=int, default=200)
    p.add_argument('--eval_iters', type=int, default=50)
    p.add_argument('--learning_rate', type=float, default=3e-4)
    p.add_argument('--min_lr', type=float, default=3e-5)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.95)
    p.add_argument('--warmup_iters', type=int, default=200)
    p.add_argument('--lr_decay_iters', type=int, default=2000)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--seed', type=int, default=1337)

    # IO
    p.add_argument('--data_path', type=str, default='train.bin')
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints_moe_bf16')
    p.add_argument('--log_interval', type=int, default=10)

    # Logging
    p.add_argument('--wandb_project', type=str, default='moe-bf16-experiments')
    p.add_argument('--wandb_run_name', type=str, default=None)

    args = p.parse_args()
    cfg = TrainConfig(**vars(args))

    # Print a quick expert size estimate
    d = cfg.d_model
    params_per_expert = 8 * (d ** 2)
    print(f"Estimated params/expert (SwiGLU): ~{params_per_expert/1e6:.2f}M for d={d}")
    train(cfg)


if __name__ == '__main__':
    main()
