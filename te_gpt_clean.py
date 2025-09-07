#!/usr/bin/env python3
"""
Clean, modern GPT with TransformerEngine FP8 (optional), RMSNorm, GQA, RoPE, SwiGLU.

Targets latest PyTorch (>=2.8) + CUDA 13 (Blackwell). Designed for speed and clarity.

Key features:
- Deep & narrow friendly; configure via ModelCfg
- RMSNorm everywhere, optional QK-Norm on Q and K
- GQA (separate KV heads); full-causal SDPA by default
- Optional sliding-window attention without additive masks (fused kernels possible)
- SwiGLU MLP
- TE fp8_autocast wrapper (HYBRID format) optional
"""
import os
os.environ.setdefault("NVTE_FUSED_ATTN_BACKEND", "1")  # prefer fused TE attention if used internally

from dataclasses import dataclass
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    TE_AVAILABLE = True
except Exception:  # pragma: no cover
    TE_AVAILABLE = False


def set_torch_defaults():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass


@dataclass
class ModelCfg:
    vocab_size: int = 32768
    n_layer: int = 48
    n_head: int = 8
    n_kv_head: int = 2  # GQA
    n_embd: int = 640
    block_size: int = 2048
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10000.0
    rope_fraction: float = 1.0  # 1.0 = full rotary
    use_qk_norm: bool = True

    # Attention windowing (optional)
    attn_window: int = 0  # 0 = full causal; >0 = local sliding window
    attn_chunk: int = 512  # chunk size along time for local attention

    # TransformerEngine FP8 control
    use_fp8: bool = False
    fp8_amax_history_len: int = 16
    fp8_amax_compute_algo: str = "max"

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        assert self.n_head % self.n_kv_head == 0
        self.head_dim = self.n_embd // self.n_head
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        if self.use_fp8:
            assert self.n_embd % 16 == 0 and self.head_dim % 16 == 0, "Dims must be mult of 16 for FP8"


def get_fp8_recipe(cfg: ModelCfg):
    return DelayedScaling(
        fp8_format=Format.HYBRID,  # E4M3 fwd, E5M2 bwd
        amax_history_len=cfg.fp8_amax_history_len,
        amax_compute_algo=cfg.fp8_amax_compute_algo,
    )


class RoPE:
    @staticmethod
    def cos_sin(T: int, rotary_dims: int, theta: float, device, dtype):
        # returns cos, sin with shape (T, rotary_dims/2)
        inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dims, 2, device=device, dtype=torch.float32) / rotary_dims))
        pos = torch.arange(0, T, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)  # (T, rotary_dims/2)
        cos = torch.cos(freqs).to(dtype)
        sin = torch.sin(freqs).to(dtype)
        return cos, sin

    @staticmethod
    def apply(x: torch.Tensor, rope_fraction: float, theta: float):
        # x: (B, H, T, Dh)
        B, H, T, Dh = x.shape
        rotary_dims = int(Dh * rope_fraction)
        if rotary_dims % 2 == 1:
            rotary_dims -= 1
        if rotary_dims <= 0:
            return x
        cos, sin = RoPE.cos_sin(T, rotary_dims, theta, x.device, x.dtype)
        x_rot = x[..., :rotary_dims]
        x_pass = x[..., rotary_dims:]
        x1, x2 = x_rot[..., ::2], x_rot[..., 1::2]  # (..., T, rotary_dims/2)
        x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return torch.cat([x_rotated, x_pass], dim=-1)


def rmsnorm_lastdim(x: torch.Tensor, eps: float = 1e-6):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, bias: bool = False):
        super().__init__()
        Linear = te.Linear if TE_AVAILABLE else nn.Linear
        self.in_proj = Linear(dim, 2 * hidden, bias=bias, params_dtype=torch.bfloat16 if TE_AVAILABLE else None)
        self.out_proj = Linear(hidden, dim, bias=bias, params_dtype=torch.bfloat16 if TE_AVAILABLE else None)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x):
        x = self.in_proj(x)
        a, b = x.chunk(2, dim=-1)
        return self.out_proj(a * F.silu(b))


class Attention(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.h = cfg.n_head
        self.kv = cfg.n_kv_head
        self.dh = cfg.head_dim
        self.groups = self.h // self.kv
        self.rope_fraction = cfg.rope_fraction
        self.rope_theta = cfg.rope_theta
        self.use_qk_norm = cfg.use_qk_norm
        self.window = cfg.attn_window
        self.chunk = cfg.attn_chunk

        Linear = te.Linear if TE_AVAILABLE else nn.Linear
        self.wq = Linear(cfg.n_embd, self.h * self.dh, bias=cfg.bias, params_dtype=torch.bfloat16 if TE_AVAILABLE else None)
        self.wk = Linear(cfg.n_embd, self.kv * self.dh, bias=cfg.bias, params_dtype=torch.bfloat16 if TE_AVAILABLE else None)
        self.wv = Linear(cfg.n_embd, self.kv * self.dh, bias=cfg.bias, params_dtype=torch.bfloat16 if TE_AVAILABLE else None)
        self.wo = Linear(self.h * self.dh, cfg.n_embd, bias=cfg.bias, params_dtype=torch.bfloat16 if TE_AVAILABLE else None)
        nn.init.zeros_(self.wo.weight)
        self.drop = nn.Dropout(cfg.dropout)
        self.drop_p = cfg.dropout

    def _sdpa(self, q, k, v, *, causal: bool):
        # Prefer fused kernels on supported builds
        try:
            with torch.nn.attention.sdpa_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0 if not self.training else self.drop_p, is_causal=causal)
        except Exception:
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0 if not self.training else self.drop_p, is_causal=causal)

    def forward(self, x: torch.Tensor):
        # x: (B, T, D)
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.h, self.dh).permute(0, 2, 1, 3)  # (B,H,T,Dh)
        k = self.wk(x).view(B, T, self.kv, self.dh).permute(0, 2, 1, 3)  # (B,KV,T,Dh)
        v = self.wv(x).view(B, T, self.kv, self.dh).permute(0, 2, 1, 3)
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)
        if self.use_qk_norm:
            q = rmsnorm_lastdim(q)
            k = rmsnorm_lastdim(k)
        q = RoPE.apply(q, self.rope_fraction, self.rope_theta)
        k = RoPE.apply(k, self.rope_fraction, self.rope_theta)

        if self.window and self.window > 0:
            cs = self.chunk
            y_full = torch.empty_like(q)
            for s in range(0, T, cs):
                e = min(s + cs, T)
                ks = max(0, s - (self.window - 1))
                q_c = q[:, :, s:e, :]
                k_s = k[:, :, ks:e, :]
                v_s = v[:, :, ks:e, :]
                y_c = self._sdpa(q_c, k_s, v_s, causal=True)
                y_full[:, :, s:e, :] = y_c
            y = y_full
        else:
            y = self._sdpa(q, k, v, causal=True)

        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, self.h * self.dh)
        y = self.wo(y)
        return self.drop(y)


class Block(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        Norm = te.RMSNorm if TE_AVAILABLE else nn.LayerNorm  # use RMSNorm via TE; fall back to LN if needed
        self.n1 = Norm(cfg.n_embd, eps=1e-6)
        self.attn = Attention(cfg)
        self.n2 = Norm(cfg.n_embd, eps=1e-6)
        hidden = (int(cfg.n_embd * 8 / 3) + 63) // 64 * 64
        self.mlp = SwiGLU(cfg.n_embd, hidden, bias=cfg.bias)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        Norm = te.RMSNorm if TE_AVAILABLE else nn.LayerNorm
        self.in_norm = Norm(cfg.n_embd, eps=1e-6)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.out_norm = Norm(cfg.n_embd, eps=1e-6)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith('wo.weight'):
                nn.init.zeros_(p)

    def _init(self, m):
        if isinstance(m, (nn.Linear,)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif TE_AVAILABLE and isinstance(m, te.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        x = self.embed(idx)
        x = self.in_norm(x)
        if self.cfg.use_fp8 and TE_AVAILABLE:
            recipe = get_fp8_recipe(self.cfg)
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                for blk in self.blocks:
                    x = blk(x)
        else:
            for blk in self.blocks:
                x = blk(x)
        x = self.out_norm(x)
        logits = F.linear(x, self.lm_head.weight)
        return logits


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

