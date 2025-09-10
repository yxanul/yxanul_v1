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
        return x * self.weight


def swiglu(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # SwiGLU: silu(u) * v
    return F.silu(u) * v


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


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
                 load_balance_alpha: float = 0.05, router_z_loss_coef: float = 0.0):
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.dropless = dropless
        self.load_balance_alpha = load_balance_alpha
        self.router_z_loss_coef = router_z_loss_coef
        self.w_gating = nn.Linear(d_model, n_experts, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing decisions.

        Returns
        - probs: [N, E] softmax probabilities
        - top1_idx: [N] selected expert indices
        - top1_prob: [N] selected expert probabilities
        - aux_loss: scalar tensor for load balancing (and router z-loss if enabled)
        """
        logits = self.w_gating(x)  # [N, E]
        probs = F.softmax(logits, dim=-1)
        top1_prob, top1_idx = probs.max(dim=-1)

        # Load balancing auxiliary loss (Switch-Transformer style)
        # fraction of tokens per expert (me) and mean probability per expert (ce)
        with torch.no_grad():
            N, E = probs.shape
            one_hot_assign = F.one_hot(top1_idx, num_classes=E).float()
            me = one_hot_assign.mean(dim=0)  # [E]
            ce = probs.mean(dim=0)           # [E]
        aux = (self.n_experts * (me * ce).sum())
        aux = self.load_balance_alpha * aux

        # Optional z-loss on router logits (stabilizes softmax)
        if self.router_z_loss_coef > 0.0:
            z = torch.logsumexp(logits.float(), dim=-1)
            z_loss = (z.square()).mean() * self.router_z_loss_coef
            aux = aux + z_loss.to(aux.dtype)

        return probs, top1_idx, top1_prob, aux


class MoE(nn.Module):
    """Mixture-of-Experts with top-1 routing.

    Implementation emphasizes simplicity for small models and BF16 speed.
    Uses dropless routing by default: every token is processed by its selected expert.
    """
    def __init__(self, d_model: int, n_experts: int, bias: bool, dropout: float,
                 capacity_factor: float = 1.25, dropless: bool = True,
                 load_balance_alpha: float = 0.05, router_z_loss_coef: float = 0.0):
        super().__init__()
        self.n_experts = n_experts
        self.router = Top1Router(
            d_model, n_experts, capacity_factor, dropless,
            load_balance_alpha, router_z_loss_coef,
        )
        self.experts = nn.ModuleList([
            ExpertSwiGLU(d_model, bias=bias, dropout=dropout) for _ in range(n_experts)
        ])
        self.dropout = nn.Dropout(dropout)

        # Fast path when n_experts=1 -> just dense SwiGLU
        self._dense_fallback = (n_experts == 1)
        if self._dense_fallback:
            self.dense = ExpertSwiGLU(d_model, bias=bias, dropout=dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        if self._dense_fallback:
            return self.dense(x), x.new_zeros(())

        # Flatten tokens to route per-token
        x_flat = x.reshape(b * t, d)
        probs, top1_idx, top1_p, aux = self.router(x_flat)

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
            return self.dropout(y), aux
        else:
            # Capacity-limited variant (tokens beyond capacity are dropped to zero contribution)
            N = b * t
            cap = int(math.ceil(self.router.capacity_factor * (N / self.n_experts)))
            y_flat = torch.zeros_like(x_flat)
            for e in range(self.n_experts):
                # Select up to cap tokens with highest prob to expert e
                p_e = probs[:, e]
                # find tokens routed to e
                routed = (top1_idx == e)
                if routed.any():
                    idx_e = torch.nonzero(routed, as_tuple=False).squeeze(-1)
                    if idx_e.numel() > cap:
                        # pick top-k by probability
                        pe = p_e[idx_e]
                        topk = torch.topk(pe, cap, dim=0).indices
                        idx_e = idx_e[topk]
                    xe = x_flat[idx_e]
                    ye = self.experts[e](xe)
                    ye = ye * p_e[idx_e].unsqueeze(-1)
                    y_flat[idx_e] = ye
            y = y_flat.reshape(b, t, d)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.n_head
        q = self.q_proj(x).view(b, t, h, -1).transpose(1, 2)  # [b,h,t,d]
        k = self.k_proj(x).view(b, t, h, -1).transpose(1, 2)
        v = self.v_proj(x).view(b, t, h, -1).transpose(1, 2)
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


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, bias: bool, dropout: float,
                 n_experts: int, capacity_factor: float, dropless: bool,
                 load_balance_alpha: float, router_z_loss_coef: float):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiheadAttention(d_model, n_head, bias=bias, dropout=dropout)
        self.ln2 = RMSNorm(d_model)
        self.moe = MoE(
            d_model, n_experts, bias=bias, dropout=dropout,
            capacity_factor=capacity_factor, dropless=dropless,
            load_balance_alpha=load_balance_alpha, router_z_loss_coef=router_z_loss_coef,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln1(x))
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
                 ):  # noqa: E501
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model

        self.wte = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList([
            TransformerBlock(
                d_model, n_head, bias, dropout,
                n_experts, capacity_factor, dropless,
                load_balance_alpha, router_z_loss_coef,
            )
            for _ in range(n_layer)
        ])
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.wte.weight

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
        for blk in self.h:
            x, aux = blk(x)
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
    ).to(device=device, dtype=torch.bfloat16)

    total_params = model.num_parameters()
    print(f"Model params: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Config: layers={cfg.n_layer}, d_model={cfg.d_model}, heads={cfg.n_head}, experts={cfg.n_experts}")
    if cfg.compile and hasattr(torch, 'compile'):
        model = torch.compile(model, mode='max-autotune')

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
    t0 = time.time()
    tokens_seen = 0
    for it in range(cfg.max_iters):
        lr = cosine_lr(it, cfg.learning_rate, cfg.min_lr, cfg.warmup_iters, cfg.lr_decay_iters)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for micro in range(cfg.gradient_accumulation_steps):
            x, y = data.get_batch('train', cfg.batch_size)
            logits, loss = model(x, y)
            (loss.float() / cfg.gradient_accumulation_steps).backward()
            total_loss += loss.item()
            tokens_seen += (x.numel())

        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # Logging
        if (it % cfg.log_interval) == 0:
            dt = max(1e-6, time.time() - t0)
            tps = tokens_seen / dt
            logger.log_metrics({
                'train/loss': total_loss,
                'train/lr': lr,
                'speed/tokens_per_s': tps,
            }, step=it)
            print(f"iter {it:6d} | loss {total_loss:.4f} | lr {lr:.3e} | {tps:.0f} tok/s")
            t0 = time.time(); tokens_seen = 0

        # Eval
        if (it % cfg.eval_interval) == 0:
            val_loss = evaluate(model, data, cfg, cfg.eval_iters)
            logger.log_metrics({'val/loss': val_loss, 'val/ppl': math.exp(min(20.0, val_loss))}, step=it)
            print(f"eval | val_loss {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
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

