"""
Option A: 270M-class deep & narrow GPT pretrain

• L = 48 layers, D = 640, H = 8, KV heads = 2 (GQA), d_head = 80
• FFN: SwiGLU with d_ff = 1728 (≈ 8/3 · D, rounded to /64)
• Norms: Pre-RMSNorm everywhere; QK-Norm on Q and K
• RoPE: rotary_fraction = 0.5, configurable base (rope_base)
• Attention windows: local 512 on most layers; every 4th layer 1536
  (both warmed up from smaller windows)
• Dot-product scale: constant 0.12 (instead of 1/sqrt(d_head))
• Head: untied, logit softcap = 20
• Init: zero-init attn.c_proj and mlp.c_proj
• Precision: bf16; optional FP8-only LM head matmul (requires PyTorch with float8/scaled_mm support)
• Weights & Biases logging integrated

Dataset: a .bin file of token IDs (int32 or int16/uint16), contiguous, with vocab_size = 32768.

This script focuses on clarity + correctness. For large context (>=4k), 
consider replacing the banded attention mask with FlashAttention-2 local-window kernels for speed.
"""
from __future__ import annotations

import os
import math
import time
import argparse
import random
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# Utilities & helpers
# -------------------------------

def set_torch_defaults():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


def rmsnorm_lastdim(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # For QK-Norm per head: x[..., d]
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


# -------------------------------
# RoPE (partial rotary: apply to first fraction of head_dim)
# -------------------------------

def apply_rope(q: torch.Tensor, k: torch.Tensor, rope_fraction: float, base: float, seq_start: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q, k: (B, H, T, Dh)
    rope_fraction: fraction of Dh to apply rotary to (e.g., 0.5)
    base: rotary base frequency (e.g., 10000.)
    seq_start: offset for position ids (useful when using packed sequences)
    """
    B, H, T, Dh = q.shape
    rotary_dims = int(Dh * rope_fraction)
    if rotary_dims % 2 == 1:
        rotary_dims -= 1  # ensure even
    if rotary_dims <= 0:
        return q, k

    # positions [0..T-1] + offset
    pos = torch.arange(seq_start, seq_start + T, device=q.device, dtype=q.dtype)
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dims, 2, device=q.device, dtype=q.dtype) / rotary_dims))
    freqs = torch.outer(pos, inv_freq)  # (T, rotary_dims/2)
    # Keep sin/cos at shape (T, rotary_dims/2); they will broadcast over (B, H)
    sin = torch.sin(freqs)  # (T, rotary_dims/2)
    cos = torch.cos(freqs)  # (T, rotary_dims/2)

    def _rope(x: torch.Tensor) -> torch.Tensor:
        x_rot = x[..., :rotary_dims]
        x_pass = x[..., rotary_dims:]
        x1, x2 = x_rot[..., ::2], x_rot[..., 1::2]  # (..., T, rotary_dims/2)
        # Broadcast sin/cos (T, rotary_dims/2) across (B, H)
        x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return torch.cat([x_rotated, x_pass], dim=-1)

    return _rope(q), _rope(k)


# -------------------------------
# SwiGLU MLP
# -------------------------------
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.in_proj = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=False)
        # Zero-init final projection for stability in deep nets
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        a, b = x.chunk(2, dim=-1)
        return self.out_proj(a * F.silu(b))


# -------------------------------
# Multi-Head Attention with GQA, QK-Norm, constant scale, local windows
# -------------------------------
class MultiheadGQA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv: int,
        head_dim: int,
        rope_fraction: float = 0.5,
        rope_base: float = 10000.0,
        attn_scale: float = 0.12,
    ):
        super().__init__()
        assert num_heads % num_kv == 0, "num_heads must be divisible by num_kv for GQA"
        self.dim = dim
        self.h = num_heads
        self.kv = num_kv
        self.dh = head_dim
        self.groups = num_heads // num_kv  # e.g., 8/2 = 4
        self.rope_fraction = rope_fraction
        self.rope_base = rope_base
        self.attn_scale = attn_scale

        self.wq = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, num_kv * head_dim, bias=False)
        self.wv = nn.Linear(dim, num_kv * head_dim, bias=False)
        self.wo = nn.Linear(num_heads * head_dim, dim, bias=False)
        # Zero-init output projection
        nn.init.zeros_(self.wo.weight)

        self.qk_norm_eps = 1e-6

    def forward(self, x: torch.Tensor, seq_start: int, attn_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        device = x.device

        # Projections
        q = self.wq(x).view(B, T, self.h, self.dh).permute(0, 2, 1, 3)  # (B, H, T, Dh)
        k = self.wk(x).view(B, T, self.kv, self.dh).permute(0, 2, 1, 3)  # (B, KV, T, Dh)
        v = self.wv(x).view(B, T, self.kv, self.dh).permute(0, 2, 1, 3)  # (B, KV, T, Dh)

        # Broadcast K/V to H by grouping: each group of heads shares one KV head
        k = k.repeat_interleave(self.groups, dim=1)  # (B, H, T, Dh)
        v = v.repeat_interleave(self.groups, dim=1)  # (B, H, T, Dh)

        # QK-Norm per head
        q = rmsnorm_lastdim(q, self.qk_norm_eps)
        k = rmsnorm_lastdim(k, self.qk_norm_eps)

        # RoPE (partial)
        q, k = apply_rope(q, k, rope_fraction=self.rope_fraction, base=self.rope_base, seq_start=seq_start)

        # Keep 4D (B, H, T, Dh) to allow fused SDPA kernels when possible

        # Scaled dot-product attention with additive mask; prefer Flash/mem-efficient kernels
        # Note: we supply scale via the 'scale' kwarg; torch will still internally use 1/sqrt(d) unless scale is provided.
        try:
            # Prefer new sdpa_kernel context (PyTorch >= 2.4)
            with torch.nn.attention.sdpa_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False):
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, is_causal=False, scale=self.attn_scale
                )
        except Exception:
            # Fallback if the provided mask or environment is unsupported by flash/mem-efficient backends
            try:
                with torch.nn.attention.sdpa_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                    y = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_mask, is_causal=False, scale=self.attn_scale
                    )
            except Exception:
                # Ultimate fallback for older PyTorch
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                    y = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_mask, is_causal=False, scale=self.attn_scale
                    )
        # y is (B, H, T, Dh)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, self.h * self.dh)
        return self.wo(y)


# -------------------------------
# Transformer Block
# -------------------------------
class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv: int, head_dim: int, d_ff: int,
                 rope_fraction: float, rope_base: float, attn_scale: float):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiheadGQA(dim, num_heads, num_kv, head_dim, rope_fraction, rope_base, attn_scale)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, d_ff)

    def forward(self, x: torch.Tensor, seq_start: int, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), seq_start=seq_start, attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# -------------------------------
# Full Model
# -------------------------------
class GPT(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 640, n_layers: int = 48,
                 num_heads: int = 8, num_kv: int = 2, head_dim: int = 80,
                 d_ff: int = 1728, rope_fraction: float = 0.5, rope_base: float = 10000.0,
                 attn_scale: float = 0.12, softcap: float = 20.0, untied_head: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.softcap = softcap
        self.untied_head = untied_head

        self.embed = nn.Embedding(vocab_size, dim)
        self.in_norm = RMSNorm(dim)
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, num_kv, head_dim, d_ff, rope_fraction, rope_base, attn_scale) for _ in range(n_layers)
        ])
        self.out_norm = RMSNorm(dim)
        if untied_head:
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        else:
            self.lm_head = None  # tie via functional call

        # schedule state (updated from trainer)
        self.register_buffer("_layer_windows", torch.zeros(n_layers, dtype=torch.int32), persistent=False)

        # FP8 toggle (head only). Actual perf requires PyTorch with float8 E5M2 matmul support.
        self.use_fp8_head = False

        # Shared attention mask cache: {(T, window, device): Tensor}
        self._mask_cache = {}

    def _band_mask(self, T: int, window: int, device: torch.device) -> torch.Tensor:
        key = (T, window, device)
        if key not in self._mask_cache:
            i = torch.arange(T, device=device)
            j = torch.arange(T, device=device)
            m = (j[None, :] <= i[:, None]) & (j[None, :] >= (i[:, None] - (window - 1)))
            M = torch.zeros((T, T), device=device, dtype=torch.bfloat16)
            M.masked_fill_(~m, float('-inf'))
            self._mask_cache[key] = M
        return self._mask_cache[key]

    @torch.no_grad()
    def set_layer_windows(self, layer_windows: List[int]):
        assert len(layer_windows) == self.n_layers
        self._layer_windows = torch.tensor(layer_windows, dtype=torch.int32, device=self.embed.weight.device)

    def _softcap_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # Smoothly cap logits magnitude, Gemma-style; tanh variant
        c = self.softcap
        if c is None or c <= 0:
            return logits
        return c * torch.tanh(logits / c)

    def forward(self, idx: torch.Tensor, *, seq_start: int = 0) -> torch.Tensor:
        # idx: (B, T)
        x = self.embed(idx)
        x = self.in_norm(x)
        B, T, _ = x.shape
        device = x.device
        for li, block in enumerate(self.blocks):
            window = int(self._layer_windows[li].item())
            attn_mask = self._band_mask(T, window, device)
            x = block(x, seq_start=seq_start, attn_mask=attn_mask)
        x = self.out_norm(x)

        if self.lm_head is not None:
            if self.use_fp8_head and hasattr(torch, 'float8_e5m2'):
                # Experimental FP8 path: cast to float8, matmul in float16/32 via dequant (placeholder for scaled_mm APIs)
                # WARNING: This is a placeholder that may not give perf benefits without scaled_mm.
                x_f8 = x.to(torch.float8_e5m2)
                w_f8 = self.lm_head.weight.t().to(torch.float8_e5m2)  # (D, V)
                # Upcast back for matmul (emulating dequantize). Real impl would use scaled_mm.
                logits = (x_f8.to(torch.float16)) @ (w_f8.to(torch.float16))
            else:
                logits = F.linear(x, self.lm_head.weight)
        else:
            # tied head: use embedding weight
            logits = F.linear(x, self.embed.weight)

        logits = self._softcap_logits(logits)
        return logits


# -------------------------------
# Data pipeline: memory-mapped token stream -> (B, T) batches
# -------------------------------
class MemMapTokens(Dataset):
    def __init__(self, path: str, seq_len: int, dtype: str = 'uint16', split: str = 'train', split_ratio: float = 0.995):
        assert os.path.exists(path), f"Dataset not found: {path}"
        # Map string -> numpy dtype and validate file alignment
        dtype_map = {
            'uint16': np.uint16,
            'int16': np.int16,
            'int32': np.int32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype '{dtype}'. Choose from {list(dtype_map.keys())}")
        np_dtype = dtype_map[dtype]
        file_size = os.path.getsize(path)
        itemsize = np.dtype(np_dtype).itemsize
        if file_size % itemsize != 0:
            raise ValueError(
                f"File size {file_size} not divisible by element size {itemsize} for dtype '{dtype}'."
            )

        self.mm = np.memmap(path, mode='r', dtype=np_dtype)
        n_tokens = self.mm.shape[0]
        split_idx = int(n_tokens * split_ratio)
        if split == 'train':
            self.start, self.end = 0, split_idx
        else:
            self.start, self.end = split_idx, n_tokens
        self.seq_len = seq_len
        # We sample contiguous windows that fit entirely in [start, end)
        self.max_start = max(self.start, self.end - (self.seq_len + 1))
        # Ensure the split has enough tokens for one full sample
        if (self.end - self.start) < (self.seq_len + 1):
            raise ValueError(
                f"Split '{split}' too small for seq_len={self.seq_len}: "
                f"has {self.end - self.start} tokens, need at least {self.seq_len + 1}."
            )

    def __len__(self):
        # Roughly number of non-overlapping windows; we sample randomly anyway
        return max(1, (self.end - self.start) // (self.seq_len + 1))

    def __getitem__(self, _: int):
        s = random.randint(self.start, self.max_start)
        buf = self.mm[s : s + self.seq_len + 1]
        x = torch.from_numpy(np.array(buf[:-1], copy=False))
        y = torch.from_numpy(np.array(buf[1:], copy=False))
        return x.long(), y.long()


# -------------------------------
# Training
# -------------------------------
@dataclass
class TrainCfg:
    data_bin: str
    data_dtype: str = 'uint16'
    vocab_size: int = 32768
    seq_len: int = 2048
    global_steps: int = 200000
    batch_size: int = 8
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    min_lr_mult: float = 0.1  # cosine to 10% of peak
    grad_clip: float = 1.0
    eval_every: int = 1000
    eval_batches: int = 50
    log_every: int = 50
    save_every: int = 0  # set >0 to periodically save
    out_dir: str = './checkpoints'
    rope_base: float = 10000.0
    device: str = 'cuda'
    compile: bool = True
    use_wandb: bool = True
    wandb_project: str = 'optionA-270M'
    wandb_run_name: Optional[str] = None


@dataclass
class ModelCfg:
    dim: int = 640
    n_layers: int = 48
    num_heads: int = 8
    num_kv: int = 2
    head_dim: int = 80
    d_ff: int = 1728
    rope_fraction: float = 0.5
    rope_base: float = 10000.0
    attn_scale: float = 0.12
    softcap: float = 20.0
    untied_head: bool = True


def cosine_lr(step: int, warmup: int, total: int, base_lr: float, min_mult: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    mult = min_mult + 0.5 * (1 - min_mult) * (1 + math.cos(math.pi * p))
    return base_lr * mult


def window_schedule(step: int, warmup: int, short_target: int = 512, long_target: int = 1536,
                    short_start: int = 256, long_start: int = 512) -> Tuple[int, int]:
    if step >= warmup:
        return short_target, long_target
    # linear warmup
    def interp(a, b):
        return int(round(a + (b - a) * (step + 1) / max(1, warmup)))
    return interp(short_start, short_target), interp(long_start, long_target)


def layer_windows(n_layers: int, short_w: int, long_w: int) -> List[int]:
    wins = []
    for i in range(n_layers):
        # Every 4th layer (3,7,11,...) uses long window; others short
        wins.append(long_w if (i % 4 == 3) else short_w)
    return wins


def train(cfg: TrainCfg, mcfg: ModelCfg):
    set_torch_defaults()

    # Data
    train_ds = MemMapTokens(cfg.data_bin, cfg.seq_len, dtype=cfg.data_dtype, split='train')
    val_ds = MemMapTokens(cfg.data_bin, cfg.seq_len, dtype=cfg.data_dtype, split='val')
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    train_iter = iter(train_loader)

    # Model
    model = GPT(
        vocab_size=cfg.vocab_size,
        dim=mcfg.dim,
        n_layers=mcfg.n_layers,
        num_heads=mcfg.num_heads,
        num_kv=mcfg.num_kv,
        head_dim=mcfg.head_dim,
        d_ff=mcfg.d_ff,
        rope_fraction=mcfg.rope_fraction,
        rope_base=cfg.rope_base,
        attn_scale=mcfg.attn_scale,
        softcap=mcfg.softcap,
        untied_head=mcfg.untied_head,
    ).to(cfg.device)

    # Compile (PyTorch 2.0+ betterments)
    if cfg.compile and hasattr(torch, 'compile'):
        model = torch.compile(model, mode='max-autotune')

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)

    # Logging
    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config={**asdict(cfg), **asdict(mcfg), 'n_params': count_parameters(model)})

    # Training loop
    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tokens_seen = 0

    for step in range(cfg.global_steps):
        # Schedules
        lr = cosine_lr(step, cfg.warmup_steps, cfg.global_steps, cfg.lr, cfg.min_lr_mult)
        for pg in opt.param_groups:
            pg['lr'] = lr
        short_w, long_w = window_schedule(step, cfg.warmup_steps)
        wins = layer_windows(mcfg.n_layers, short_w, long_w)
        model.set_layer_windows(wins)

        # Batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)

        # Forward / loss
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x, seq_start=0)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1), reduction='mean')

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        tokens_seen += x.numel()

        # Logging
        if (step + 1) % cfg.log_every == 0:
            elapsed = time.time() - t0
            tps = tokens_seen / max(1e-9, elapsed)
            mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
            log = {
                'step': step + 1,
                'loss/train': loss.item(),
                'lr': lr,
                'tokens_per_sec': tps,
                'max_mem_gb': mem_gb,
                'short_window': short_w,
                'long_window': long_w,
            }
            if cfg.use_wandb:
                wandb.log(log, step=step + 1)
            else:
                print(log)

        # Eval
        if cfg.eval_every > 0 and (step + 1) % cfg.eval_every == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                it = iter(val_loader)
                for _ in range(cfg.eval_batches):
                    try:
                        vx, vy = next(it)
                    except StopIteration:
                        it = iter(val_loader)
                        vx, vy = next(it)
                    vx = vx.to(cfg.device, non_blocking=True)
                    vy = vy.to(cfg.device, non_blocking=True)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        vlogits = model(vx, seq_start=0)
                        vloss = F.cross_entropy(vlogits.reshape(-1, cfg.vocab_size), vy.reshape(-1), reduction='mean')
                    losses.append(vloss.item())
            val_loss = sum(losses) / len(losses)
            if cfg.use_wandb:
                wandb.log({'loss/val': val_loss}, step=step + 1)
            else:
                print({'step': step + 1, 'loss/val': val_loss})
            model.train()

        # Save
        if cfg.save_every and (step + 1) % cfg.save_every == 0:
            os.makedirs(cfg.out_dir, exist_ok=True)
            ckpt_path = os.path.join(cfg.out_dir, f"model_step{step+1}.pt")
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'cfg': asdict(cfg), 'mcfg': asdict(mcfg)}, ckpt_path)

    # Final save
    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.out_dir, f"model_final.pt")
    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'cfg': asdict(cfg), 'mcfg': asdict(mcfg)}, ckpt_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_bin', type=str, required=True, help='Path to contiguous token IDs (.bin)')
    parser.add_argument('--data_dtype', type=str, default='uint16', help="Storage dtype of token IDs in data_bin: 'uint16', 'int16', or 'int32'")
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--global_steps', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--rope_base', type=float, default=10000.0)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='optionA-270M')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()

    cfg = TrainCfg(
        data_bin=args.data_bin,
        data_dtype=args.data_dtype,
        seq_len=args.seq_len,
        global_steps=args.global_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        rope_base=args.rope_base,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        compile=args.compile,
    )

    mcfg = ModelCfg()

    train(cfg, mcfg)


if __name__ == '__main__':
    main()
