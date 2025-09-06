"""
Deep & Narrow ~270M GPT (single GPU) based on KellerJordan/modded-nanogpt ideas
-------------------------------------------------------------------------------
- Single-GPU friendly
- 32k vocab (padded to multiple of 128)
- 38 layers, d_model=640, 5 heads x 128 head_dim (deep & narrow)
- QK RMSNorm, half-truncated RoPE, ReLU^2 MLP, zero-init projections
- Value Embeddings (2 tables) injected into V stream on early/late layers
- FlexAttention w/ sliding windows + long/short per-layer pattern (fallback to SDPA)
- Progressive window growth schedule (128 -> ... -> 1792 tokens)
- FP8-ish LM head (tries float8 if available; falls back to BF16 with fake quant)
- Muon-like optimizer for 2D weights (single GPU), AdamW for others
- torch.compile + warmup steps then state reset
- NEW: dataset-size aware step planning, best-on-val-loss checkpoints, Weights & Biases logging,
       deep-layer gradient norms and update-ratio monitoring

This is a compact educational implementation that mirrors the repo's "speed & power"
choices while remaining self-contained for one GPU. It aims for clarity and heavy
inline comments rather than absolute parity with the original kernels.

Tested with: PyTorch 2.3+ (BF16), CUDA 12, single H100/A100/4090 (BF16 on 4090 via emu)
Note: FlexAttention API is still evolving. If unavailable, we gracefully fall back to SDPA.
"""
from __future__ import annotations
import math
import os
import time
import copy
import types
import random
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

# -----------------------------------------------------------------------------
# Optional FlexAttention import (PyTorch 2.3+ nightlies/2.4)
# We use it if present; otherwise we fall back to SDPA with an additive mask.
try:
    from torch.nn.attention import flex_attention, BlockMask  # type: ignore
    HAS_FLEX = True
except Exception:
    flex_attention, BlockMask = None, None
    HAS_FLEX = False

# Optional float8 dtypes (PyTorch 2.3+). Not all GPUs support real float8 matmul.
HAS_F8 = hasattr(torch, 'float8_e4m3fn') and hasattr(torch, 'float8_e5m2')

# Optional Weights & Biases
try:
    import wandb  # type: ignore
    HAS_WANDB = True
except Exception:
    wandb = None
    HAS_WANDB = False

# -----------------------------------------------------------------------------
# Utilities & hyperparameters

@dataclass
class Config:
    # Model
    vocab_size: int = 32768
    pad_vocab_multiple: int = 128
    n_layers: int = 38           # must be even for U-Net-like skip stacking
    d_model: int = 640
    n_heads: int = 5             # head_dim fixed at 128 (like in the paper/codebase)
    head_dim: int = 128
    mlp_mult: int = 4
    value_tables: int = 2        # 2 instead of 3 to keep ~270M while preserving FP8 head
    max_seq_len_train: int = 16*1024
    max_seq_len_val: int = 32*1024

    # Training schedule
    steps: int = 1750            # if auto_steps_from_dataset=True, this will be replaced
    cooldown_frac: float = 0.45  # keep LR flat then cosine down to 10%
    warmup_muon_momentum: Tuple[float,float,int] = (0.85, 0.95, 300)
    auto_steps_from_dataset: bool = True    # Auto-calculate steps from dataset size
    dataset_tokens: int = 6_247_248_360     # Actual tokens in train.bin

    # LR & wd (mirrors the repo defaults conceptually)
    base_lr: float = 0.008  # not used anymore, keeping for compatibility
    muon_lr: float = 0.028
    weight_decay: float = 0.0
    betas: Tuple[float,float] = (0.8, 0.95)
    eps: float = 1e-10

    # Direct LR values for different parameter groups
    lr_head: float = 1/384  # ≈ 0.002604
    lr_embed: float = 0.22
    lr_value_embed: float = 0.22
    lr_scalar: float = 0.012

    # Attention windows (in tokens). Kept multiples of block_size.
    block_size: int = 128
    max_window_tokens: int = 1792   # final target window (tokens)
    short_window_tokens: int = 512  # short window (tokens)

    # Progressive window schedule: start small and increase
    window_start_tokens: int = 128
    window_grow_every: int = 150    # steps between window increases

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_on_val: bool = True
    save_every_eval: bool = True

    # Logging & monitoring
    seed: int = 1337
    device: str = 'cuda'
    dtype: torch.dtype = torch.bfloat16
    compile: bool = True
    grad_clip: float = 1.0
    eval_every: int = 200
    log_every: int = 50
    warmup_steps_compile: int = 10   # compile warmup steps

    # Weights & Biases
    wandb_project: Optional[str] = None  # e.g., "gpt-270m"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "disabled"        # "online", "offline", or "disabled"
    wandb_log_update_ratio: bool = True
    wandb_log_grads: bool = True
    deep_layer_monitor_frac: float = 0.33  # monitor top X fraction of layers for ratios/grad norms


def set_seed(seed: int):
    """Make runs deterministic-ish across GPU and data shuffling."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_vocab(vocab_size: int, multiple: int) -> int:
    """Pad vocab size up to a multiple for matmul efficiency (e.g., 128)."""
    if vocab_size % multiple == 0:
        return vocab_size
    return ((vocab_size + multiple - 1) // multiple) * multiple


# -----------------------------------------------------------------------------
# Norms & activations

class RMSNorm(nn.Module):
    """Root Mean Square Layer Norm (no mean-centering), numerically friendly in BF16.
    Normalizes last dimension; learnable scale gamma.
    """
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return x_norm * self.weight


def relu_squared(x: torch.Tensor) -> torch.Tensor:
    """ReLU^2 as used in the repo. Kernel-friendly and competitive with GELU."""
    return F.relu(x).square()


# -----------------------------------------------------------------------------
# Rotary embeddings (RoPE) with half-truncation

class RotaryCache:
    """Precompute cos/sin for RoPE up to a max sequence length.
    Half-truncation: only the first half of head_dim gets rotary; second half is passthrough.
    """
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        assert head_dim % 2 == 0, "head_dim should be even for RoPE"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        # Use half the dims for rotation
        rot_dims = head_dim // 2  # channels under rotation (e.g., 64)
        pair_dims = rot_dims // 2  # number of (even,odd) pairs (e.g., 32)
        
        inv_freq = 1.0 / (base ** (torch.arange(0, pair_dims, 1).float() / pair_dims))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', t, inv_freq)  # [T, pair_dims]
        
        self.cos_cached = freqs.cos()  # [T, pair_dims] - shape matches even/odd splits
        self.sin_cached = freqs.sin()  # [T, pair_dims]

    def to(self, device, dtype):
        self.cos_cached = self.cos_cached.to(device=device, dtype=dtype)
        self.sin_cached = self.sin_cached.to(device=device, dtype=dtype)
        return self


def apply_rope(q: torch.Tensor, k: torch.Tensor, rope: RotaryCache, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE on first half of channels; leave second half as is.
    q,k: [B, H, T, Dh]
    positions: [T] token positions (0..T-1)
    """
    B,H,T,Dh = q.shape
    rot = Dh // 2
    cos = rope.cos_cached.index_select(0, positions)  # [T, rot]
    sin = rope.sin_cached.index_select(0, positions)
    # Split channels
    q1, q2 = q[..., :rot], q[..., rot:]
    k1, k2 = k[..., :rot], k[..., rot:]
    # Rotate (x_even, x_odd) interleaved trick via view
    def rope_rotate(x):
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]
        x_rot_even = x_even * cos.unsqueeze(0).unsqueeze(0) - x_odd * sin.unsqueeze(0).unsqueeze(0)
        x_rot_odd  = x_even * sin.unsqueeze(0).unsqueeze(0) + x_odd * cos.unsqueeze(0).unsqueeze(0)
        x_out = torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)
        return x_out
    q1 = rope_rotate(q1)
    k1 = rope_rotate(k1)
    q = torch.cat([q1, q2], dim=-1)
    k = torch.cat([k1, k2], dim=-1)
    return q, k


# -----------------------------------------------------------------------------
# Linear layers with special init & optional FP8-ish matmul for LM head

class CastedLinear(nn.Module):
    """Bias-free Linear with tighter init (std ~ 0.5/sqrt(in_features)).
    Optionally attempts a float8 path (for LM head) and falls back safely.
    """
    def __init__(self, in_features: int, out_features: int, use_fp8: bool = False, zero_init: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.use_fp8 = use_fp8 and HAS_F8
        # Initialize
        if zero_init:
            nn.init.zeros_(self.weight)
        else:
            bound = 0.5 / math.sqrt(in_features)
            nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fp8:
            # Best-effort: cast inputs/weights to float8 types, matmul in float16/bfloat16 accumulator.
            # Real speedup requires custom kernels; here we emulate the quantize/dequantize pipeline.
            w = self.weight
            # Fake-quantize to float8 and back to dtype to keep numerics stable if matmul on f8 is unsupported
            try:
                w8 = w.to(torch.float8_e4m3fn)
                x8 = x.to(torch.float8_e4m3fn)
                out = (x8.float() @ w8.float().T).to(x.dtype)
            except Exception:
                out = F.linear(x, self.weight)  # safe fallback
            return out
        else:
            return F.linear(x, self.weight)


# -----------------------------------------------------------------------------
# Attention with merged QKV, QK RMSNorm, constant scale, FlexAttention/SDPA

class CausalSelfAttention(nn.Module):
    """Self-attention block with:
    - merged QKV projection
    - RMSNorm on Q and K
    - RoPE (half-truncated)
    - constant attention scaling (0.12) instead of 1/sqrt(d)
    - FlexAttention with sliding window (fallback to SDPA)
    - optional Value Embedding injection into V stream (scaled by learnable lambda)
    """
    def __init__(self, d_model: int, n_heads: int, head_dim: int, rope: RotaryCache, layer_idx: int,
                 long_layer: bool, value_embedder: Optional[nn.Module], use_value_embed: bool):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.hdim = n_heads * head_dim
        assert self.hdim == d_model, "(n_heads * head_dim) must equal d_model"
        self.layer_idx = layer_idx
        self.long_layer = long_layer
        self.value_embedder = value_embedder
        self.use_value_embed = use_value_embed

        # One fused weight for QKV: [3*d_model, d_model]
        self.w_qkv = CastedLinear(d_model, 3*d_model, use_fp8=False, zero_init=False)
        # Output projection, zero-initialized to stabilize residuals
        self.w_o = CastedLinear(d_model, d_model, use_fp8=False, zero_init=True)

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        # NO v_norm - reference doesn't normalize V
        self.rope = rope
        self.scale_const = 0.12  # empirical constant scale

        # Two lambdas for injecting value embeddings into V path (like reference)
        if self.use_value_embed and self.value_embedder is not None:
            # Initialize as [0.5, 0.5] like reference SA lambdas
            self.sa_lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        else:
            self.register_parameter('sa_lambdas', None)

    def _sliding_blockmask(self, T: int, window_tokens: int, block_size: int) -> Optional[BlockMask]:
        if not HAS_FLEX:
            return None
        # Build a simple causal sliding window mask at block granularity.
        # For educational clarity: we allow attending to the last `window_tokens` (including self) within the past.
        w_blocks = max(1, window_tokens // block_size)
        # BlockMask takes: num_queries, num_keys, block_size, layout function.
        def allow(q_block: int, k_block: int) -> bool:
            # Each block can see itself and previous w_blocks-1 blocks (causal). No future.
            return (k_block <= q_block) and (q_block - k_block) < w_blocks
        return BlockMask.make_causal(T, T, block_size=block_size, layout=allow)

    def forward(self, x: torch.Tensor, tokens: torch.Tensor, pos: torch.Tensor,
                window_tokens: int, block_size: int) -> torch.Tensor:
        B, T, C = x.shape
        assert B == 1, "This implementation is single-GPU B=1 by design."
        qkv = self.w_qkv(x)  # [B, T, 3*C]
        q, k, v = qkv.chunk(3, dim=-1)
        # Reshape to heads
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T,Dh]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # QK RMSNorm per head (NO V norm - reference doesn't do it)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE on first half channels
        q, k = apply_rope(q, k, self.rope, pos)

        # Optional Value Embedding injection with two lambdas (like reference)
        if self.use_value_embed and self.value_embedder is not None:
            ve = self.value_embedder(tokens)  # [B,T,D]
            ve = ve.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            # Use two lambdas like reference: v = lambdas[0] * v + lambdas[1] * ve
            v = self.sa_lambdas[0] * v + self.sa_lambdas[1] * ve
        else:
            # When no value embedding, scale v by first lambda (start at 0.5)
            v = v * 0.5

        # Attention computation: FlexAttention (if available) or SDPA fallback
        if HAS_FLEX:
            # Sliding window mask
            mask = self._sliding_blockmask(T, window_tokens, block_size)
            # FlexAttention wants score_mod, value_mod callables.
            def score_mod(scores: torch.Tensor) -> torch.Tensor:
                return scores * self.scale_const
            def value_mod(values: torch.Tensor) -> torch.Tensor:
                return values
            out = flex_attention(q, k, v, score_mod, value_mod, block_mask=mask)
        else:
            # Use SDPA with an additive sliding window causal mask
            device = x.device
            idx = torch.arange(T, device=device)
            # Allow attend to last window_tokens within past
            allowed = (idx.view(T, 1) - idx.view(1, T))
            # allowed >= 0 (no future) and allowed < window_tokens
            causal = (allowed >= 0) & (allowed < window_tokens)
            
            # IMPORTANT: keep mask in float32 to avoid bf16/-inf NaNs
            attn_mask = torch.empty((T, T), device=device, dtype=torch.float32)
            attn_mask[causal] = 0.0
            attn_mask[~causal] = float('-inf')
            
            # Pass the scale to SDPA (scale the *scores*, not the output)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=False, scale=self.scale_const
            )

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.w_o(out)
        return out


# -----------------------------------------------------------------------------
# MLP block (4x width) with ReLU^2 and zero-init projection

class MLP(nn.Module):
    def __init__(self, d_model: int, mult: int = 4):
        super().__init__()
        inner = mult * d_model
        self.fc1 = CastedLinear(d_model, inner, use_fp8=False, zero_init=False)
        self.fc2 = CastedLinear(inner, d_model, use_fp8=False, zero_init=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(relu_squared(self.fc1(x)))


# -----------------------------------------------------------------------------
# Value Embeddings (2 tables) and token embedding

class ValueEmbeddings(nn.Module):
    """Two separate embedding tables projected to model dim and summed.
    These are injected into the V stream of attention in early/late layers.
    """
    def __init__(self, vocab_size: int, d_model: int, n_tables: int = 2):
        super().__init__()
        self.tables = nn.ModuleList([nn.Embedding(vocab_size, d_model) for _ in range(n_tables)])
        # Slightly tighter uniform init
        bound = 0.5 / math.sqrt(d_model)
        for tab in self.tables:
            nn.init.uniform_(tab.weight, -bound, bound)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # Sum all tables (could be concatenated + linear proj; keeping simple)
        out = sum(tab(tokens) for tab in self.tables)
        return out


# -----------------------------------------------------------------------------
# Transformer block with pre-norm, U-Net style skip mixing, optional attn skip

class Block(nn.Module):
    def __init__(self, cfg: Config, rope: RotaryCache, layer_idx: int,
                 long_layer: bool, value_embedder: Optional[ValueEmbeddings], use_value_embed: bool,
                 skip_attention: bool = False):
        super().__init__()
        self.layer_idx = layer_idx
        self.skip_attention = skip_attention

        self.ln1 = RMSNorm(cfg.d_model)
        self.ln2 = RMSNorm(cfg.d_model)

        self.attn = CausalSelfAttention(
            cfg.d_model, cfg.n_heads, cfg.head_dim, rope,
            layer_idx, long_layer,
            value_embedder=value_embedder,
            use_value_embed=use_value_embed,
        )
        self.mlp = MLP(cfg.d_model, cfg.mlp_mult)

        # U-Net style learned skip from x0 (input embedding) at each layer
        self.skip_lambda = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, x0: torch.Tensor, tokens: torch.Tensor, pos: torch.Tensor,
                window_tokens: int, block_size: int) -> torch.Tensor:
        # pre-norm + attention (or skip)
        h = x + torch.tanh(self.skip_lambda) * x0
        if not self.skip_attention:
            h = h + self.attn(self.ln1(h), tokens, pos, window_tokens, block_size)
        # MLP
        h = h + self.mlp(self.ln2(h))
        return h


# -----------------------------------------------------------------------------
# GPT model tying it all together, with U-Net-like two-half structure

class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_layers % 2 == 0, "Use an even number of layers for U-Net-style stacking"
        self.cfg = cfg

        # Vocab padding for matmul friendliness
        self.vocab_size = pad_vocab(cfg.vocab_size, cfg.pad_vocab_multiple)

        self.embed = nn.Embedding(self.vocab_size, cfg.d_model)
        nn.init.uniform_(self.embed.weight, -0.5/math.sqrt(cfg.d_model), 0.5/math.sqrt(cfg.d_model))

        # Rotary cache up to max of train/val seq len
        max_T = max(cfg.max_seq_len_train, cfg.max_seq_len_val)
        self.rope = RotaryCache(cfg.head_dim, max_T)

        # Value embeddings (2 tables)
        self.value_emb = ValueEmbeddings(self.vocab_size, cfg.d_model, n_tables=cfg.value_tables)

        # Decide which layers are long-window vs short-window and where to use value embeddings
        long_layers = set()
        # One long layer every ~4 layers in each half
        half = cfg.n_layers // 2
        for i in range(0, half, 4):
            long_layers.add(i)
        for i in range(half, cfg.n_layers, 4):
            long_layers.add(i)

        use_value_layers = set()
        # Inject value embeddings in early 4 and last 4 layers
        for i in range(0, min(4, cfg.n_layers)):
            use_value_layers.add(i)
        for i in range(max(0, cfg.n_layers-4), cfg.n_layers):
            use_value_layers.add(i)

        # Optionally skip attention in a mid-layer (like repo's layer 7);
        # here we generalize: skip at ~1/4 depth
        skip_attn_layer = cfg.n_layers // 4

        self.blocks = nn.ModuleList([
            Block(cfg, self.rope, i,
                  long_layer=(i in long_layers),
                  value_embedder=self.value_emb,
                  use_value_embed=(i in use_value_layers),
                  skip_attention=(i == skip_attn_layer))
            for i in range(cfg.n_layers)
        ])

        self.ln_f = RMSNorm(cfg.d_model)
        # Separate LM head (untied) -- run in bf16 for stability during bring-up
        self.lm_head = CastedLinear(cfg.d_model, self.vocab_size, use_fp8=False, zero_init=True)

    def forward(self, tokens: torch.Tensor, step: int, train_mode: bool,
                window_tokens: int, block_size: int) -> torch.Tensor:
        """Forward pass.
        tokens: [B=1, T] int64
        step: training step (for scheduling); not used directly here
        window_tokens: current sliding window size (in tokens)
        """
        B, T = tokens.shape
        assert B == 1, "This implementation assumes batch size 1."

        device = tokens.device
        if (self.rope.cos_cached.device != device) or (self.rope.cos_cached.dtype != tokens.dtype):
            # Keep RoPE cache on correct device/dtype for speed
            self.rope.to(device=device, dtype=torch.float32)

        pos = torch.arange(T, device=device, dtype=torch.long)  # positions 0..T-1

        x = self.embed(tokens)
        x = self.ln_f(x)  # CRITICAL: normalize embeddings like in reference
        x0 = x  # save normalized input stream for U-Net skips

        for blk in self.blocks:
            x = blk(x, x0, tokens, pos, window_tokens, block_size)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        # Logit soft-capping (stabilize extreme logits)
        d = float(self.cfg.d_model)
        logits = 30.0 * torch.sigmoid(logits / (7.5 * math.sqrt(d)))
        return logits


# -----------------------------------------------------------------------------
# Loss helper (cross-entropy with shift by one)

def cross_entropy_shifted(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute token-level cross entropy.
    logits: [B,T,V], targets: [B,T]
    We drop the last logit and the first target to align for next-token prediction.
    """
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, V),
        targets[:, 1:].contiguous().view(-1)
    )
    return loss


# -----------------------------------------------------------------------------
# Simple streaming dataset from a uint16 token file or synthetic fallback

class TokenStream:
    """Memory-mapped uint16 token stream with BOS-aligned windows.
    The file is expected to contain raw uint16 token IDs (no header). If a file
    isn't provided, we synthesize dummy data for smoke tests.
    """
    def __init__(self, path: Optional[str], bos_token: int = 2, min_len: int = 1_000_000):
        if path and os.path.exists(path):
            arr = torch.from_file(path, dtype=torch.int16).to(torch.int32)
            self.tokens = arr.to(torch.int64)
        else:
            # synthetic data: alternating doc spans separated by BOS
            T = min_len
            toks = torch.randint(10, 32000, (T,), dtype=torch.int64)
            toks[::512] = bos_token
            self.tokens = toks
        self.bos = bos_token

    def sample(self, T: int) -> torch.Tensor:
        """Sample a contiguous segment [T] that starts at or shortly after a BOS.
        For simplicity we allow starting anywhere, but try to align roughly on BOS.
        """
        length = self.tokens.numel()
        if length <= T + 1:
            start = 0
        else:
            start = random.randint(0, length - T - 1)
        seg = self.tokens[start:start+T]
        return seg.clone()


# -----------------------------------------------------------------------------
# Optimizers: AdamW for embeddings/scalars/head; Muon-like for 2D matrices

class MuonSingleGPU(torch.optim.Optimizer):
    """A compact single-GPU Muon-like optimizer.
    - Maintains momentum buffer m.
    - Orthogonalizes the update for 2D weight matrices using a few Newton–Schulz iterations
      to approximate the zeroth power transform (improves conditioning).
    - Applies weight decay (decoupled-style if desired; here zero).
    Reference behavior, simplified for clarity and stability.
    """
    def __init__(self, params, lr=0.05, momentum=0.95, weight_decay=0.0, ns_iters: int = 5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_iters=ns_iters)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            mom = group['momentum']
            wd = group['weight_decay']
            ns_iters = group['ns_iters']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p, alpha=wd)
                state = self.state[p]
                if 'm' not in state:
                    state['m'] = torch.zeros_like(p)
                m = state['m']
                m.mul_(mom).add_(g, alpha=1.0 - mom)

                # Only orthogonalize updates for 2D tensors (matrices)
                upd = m
                if p.ndim == 2:
                    # Newton–Schulz for (X X^T)^(-1/2) * X  ~ zeroth-power transform
                    X = upd
                    # Normalize by Frobenius to improve stability
                    frob = torch.norm(X, p='fro') + 1e-9
                    Y = X / frob
                    I = torch.eye(Y.shape[0], device=Y.device, dtype=Y.dtype)
                    Z = I.clone()
                    # Iterations (5 is a good compromise on BF16)
                    for _ in range(ns_iters):
                        Tm = 0.5 * (3*I - Z @ (Y @ Y.transpose(0,1)) @ Z)
                        Y = Tm @ Y  # Left-multiply for correct dimensions
                        Z = Tm @ Z
                    upd = Y * frob  # rescale back

                p.add_(upd, alpha=-lr)
        return loss


# -----------------------------------------------------------------------------
# Training utilities

def param_groups_for_optim(model: GPT, cfg: Config):
    """Split parameters into groups with explicit learning rates.
    - Muon on 2D hidden matrices inside blocks
    - AdamW on embeddings, scalars, and LM head with specific LRs
    """
    muon_params = []
    head_params = []
    embed_params = []
    value_embed_params = []
    scalar_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Identify groups by structure/name
        if p.ndim >= 2 and ('blocks' in name) and ('embed' not in name):
            muon_params.append(p)
        elif 'lm_head' in name:
            head_params.append(p)
        elif 'value_emb' in name:
            value_embed_params.append(p)
        elif 'embed' in name:
            embed_params.append(p)
        elif p.ndim < 2:
            scalar_params.append(p)
        else:
            # Default: treat as scalar params for any unmatched params
            scalar_params.append(p)

    # Build optimizers
    opt_muon = MuonSingleGPU(muon_params, lr=cfg.muon_lr, momentum=cfg.warmup_muon_momentum[0], weight_decay=cfg.weight_decay)
    
    # AdamW groups with explicit LRs
    adam_param_groups = [
        {'params': head_params, 'lr': cfg.lr_head, 'betas': cfg.betas, 'eps': cfg.eps, 'weight_decay': cfg.weight_decay},
        {'params': embed_params, 'lr': cfg.lr_embed, 'betas': cfg.betas, 'eps': cfg.eps, 'weight_decay': cfg.weight_decay},
        {'params': value_embed_params, 'lr': cfg.lr_value_embed, 'betas': cfg.betas, 'eps': cfg.eps, 'weight_decay': cfg.weight_decay},
        {'params': scalar_params, 'lr': cfg.lr_scalar, 'betas': cfg.betas, 'eps': cfg.eps, 'weight_decay': cfg.weight_decay},
    ]
    # Filter out empty groups
    adam_param_groups = [g for g in adam_param_groups if g['params']]
    opt_adam = torch.optim.AdamW(adam_param_groups)
    return opt_muon, opt_adam


def current_window(step: int, cfg: Config) -> int:
    """Progressively grow the sliding window to max_window_tokens.
    Increases every `window_grow_every` steps by block_size increments.
    """
    incs = max(0, step // cfg.window_grow_every)
    win = cfg.window_start_tokens + incs * cfg.block_size
    return int(min(win, cfg.max_window_tokens))


def cosine_cooldown_lr(step: int, total_steps: int, cooldown_frac: float, warmup_steps: int = 200) -> float:
    """Linear warmup, then flat LR, then cosine decay to 10% of base."""
    # Warmup phase
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    
    # Flat phase
    cutoff = int((1.0 - cooldown_frac) * total_steps)
    if step < cutoff:
        return 1.0
    
    # Cosine cooldown from 1.0 down to 0.1
    t = (step - cutoff) / max(1, (total_steps - cutoff))
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))


def save_state(model: GPT, opt_muon, opt_adam):
    return {
        'model': copy.deepcopy(model.state_dict()),
        'muon': copy.deepcopy(opt_muon.state_dict()),
        'adam': copy.deepcopy(opt_adam.state_dict()),
    }


def load_state(model: GPT, opt_muon, opt_adam, state):
    model.load_state_dict(state['model'])
    opt_muon.load_state_dict(state['muon'])
    opt_adam.load_state_dict(state['adam'])


# ---- Monitoring helpers ------------------------------------------------------

def layers_to_monitor(n_layers: int, frac: float) -> List[int]:
    k = max(1, int(math.ceil(n_layers * frac)))
    start = n_layers - k
    return list(range(start, n_layers))


def snapshot_params(model: GPT, layer_ids: List[int]) -> Dict[str, torch.Tensor]:
    """Return a {name: clone()} snapshot for parameters in given layers."""
    wanted = tuple(f"blocks.{i}." for i in layer_ids)
    snap = {}
    for n, p in model.named_parameters():
        if p.requires_grad and any(n.startswith(w) for w in wanted):
            snap[n] = p.data.detach().clone()
    return snap


def per_layer_grad_norms(model: GPT, layer_ids: List[int]) -> Dict[int, float]:
    """Compute L2 grad norm per selected layer (sum over params, sqrt at end)."""
    wanted = {i: f"blocks.{i}." for i in layer_ids}
    sq_sums: Dict[int, float] = {i: 0.0 for i in layer_ids}
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        for i, prefix in wanted.items():
            if n.startswith(prefix):
                g = p.grad
                sq_sums[i] += float(g.pow(2).sum().item())
                break
    return {i: (sq_sums[i] ** 0.5) for i in layer_ids}


def per_layer_update_ratio(model: GPT, prev: Dict[str, torch.Tensor], layer_ids: List[int]) -> Dict[int, float]:
    """Compute ||Δθ||/||θ|| per layer, aggregated over its params.
    Uses max(prev_norm, curr_norm) + eps to avoid div-by-zero with zero-inits."""
    eps = 1e-12
    num_sq = {i: 0.0 for i in layer_ids}
    den_sq = {i: 0.0 for i in layer_ids}
    
    for n, p in model.named_parameters():
        if n not in prev:
            continue
        # infer layer id from name
        try:
            i = int(n.split('.')[1])
        except Exception:
            continue
        if i not in num_sq:
            continue
        
        cur = p.data
        delta = cur - prev[n]
        num_sq[i] += float(delta.pow(2).sum().item())
        
        # Robust denominator: max of prev and current norms
        prev_norm_sq = float(prev[n].pow(2).sum().item())
        curr_norm_sq = float(cur.pow(2).sum().item())
        den_sq[i] += max(prev_norm_sq, curr_norm_sq)
        
        # refresh snapshot in-place
        prev[n] = cur.detach().clone()
    
    # Finalize with epsilon guard
    ratios = {}
    for i in layer_ids:
        num = (num_sq[i] + eps) ** 0.5
        den = (den_sq[i] + eps) ** 0.5
        ratios[i] = num / den
    return ratios


def save_checkpoint(path: str, model: GPT, opt_muon, opt_adam, cfg: Config, step: int, best_val: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'step': step,
        'best_val': best_val,
        'model': model.state_dict(),
        'muon': opt_muon.state_dict(),
        'adam': opt_adam.state_dict(),
        'config': asdict(cfg),
    }, path)


# -----------------------------------------------------------------------------
# Main training loop (single GPU)

def train(cfg: Config, data_path: Optional[str] = None):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Build model
    model = GPT(cfg).to(device)
    model = model.to(dtype=cfg.dtype)
    
    # CRITICAL: Cast embeddings to bfloat16 (like reference)
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.to(dtype=torch.bfloat16)

    # Try compile for fused kernels / better scheduling
    if cfg.compile:
        model = torch.compile(model, mode='default', dynamic=False)

    # Prepare optimizers
    opt_muon, opt_adam = param_groups_for_optim(model, cfg)

    # Data
    stream = TokenStream(data_path)

    # Figure out total steps if dataset-driven
    steps_total = cfg.steps
    if cfg.auto_steps_from_dataset and cfg.dataset_tokens > 0:
        steps_total = int(math.ceil(cfg.dataset_tokens / cfg.max_seq_len_train))
        print(f"[plan] auto steps from dataset_tokens={cfg.dataset_tokens:,} and train_seq_len={cfg.max_seq_len_train:,} -> steps_total={steps_total:,}")

    # Warmup compile: run a few steps and then reset state so timing/metrics are fair
    init_state = save_state(model, opt_muon, opt_adam)

    # Deep-layer monitoring setup
    monitor_layers = layers_to_monitor(cfg.n_layers, cfg.deep_layer_monitor_frac)
    prev_snap = snapshot_params(model, monitor_layers)

    # Optional W&B
    if HAS_WANDB and cfg.wandb_project is not None and cfg.wandb_mode != "disabled":
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, mode=cfg.wandb_mode, config=asdict(cfg))
        # do not enable wandb.watch by default (can be slow); we log custom metrics instead

    def step_once(step: int, tokens: torch.Tensor, train_mode: bool):
        model.train(train_mode)
        tokens = tokens.to(device)
        window_tokens = current_window(step, cfg)
        grad_total_norm = None
        layer_grad_norms: Dict[int, float] = {}
        with torch.autocast(device_type='cuda', dtype=cfg.dtype):
            logits = model(tokens, step, train_mode, window_tokens, cfg.block_size)
            loss = cross_entropy_shifted(logits, tokens)
        if train_mode:
            opt_muon.zero_grad(set_to_none=True)
            opt_adam.zero_grad(set_to_none=True)
            loss.backward()
            # metrics BEFORE clipping
            if cfg.wandb_log_grads:
                layer_grad_norms = per_layer_grad_norms(model, monitor_layers)
            grad_total_norm = float(clip_grad_norm_(model.parameters(), cfg.grad_clip).item())
            opt_muon.step()
            opt_adam.step()
        return loss.detach().item(), grad_total_norm, layer_grad_norms

    # Warmup (compile kernels)
    for ws in range(cfg.warmup_steps_compile):
        toks = stream.sample(cfg.max_seq_len_train).unsqueeze(0)
        _ = step_once(ws, toks, train_mode=True)

    # Reset to initial state
    load_state(model, opt_muon, opt_adam, init_state)

    # Training proper
    t0 = time.time()
    best_val = float('inf')

    for step in range(steps_total):
        toks = stream.sample(cfg.max_seq_len_train).unsqueeze(0)
        loss, grad_total_norm, layer_grad_norms = step_once(step, toks, train_mode=True)

        # Muon momentum warmup
        m0, m1, m_steps = cfg.warmup_muon_momentum
        if step < m_steps:
            cur_m = m0 + (m1 - m0) * (step / max(1, m_steps))
            for g in opt_muon.param_groups:
                g['momentum'] = cur_m

        # Cosine cooldown on LR (both optimizers). We apply LR scaling by replacing their lr each step.
        lr_scale = cosine_cooldown_lr(step, steps_total, cfg.cooldown_frac)
        for g in opt_muon.param_groups:
            g['lr'] = cfg.muon_lr * lr_scale
        # AdamW groups already baked with lr_mul; scale all by lr_scale
        for g in opt_adam.param_groups:
            base_lr_group = g.get('_base_lr', None)
            if base_lr_group is None:
                g['_base_lr'] = g['lr']
                base_lr_group = g['lr']
            g['lr'] = base_lr_group * lr_scale

        # Compute update ratios AFTER step
        update_ratios = {}
        if cfg.wandb_log_update_ratio:
            update_ratios = per_layer_update_ratio(model, prev_snap, monitor_layers)

        # Logging
        if (step + 1) % cfg.log_every == 0:
            dt = time.time() - t0
            toks_per_step = cfg.max_seq_len_train
            tok_rate = (cfg.log_every * toks_per_step) / max(1e-6, dt)
            msg = f"step {step+1:5d}/{steps_total} | loss {loss:.3f} | {tok_rate:.0f} tok/s | window {current_window(step, cfg)}"
            if grad_total_norm is not None:
                msg += f" | grad_norm {grad_total_norm:.2f}"
            print(msg)
            if HAS_WANDB and cfg.wandb_project is not None and cfg.wandb_mode != "disabled":
                log_dict = {
                    'train/loss': loss,
                    'train/tok_per_s': tok_rate,
                    'train/lr_scale': lr_scale,
                }
                if grad_total_norm is not None:
                    log_dict['train/grad_norm'] = grad_total_norm
                # per-layer metrics (top layers)
                for i, val in layer_grad_norms.items():
                    log_dict[f'train/grad_norm_layer/{i}'] = val
                for i, val in update_ratios.items():
                    log_dict[f'train/update_ratio_layer/{i}'] = val
                wandb.log(log_dict, step=step+1)
            t0 = time.time()

        # Validation & checkpoints
        if (step + 1) % cfg.eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(4):
                    vtoks = stream.sample(cfg.max_seq_len_val).unsqueeze(0).to(device)
                    with torch.autocast(device_type='cuda', dtype=cfg.dtype):
                        v_logits = model(vtoks, step, False, current_window(step, cfg), cfg.block_size)
                        v_loss = cross_entropy_shifted(v_logits, vtoks)
                    val_losses.append(v_loss.item())
            mv = sum(val_losses) / len(val_losses)
            improved = mv < best_val
            best_val = min(best_val, mv)
            print(f"[eval] step {step+1} | val_loss {mv:.3f} | best {best_val:.3f}{' *' if improved else ''}")

            if HAS_WANDB and cfg.wandb_project is not None and cfg.wandb_mode != "disabled":
                wandb.log({'eval/val_loss': mv, 'eval/best_val': best_val}, step=step+1)

            # Save checkpoints
            if cfg.save_every_eval:
                save_checkpoint(os.path.join(cfg.checkpoint_dir, 'last.pt'), model, opt_muon, opt_adam, cfg, step+1, best_val)
            if cfg.save_best_on_val and improved:
                save_checkpoint(os.path.join(cfg.checkpoint_dir, 'best.pt'), model, opt_muon, opt_adam, cfg, step+1, best_val)

    print("Training complete.")


# -----------------------------------------------------------------------------
# Entry

if __name__ == '__main__':
    cfg = Config()
    # Tip: to plan steps for a 5.4B-token dataset at 16k T, set
    #   cfg.auto_steps_from_dataset = True
    # and adjust cfg.dataset_tokens accordingly.
    # Optional: point to a uint16 token file (raw) via env var
    data_path = os.environ.get('TOKENS_BIN', None)
    train(cfg, data_path)
