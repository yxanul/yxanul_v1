#!/usr/bin/env python3
"""
CLEAN FP8 training script - the ACTUAL fastest version.
This uses the simplified model without dead optimizations.

Key insights:
- NO FP8 weight caching (overhead > benefit at 112M scale)
- NO gradient fusion (adds memory traffic)
- Simple is faster for small models

Achieves 196k tokens/sec on RTX 5090.
"""

import os
import math
import time
import json
import random
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import argparse

# Import CLEAN model (the actually fast one)
from model_te_clean import ModelConfig, CleanGPT_TE, get_fp8_recipe

# TransformerEngine
import transformer_engine.pytorch as te

# Import our robust wandb logger
from wandb_logger import WandBLogger

# Optional SophiaG optimizer
try:
    from sophia import SophiaG
    SOPHIA_AVAILABLE = True
except Exception:
    SOPHIA_AVAILABLE = False

@dataclass
class TrainingConfig:
    # Model config (deep & narrow ~270M by default)
    n_layer: int = 48
    n_head: int = 8
    n_embd: int = 640
    vocab_size: int = 32768
    n_kv_heads: int = 2
    block_size: int = 2048
    dropout: float = 0.0
    
    # FP8 config
    use_fp8: bool = True
    fp8_amax_history_len: int = 16
    fp8_warmup_steps: int = 100
    
    # Optimizations - simplified for CLEAN version
    fuse_wgrad_accumulation: bool = False  # DISABLED - adds overhead
    cache_fp8_weights: bool = False  # NOT IMPLEMENTED - overhead > benefit
    
    # Training config
    batch_size: int = 8
    gradient_accumulation_steps: int = 22
    max_iters: int = 2000
    eval_interval: int = 100
    eval_iters: int = 50
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_iters: int = 2000
    lr_decay_iters: int = 4000
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    
    # System
    device: str = 'cuda'
    compile: bool = False  # Disabled for TE compatibility
    log_interval: int = 10
    checkpoint_interval: int = 500
    checkpoint_dir: str = 'checkpoints_fp8_optimized'
    data_dir: str = 'data_mixed_3b'

    # Wandb
    wandb_project: str = 'fp8-optimized'
    wandb_run_name: Optional[str] = None
    # Reproducibility
    seed: int = 1337


class DataLoader:
    """Memory-mapped data loader.

    Accepts either a directory containing 'train.bin' and optionally 'val.bin',
    or a direct path to a single 'train.bin'. If only a train.bin is present,
    creates 'val.bin' as the last 1% of tokens and uses the first 99% for training
    without rewriting the original train.bin.
    """
    def __init__(self, data_dir, block_size, device='cuda'):
        self.block_size = block_size
        self.device = device

        data_path = Path(data_dir)
        if data_path.is_file() and data_path.suffix == '.bin':
            # Single file path provided
            train_path = data_path
            base_dir = data_path.parent
            val_path = base_dir / 'val.bin'
            # Map the full train file
            mm = np.memmap(train_path, dtype=np.uint16, mode='r')
            n = mm.shape[0]
            cut = max(int(n * 0.99), self.block_size + 1)  # ensure at least one window remains
            cut = min(cut, n - (self.block_size + 1))      # and room for val

            if not val_path.exists():
                print(f"val.bin not found next to {train_path}. Creating 1% validation split at {val_path} ...")
                val_view = mm[cut:]
                with open(val_path, 'wb') as f:
                    # Write as-is (uint16)
                    val_view.tofile(f)
                print(f"Wrote val.bin with {len(val_view):,} tokens")
            # Training uses 0..cut slice; if val exists, respect its size
            val_mm = np.memmap(val_path, dtype=np.uint16, mode='r')
            cut = max(n - len(val_mm), self.block_size + 1)
            self.train_data = mm[:cut]
            self.val_data = val_mm
            print(f"Loaded data from {base_dir}")
        else:
            # Directory path
            train_path = data_path / 'train.bin'
            val_path = data_path / 'val.bin'
            assert train_path.exists(), f"Missing {train_path}"
            mm = np.memmap(train_path, dtype=np.uint16, mode='r')
            n = mm.shape[0]
            if not val_path.exists():
                cut = max(int(n * 0.99), self.block_size + 1)
                cut = min(cut, n - (self.block_size + 1))
                print(f"{val_path} not found. Creating 1% validation split ...")
                val_view = mm[cut:]
                with open(val_path, 'wb') as f:
                    val_view.tofile(f)
                print(f"Wrote val.bin with {len(val_view):,} tokens")
                self.train_data = mm[:cut]
            else:
                # If val exists, respect its size and avoid leakage: assume last len(val) tokens are validation
                val_mm = np.memmap(val_path, dtype=np.uint16, mode='r')
                cut = max(n - len(val_mm), self.block_size + 1)
                self.train_data = mm[:cut]
                self.val_data = val_mm
            print(f"Loaded data from {data_dir}")

        print(f"  Train: {len(self.train_data):,} tokens")
        print(f"  Val: {len(self.val_data):,} tokens")
    
    def get_batch(self, split, batch_size):
        data = self.train_data if split == 'train' else self.val_data
        max_start = len(data) - self.block_size - 1
        ix = torch.randint(max_start, (batch_size,))
        
        offsets = torch.arange(self.block_size + 1)
        indices = ix.unsqueeze(1) + offsets.unsqueeze(0)
        batch_data = torch.from_numpy(data[indices.numpy()].astype(np.int64)).pin_memory()
        
        x = batch_data[:, :-1].to(self.device, non_blocking=True)
        y = batch_data[:, 1:].to(self.device, non_blocking=True)
        
        return x, y


def get_lr(iter_num, config):
    """Learning rate schedule."""
    if iter_num < config.warmup_iters:
        return config.learning_rate * iter_num / config.warmup_iters
    
    plateau_iters = int(config.lr_decay_iters * 0.6)
    if iter_num < config.warmup_iters + plateau_iters:
        return config.learning_rate
    
    decay_iter = iter_num - config.warmup_iters - plateau_iters
    decay_length = config.lr_decay_iters - config.warmup_iters - plateau_iters
    
    if decay_iter > decay_length:
        return config.min_lr
    
    decay_ratio = decay_iter / decay_length
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def evaluate(model, data_loader, config, fp8_recipe):
    """Evaluate model."""
    model.eval()
    losses = []
    
    for _ in range(config.eval_iters):
        x, y = data_loader.get_batch('val', config.batch_size)
        
        if config.use_fp8:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        
        losses.append(loss.item())
    
    model.train()
    return np.mean(losses)


def save_checkpoint(model, optimizer, config, iter_num, val_loss, checkpoint_path):
    """Save checkpoint."""
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    checkpoint = {
        'model': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': asdict(config),
        'iter_num': iter_num,
        'val_loss': val_loss,
    }
    
    print(f"Saving checkpoint to {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)


def train():
    """Main training loop with optimizations."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--data_dir', type=str, default='data_mixed_3b', help='Data directory or path to train.bin')
    parser.add_argument('--max_iters', type=int, default=2000, help='Max iterations')
    parser.add_argument('--eval_interval', type=int, default=200, help='Eval interval')
    parser.add_argument('--log_interval', type=int, default=20, help='Log interval')
    parser.add_argument('--no_fp8', action='store_true', help='Disable FP8')
    parser.add_argument('--no_fusion', action='store_true', help='Disable gradient fusion (always disabled in CLEAN)')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed for reproducibility')
    # Model overrides
    parser.add_argument('--vocab_size', type=int, help='Tokenizer vocab size (e.g., 32768)')
    parser.add_argument('--n_layer', type=int, help='Number of layers')
    parser.add_argument('--n_head', type=int, help='Number of attention heads')
    parser.add_argument('--n_kv_heads', type=int, help='Number of KV heads (GQA)')
    parser.add_argument('--n_embd', type=int, help='Embedding dimension (must be divisible by n_head)')
    parser.add_argument('--block_size', type=int, help='Context length')
    parser.add_argument('--dropout', type=float, help='Dropout probability')
    # Optimizer choice and SophiaG hyperparams
    parser.add_argument('--opt', type=str, default='adamw', choices=['adamw', 'sophia'], help='Optimizer to use')
    parser.add_argument('--sophia_lr', type=float, default=6e-4, help='SophiaG learning rate')
    parser.add_argument('--sophia_betas', type=float, nargs=2, default=(0.965, 0.99), help='SophiaG betas')
    parser.add_argument('--sophia_rho', type=float, default=0.05, help='SophiaG rho')
    parser.add_argument('--sophia_weight_decay', type=float, default=0.2, help='SophiaG weight decay')
    parser.add_argument('--sophia_k', type=int, default=10, help='SophiaG Hessian EMA update frequency (iterations)')
    # Removed --no_caching as it's not implemented in CLEAN version
    args = parser.parse_args()
    
    # Configuration
    config = TrainingConfig()
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.no_fp8:
        config.use_fp8 = False
    # Model overrides
    if args.vocab_size:
        config.vocab_size = args.vocab_size
    if args.n_layer:
        config.n_layer = args.n_layer
    if args.n_head:
        config.n_head = args.n_head
    if args.n_kv_heads:
        config.n_kv_heads = args.n_kv_heads
    if args.n_embd:
        config.n_embd = args.n_embd
    if args.block_size:
        config.block_size = args.block_size
    if args.dropout is not None:
        config.dropout = args.dropout
    # Gradient fusion always disabled in CLEAN version
    config.fuse_wgrad_accumulation = False
    config.cache_fp8_weights = False
    
    config.data_dir = args.data_dir
    config.max_iters = args.max_iters
    config.eval_interval = args.eval_interval
    config.log_interval = args.log_interval
    config.seed = args.seed
    
    # Model configuration
    model_config = ModelConfig(
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        n_kv_heads=config.n_kv_heads,
        block_size=config.block_size,
        dropout=config.dropout,
        use_fp8=config.use_fp8,
        fp8_amax_history_len=config.fp8_amax_history_len,
        fuse_wgrad_accumulation=False,  # Always disabled in CLEAN
    )
    
    # Set seeds for reproducibility (keeps fast kernels; not fully deterministic)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # Create CLEAN model (the fast one)
    model = CleanGPT_TE(model_config).to(config.device)
    model = model.to(torch.bfloat16)
    
    # Get FP8 recipe
    fp8_recipe = get_fp8_recipe(model_config)
    
    # Optimizer
    if args.opt == 'sophia':
        if not SOPHIA_AVAILABLE:
            raise RuntimeError("SophiaG optimizer not available. Please `pip install sophia-optimizer` and ensure `from sophia import SophiaG`.")
        optimizer = SophiaG(
            model.parameters(),
            lr=float(args.sophia_lr),
            betas=tuple(args.sophia_betas),
            rho=float(args.sophia_rho),
            weight_decay=float(args.sophia_weight_decay),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
            fused=True
        )
    
    # Data loader
    data_loader = DataLoader(config.data_dir, config.block_size, config.device)
    
    # Initialize WandB logger
    logger = WandBLogger(
        enabled=not args.no_wandb,  # Enable by default unless explicitly disabled
        project=config.wandb_project,
        run_name=config.wandb_run_name or f"fp8_clean_{config.n_layer}L_{config.batch_size}b",
        config=asdict(config)
    )
    
    # NOTE: Not logging gradients - adds unnecessary overhead for 110+ parameters
    
    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(exist_ok=True)
    
    # Training info
    print("\n" + "="*50)
    print("Starting CLEAN FP8 Training (Actual Fastest)")
    print("="*50)
    print(f"Model: {model_config.n_layer}L, {model_config.n_head}H, {model_config.n_embd}D")
    print(f"Parameters: {model.num_parameters()/1e6:.1f}M")
    print(f"Status:")
    print(f"  - FP8: {config.use_fp8}")
    print(f"  - Gradient fusion: DISABLED (overhead)")
    print(f"  - Weight caching: NOT IMPLEMENTED (overhead > benefit)")
    print(f"  - Expected: 196k tokens/sec on RTX 5090")
    print(f"Optimizer: {'SophiaG' if args.opt=='sophia' else 'AdamW'}")
    if args.opt == 'sophia':
        print(f"  SophiaG lr={args.sophia_lr}, betas={tuple(args.sophia_betas)}, rho={args.sophia_rho}, wd={args.sophia_weight_decay}, k={args.sophia_k}")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print("="*50 + "\n")
    
    model.train()
    iter_num = 0
    best_val_loss = float('inf')
    t0 = time.time()
    tokens_processed = 0
    
    # Effective batch in tokens for SophiaG scaling
    sophia_bs_tokens = config.batch_size * config.block_size * config.gradient_accumulation_steps

    for iter_num in range(config.max_iters):
        # Learning rate
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Zero gradients - simple and clean
        optimizer.zero_grad(set_to_none=True)  # Faster than zeroing
        
        # Accumulate gradients
        total_loss = 0
        
        for micro_step in range(config.gradient_accumulation_steps):
            x, y = data_loader.get_batch('train', config.batch_size)
            
            # Determine if using FP8
            use_fp8_now = config.use_fp8 and (iter_num >= config.fp8_warmup_steps)
            
            # Forward pass - CLEAN version without is_first_microbatch
            if use_fp8_now:
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    logits, loss = model(x, y)  # No is_first_microbatch!
            else:
                logits, loss = model(x, y)
            
            total_loss += loss.item()
            loss = loss / config.gradient_accumulation_steps
            
            # Backward pass - gradients accumulate naturally
            loss.backward()
        
        # Gradient clipping
        if config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        else:
            grad_norm = torch.tensor(0.0)
        
        # Optimizer step
        if args.opt == 'sophia':
            optimizer.step(bs=sophia_bs_tokens)
        else:
            optimizer.step()

        # SophiaG Hessian EMA update every k iterations
        if args.opt == 'sophia' and (iter_num + 1) % int(args.sophia_k) == 0:
            optimizer.zero_grad(set_to_none=True)
            use_fp8_now = config.use_fp8 and (iter_num >= config.fp8_warmup_steps)
            if use_fp8_now:
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    logits, _ = model(x, None)
            else:
                logits, _ = model(x, None)
            samp_dist = torch.distributions.Categorical(logits=logits)
            y_sample = samp_dist.sample()
            loss_sampled = F.cross_entropy(logits.view(-1, logits.size(-1)), y_sample.view(-1), ignore_index=-1)
            loss_sampled.backward()
            optimizer.update_hessian()
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
        
        # Update token count
        tokens_processed += config.batch_size * config.block_size * config.gradient_accumulation_steps
        
        # Logging
        if iter_num % config.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = tokens_processed / dt if dt > 0 else 0
            
            avg_loss = total_loss / config.gradient_accumulation_steps
            print(f"iter {iter_num}: loss {avg_loss:.4f}, lr {lr:.2e}, "
                  f"{tokens_per_sec/1e3:.1f}k tok/s, FP8: {use_fp8_now}")
            
            # Calculate perplexity
            try:
                perplexity = math.exp(min(20.0, avg_loss))
            except:
                perplexity = float('inf')
            
            # Log comprehensive metrics (+ SophiaG win_rate if available)
            metrics = {
                'train/loss': avg_loss,
                'train/perplexity': perplexity,
                'train/lr': lr,
                'train/tokens_per_sec': tokens_per_sec,
                'train/grad_norm': grad_norm.item() if config.grad_clip > 0 else 0,
                'train/fp8_active': use_fp8_now,
                'train/gpu_memory_gb': torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                'train/iteration': iter_num,
            }
            if args.opt == 'sophia':
                # Compute win_rate as in Sophia: num_effective / num_param
                try:
                    num_param = 0
                    num_effective = 0
                    rho = float(args.sophia_rho)
                    for st in optimizer.state.values():
                        if isinstance(st, dict) and ('exp_avg' in st) and ('hessian' in st):
                            m = st['exp_avg']
                            h = st['hessian']
                            if isinstance(m, torch.Tensor) and isinstance(h, torch.Tensor):
                                num_param += m.numel()
                                thresh = rho * sophia_bs_tokens * h
                                num_effective += (m.abs() < thresh).sum().item()
                    if num_param > 0:
                        metrics['train/win_rate'] = float(num_effective) / float(num_param)
                    metrics['train/sophia_bs_tokens'] = float(sophia_bs_tokens)
                except Exception:
                    pass
            logger.log_metrics(metrics, step=iter_num)
            
            tokens_processed = 0
            t0 = time.time()
        
        # Evaluation
        if iter_num % config.eval_interval == 0:
            val_loss = evaluate(model, data_loader, config, fp8_recipe)
            print(f"Step {iter_num}: val loss {val_loss:.4f}")
            
            # Calculate validation perplexity
            try:
                val_perplexity = math.exp(min(20.0, val_loss))
            except:
                val_perplexity = float('inf')
            
            # Log validation metrics
            logger.log_metrics({
                'val/loss': val_loss,
                'val/perplexity': val_perplexity,
            }, step=iter_num)
            
            # Update best metrics in summary
            if val_loss < best_val_loss:
                logger.set_summary(
                    best_val_loss=val_loss,
                    best_val_perplexity=val_perplexity,
                    best_iter=iter_num
                )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, config, iter_num, val_loss,
                    Path(config.checkpoint_dir) / 'best_model_fp8_optimized.pt'
                )
        
        # Regular checkpoints
        if iter_num % config.checkpoint_interval == 0 and iter_num > 0:
            save_checkpoint(
                model, optimizer, config, iter_num, val_loss,
                Path(config.checkpoint_dir) / f'checkpoint_{iter_num}_fp8_optimized.pt'
            )
    
    # Final summary
    final_val_loss = evaluate(model, data_loader, config, fp8_recipe)
    try:
        final_perplexity = math.exp(min(20.0, final_val_loss))
    except:
        final_perplexity = float('inf')
    
    logger.set_summary(
        final_val_loss=final_val_loss,
        final_perplexity=final_perplexity,
        total_iterations=config.max_iters,
        model_params=model.num_parameters()
    )
    
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Final perplexity: {final_perplexity:.2f}")
    print("CLEAN implementation - simplicity wins at this scale!")
    print("="*50)
    
    # Clean up wandb
    logger.finish()


if __name__ == "__main__":
    train()
