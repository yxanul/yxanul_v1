#!/usr/bin/env python3
"""
Simple trainer for te_gpt_clean.GPT
"""
import os
import time
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from te_gpt_clean import GPT, ModelCfg, set_torch_defaults, count_params


class BinTokens(Dataset):
    def __init__(self, path: str, seq_len: int, dtype: str = 'uint16', split_ratio: float = 0.995, split: str = 'train'):
        assert os.path.exists(path)
        dmap = {'uint16': np.uint16, 'int16': np.int16, 'int32': np.int32}
        assert dtype in dmap
        self.mm = np.memmap(path, mode='r', dtype=dmap[dtype])
        n = self.mm.shape[0]
        cut = int(n * split_ratio)
        if split == 'train':
            self.lo, self.hi = 0, cut
        else:
            self.lo, self.hi = cut, n
        self.seq_len = seq_len
        if (self.hi - self.lo) < (seq_len + 1):
            raise ValueError('split too small')
        self.max_start = self.hi - (seq_len + 1)

    def __len__(self):
        return max(1, (self.hi - self.lo) // (self.seq_len + 1))

    def __getitem__(self, _):
        s = np.random.randint(self.lo, self.max_start + 1)
        buf = self.mm[s: s + self.seq_len + 1]
        x = torch.from_numpy(np.array(buf[:-1], copy=True)).long()
        y = torch.from_numpy(np.array(buf[1:], copy=True)).long()
        return x, y


@dataclass
class TrainCfg:
    data_bin: str
    data_dtype: str = 'uint16'
    seq_len: int = 2048
    vocab_size: int = 32768
    batch_size: int = 4
    steps: int = 200000
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    warmup: int = 2000
    min_lr_mult: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 50
    eval_every: int = 0
    eval_batches: int = 50
    out_dir: str = './checkpoints_te'
    device: str = 'cuda'
    use_wandb: bool = False
    wandb_project: str = 'te-gpt'
    wandb_run_name: str = None
    compile: bool = False
    timing: bool = True


def cosine_lr(step: int, warmup: int, total: int, base_lr: float, min_mult: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    mult = min_mult + 0.5 * (1 - min_mult) * (1 + math.cos(math.pi * p))
    return base_lr * mult


def train(tcfg: TrainCfg, mcfg: ModelCfg):
    set_torch_defaults()

    train_ds = BinTokens(tcfg.data_bin, tcfg.seq_len, dtype=tcfg.data_dtype, split='train')
    val_ds = BinTokens(tcfg.data_bin, tcfg.seq_len, dtype=tcfg.data_dtype, split='val')
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    it = iter(train_loader)

    model = GPT(mcfg).to(tcfg.device)
    if tcfg.compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='max-autotune')
        except Exception:
            pass
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, betas=tcfg.betas, eps=tcfg.eps, weight_decay=tcfg.weight_decay)

    if tcfg.use_wandb:
        import wandb
        wandb.init(project=tcfg.wandb_project, name=tcfg.wandb_run_name, config={**asdict(tcfg), **asdict(mcfg), 'n_params': count_params(model)})

    model.train()
    torch.cuda.reset_peak_memory_stats()
    tokens_seen = 0
    t0 = time.time()

    fetch_t = fwd_t = bwd_t = opt_t = 0.0
    for step in range(tcfg.steps):
        lr = cosine_lr(step, tcfg.warmup, tcfg.steps, tcfg.lr, tcfg.min_lr_mult)
        for pg in opt.param_groups:
            pg['lr'] = lr

        t_fetch0 = time.time()
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        fetch_t += (time.time() - t_fetch0)
        x = x.to(tcfg.device, non_blocking=True)
        y = y.to(tcfg.device, non_blocking=True)

        t_fwd0 = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, tcfg.vocab_size), y.view(-1), reduction='mean')
        if tcfg.timing:
            torch.cuda.synchronize()
        fwd_t += (time.time() - t_fwd0)

        opt.zero_grad(set_to_none=True)
        t_bwd0 = time.time()
        loss.backward()
        if tcfg.grad_clip and tcfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        if tcfg.timing:
            torch.cuda.synchronize()
        bwd_t += (time.time() - t_bwd0)

        t_opt0 = time.time()
        opt.step()
        if tcfg.timing:
            torch.cuda.synchronize()
        opt_t += (time.time() - t_opt0)

        tokens_seen += x.numel()

        if (step + 1) % tcfg.log_every == 0:
            elapsed = time.time() - t0
            tps = tokens_seen / max(1e-9, elapsed)
            mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            log = {
                'step': step + 1,
                'loss/train': loss.item(),
                'lr': lr,
                'tokens_per_sec': tps,
                'max_mem_gb': mem_gb,
            }
            if tcfg.use_wandb:
                import wandb
                wandb.log(log, step=step + 1)
            print(log)
            if tcfg.timing:
                steps = tcfg.log_every
                timing = {
                    'timing/fetch_s': fetch_t / steps,
                    'timing/forward_s': fwd_t / steps,
                    'timing/backward_s': bwd_t / steps,
                    'timing/opt_s': opt_t / steps,
                }
                if tcfg.use_wandb:
                    wandb.log(timing, step=step + 1)
                print(timing)
                fetch_t = fwd_t = bwd_t = opt_t = 0.0

        if tcfg.eval_every and (step + 1) % tcfg.eval_every == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                vit = iter(val_loader)
                for _ in range(tcfg.eval_batches):
                    try:
                        vx, vy = next(vit)
                    except StopIteration:
                        break
                    vx = vx.to(tcfg.device, non_blocking=True)
                    vy = vy.to(tcfg.device, non_blocking=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        vlogits = model(vx)
                        vloss = F.cross_entropy(vlogits.view(-1, tcfg.vocab_size), vy.view(-1), reduction='mean')
                    losses.append(vloss.item())
            val_loss = sum(losses) / max(1, len(losses))
            if tcfg.use_wandb:
                import wandb
                wandb.log({'loss/val': val_loss}, step=step + 1)
            print({'step': step + 1, 'loss/val': val_loss})
            model.train()

    os.makedirs(tcfg.out_dir, exist_ok=True)
    ckpt = {
        'model': model.state_dict(),
        'tcfg': asdict(tcfg),
        'mcfg': asdict(mcfg),
    }
    torch.save(ckpt, os.path.join(tcfg.out_dir, 'model_final.pt'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_bin', type=str, required=True)
    p.add_argument('--data_dtype', type=str, default='uint16')
    p.add_argument('--seq_len', type=int, default=2048)
    p.add_argument('--vocab_size', type=int, default=32768)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--steps', type=int, default=200000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup', type=int, default=2000)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--use_wandb', action='store_true')
    p.add_argument('--wandb_project', type=str, default='te-gpt')
    p.add_argument('--wandb_run_name', type=str, default=None)

    # Model knobs
    p.add_argument('--n_layer', type=int, default=48)
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--n_kv_head', type=int, default=2)
    p.add_argument('--n_embd', type=int, default=640)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--rope_theta', type=float, default=10000.0)
    p.add_argument('--rope_fraction', type=float, default=1.0)
    p.add_argument('--use_qk_norm', action='store_true')
    p.add_argument('--attn_window', type=int, default=0)
    p.add_argument('--attn_chunk', type=int, default=512)
    p.add_argument('--use_fp8', action='store_true')

    args = p.parse_args()

    mcfg = ModelCfg(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
        block_size=args.seq_len,
        dropout=args.dropout,
        rope_theta=args.rope_theta,
        rope_fraction=args.rope_fraction,
        use_qk_norm=args.use_qk_norm,
        attn_window=args.attn_window,
        attn_chunk=args.attn_chunk,
        use_fp8=args.use_fp8,
    )
    tcfg = TrainCfg(
        data_bin=args.data_bin,
        data_dtype=args.data_dtype,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        warmup=args.warmup,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        compile=args.compile,
    )

    train(tcfg, mcfg)


if __name__ == '__main__':
    main()

