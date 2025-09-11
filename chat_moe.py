#!/usr/bin/env python3
"""
Interactive chat for TinyMoETransformer checkpoints.

Loads checkpoints_moe_bf16/best_moe_bf16.pt, restores the model, and
provides a simple CLI chat. For each user turn it will generate K
candidate responses (default K=5) with configurable temperature, top-k,
and top-p (nucleus) sampling. Also applies a mild repetition penalty.

Requirements:
- transformers (for tokenizer)
- torch

Example:
  python chat_moe.py \
    --ckpt checkpoints_moe_bf16/best_moe_bf16.pt \
    --tokenizer mixed_dataset_hf_tokenizer \
    --device cuda --k 5 --temperature 0.7 --max_new_tokens 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer

from model_experimental import TinyMoETransformer


def load_model(ckpt_path: str, device: str = 'cuda') -> tuple[TinyMoETransformer, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get('cfg', {})

    # Instantiate model from saved config (fall back to sensible defaults)
    model = TinyMoETransformer(
        vocab_size=cfg.get('vocab_size', 32768),
        n_layer=cfg.get('n_layer', 10),
        n_head=cfg.get('n_head', 8),
        d_model=cfg.get('d_model', 512),
        block_size=cfg.get('block_size', 2048),
        dropout=cfg.get('dropout', 0.0),
        bias=cfg.get('bias', False),
        n_experts=cfg.get('n_experts', 4),
        capacity_factor=1.0,              # inference: irrelevant when dropless=True
        dropless=True,                    # force dropless at inference to avoid any token drops
        load_balance_alpha=cfg.get('load_balance_alpha', 0.05),
        router_z_loss_coef=cfg.get('router_z_loss_coef', 0.0),
        attn_gate=cfg.get('attn_gate', 'none'),
        use_rope=cfg.get('use_rope', True),
        rope_theta=cfg.get('rope_theta', 10000.0),
        moe_grouped=cfg.get('moe_grouped', False),
    ).to(device=device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    # Ensure router has deterministic, non-noisy behavior at inference
    for blk in model.h:
        if hasattr(blk, 'moe'):
            blk.moe.router.set_router_state(temperature=1.0, noise_std=0.0)
    return model, {
        'iter': ckpt.get('iter', None),
        'val_loss': ckpt.get('val_loss', None),
        'cfg': cfg,
    }


def build_prompt(history: List[dict], system: str | None = None) -> str:
    parts: List[str] = []
    if system:
        parts.append(f"System: {system}\n")
    for turn in history:
        role = turn.get('role', 'user')
        content = turn.get('content', '')
        if role == 'user':
            parts.append(f"User: {content}\n")
        else:
            parts.append(f"Assistant: {content}\n")
    parts.append("Assistant:")
    return ''.join(parts)


def _apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> None:
    if penalty == 1.0 or generated.numel() == 0:
        return
    # In-place repetition penalty (CTRL/GPT-2 style)
    unique_tokens = torch.unique(generated)
    logits_unique = logits[..., unique_tokens]
    neg_mask = logits_unique < 0
    logits_unique[neg_mask] *= penalty
    logits_unique[~neg_mask] /= penalty
    logits[..., unique_tokens] = logits_unique


def _top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    # logits: [V]
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        thresh = torch.topk(logits, top_k).values[..., -1]
        logits = torch.where(logits < thresh, torch.full_like(logits, float('-inf')), logits)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        # keep minimal set with cumsum <= top_p
        mask = cumsum > top_p
        # always keep at least 1
        mask[..., 0] = False
        sorted_logits = torch.where(mask, torch.full_like(sorted_logits, float('-inf')), sorted_logits)
        # Scatter back to original order
        logits_new = torch.full_like(logits, float('-inf'))
        logits_new.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
        logits = logits_new
    return logits


@torch.no_grad()
def sample_candidates(model: TinyMoETransformer, tokenizer: AutoTokenizer, prompt: str,
                      k: int = 5, max_new_tokens: int = 256, temperature: float = 0.9,
                      top_k: int = 50, top_p: float = 0.9, repetition_penalty: float = 1.1,
                      device: str = 'cuda', seed: int | None = None,
                      add_bos: bool = False) -> List[str]:
    # Encode prompt
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if add_bos and tokenizer.bos_token_id is not None:
        ids = [int(tokenizer.bos_token_id)] + ids
    input_ids = torch.tensor([ids], device=device)
    # Trim to block size if needed
    if input_ids.size(1) > model.block_size:
        input_ids = input_ids[:, -model.block_size:]

    outs: List[str] = []
    eos_id = tokenizer.eos_token_id
    for i in range(k):
        if seed is not None:
            torch.manual_seed(seed + i)
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            idx_cond = generated if generated.size(1) <= model.block_size else generated[:, -model.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :].float().squeeze(0)  # [V]
            # repetition penalty on the full sequence so far
            _apply_repetition_penalty(logits, generated.view(-1), repetition_penalty)
            # temperature
            logits = logits / max(1e-5, float(temperature))
            # top-k / top-p filtering
            logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [1]
            generated = torch.cat([generated, next_id.view(1,1)], dim=1)
            if eos_id is not None and int(next_id) == int(eos_id):
                break
        new_tokens = generated[0, input_ids.size(1):].detach().tolist()
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        outs.append(text.strip())
    return outs


def main():
    ap = argparse.ArgumentParser(description='Interactive chat for TinyMoETransformer checkpoints')
    ap.add_argument('--ckpt', type=str, default='checkpoints_moe_bf16/best_moe_bf16.pt')
    ap.add_argument('--tokenizer', type=str, default='mixed_dataset_hf_tokenizer')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--k', type=int, default=5, help='number of candidate responses to sample')
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--top_k', type=int, default=50, help='top-k filtering for sampling')
    ap.add_argument('--top_p', type=float, default=0.9, help='nucleus sampling probability mass')
    ap.add_argument('--repetition_penalty', type=float, default=1.1)
    ap.add_argument('--max_new_tokens', type=int, default=256)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--system', type=str, default='You are a helpful assistant.')
    ap.add_argument('--add_bos', action='store_true', help='Prepend BOS token to the prompt before sampling')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, meta = load_model(args.ckpt, device=str(device))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, trust_remote_code=False)

    print('\nLoaded checkpoint:', args.ckpt)
    if meta.get('iter') is not None or meta.get('val_loss') is not None:
        print('Checkpoint info:', f"iter={meta.get('iter')}, val_loss={meta.get('val_loss')}")
    # Vocab sanity check
    try:
        tok_len = len(tokenizer)
        if hasattr(model, 'vocab_size') and tok_len != int(model.vocab_size):
            print(f"WARNING: tokenizer size ({tok_len}) != model.vocab_size ({model.vocab_size}). Decoding may be incorrect.")
    except Exception:
        pass
    print('Device:', device)
    print('Sampling:', f'k={args.k}, temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, rep_pen={args.repetition_penalty}')
    print('Max new tokens:', args.max_new_tokens)
    print("Type 'exit' to quit.\n")

    history: List[dict] = []
    while True:
        try:
            user = input('User> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nBye!')
            break
        if user.lower() in {'exit', 'quit'}:
            print('Bye!')
            break
        if not user:
            continue

        history.append({'role': 'user', 'content': user})
        prompt = build_prompt(history, system=args.system)
        candidates = sample_candidates(
            model, tokenizer, prompt,
            k=args.k, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=str(device), seed=args.seed, add_bos=args.add_bos,
        )

        # Show candidates
        print('\nAssistant (top-k candidates):')
        for i, txt in enumerate(candidates, 1):
            print(f'[{i}] {txt}\n')

        # Accept the first as the assistant reply to keep context moving
        history.append({'role': 'assistant', 'content': candidates[0]})


if __name__ == '__main__':
    main()
