#!/usr/bin/env python3
"""
Interactive chat for TinyMoETransformer checkpoints.

Loads checkpoints_moe_bf16/best_moe_bf16.pt, restores the model, and
provides a simple CLI chat. For each user turn it will generate K
candidate responses (default K=5) with temperature 0.7 and top-k sampling.

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


def load_model(ckpt_path: str, device: str = 'cuda') -> TinyMoETransformer:
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
        capacity_factor=cfg.get('capacity_factor', 1.25),
        dropless=cfg.get('dropless', True),
        load_balance_alpha=cfg.get('load_balance_alpha', 0.05),
        router_z_loss_coef=cfg.get('router_z_loss_coef', 0.0),
        attn_gate=cfg.get('attn_gate', 'none'),
        use_rope=cfg.get('use_rope', True),
        rope_theta=cfg.get('rope_theta', 10000.0),
        moe_grouped=cfg.get('moe_grouped', False),
    ).to(device=device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    return model


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


@torch.no_grad()
def sample_candidates(model: TinyMoETransformer, tokenizer: AutoTokenizer, prompt: str,
                      k: int = 5, max_new_tokens: int = 256, temperature: float = 0.7,
                      top_k_sample: int = 5, device: str = 'cuda', seed: int | None = None) -> List[str]:
    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([input_ids], device=device)
    # Trim to block size if needed
    if input_ids.size(1) > model.block_size:
        input_ids = input_ids[:, -model.block_size:]

    outs: List[str] = []
    for i in range(k):
        if seed is not None:
            torch.manual_seed(seed + i)
        out_ids = model.generate(input_ids.clone(), max_new_tokens=max_new_tokens,
                                 temperature=temperature, top_k=top_k_sample)
        new_tokens = out_ids[0, input_ids.size(1):].detach().tolist()
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        outs.append(text.strip())
    return outs


def main():
    ap = argparse.ArgumentParser(description='Interactive chat for TinyMoETransformer checkpoints')
    ap.add_argument('--ckpt', type=str, default='checkpoints_moe_bf16/best_moe_bf16.pt')
    ap.add_argument('--tokenizer', type=str, default='mixed_dataset_hf_tokenizer')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--k', type=int, default=5, help='number of candidate responses to sample')
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top_k_sample', type=int, default=5, help='top-k filtering for sampling')
    ap.add_argument('--max_new_tokens', type=int, default=256)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--system', type=str, default='You are a helpful assistant.')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.ckpt, device=str(device))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, trust_remote_code=False)

    print('\nLoaded checkpoint:', args.ckpt)
    print('Device:', device)
    print('Sampling:', f'k={args.k}, temperature={args.temperature}, top_k={args.top_k_sample}')
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
            temperature=args.temperature, top_k_sample=args.top_k_sample,
            device=str(device), seed=args.seed,
        )

        # Show candidates
        print('\nAssistant (top-k candidates):')
        for i, txt in enumerate(candidates, 1):
            print(f'[{i}] {txt}\n')

        # Accept the first as the assistant reply to keep context moving
        history.append({'role': 'assistant', 'content': candidates[0]})


if __name__ == '__main__':
    main()

