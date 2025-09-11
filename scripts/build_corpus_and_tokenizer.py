#!/usr/bin/env python3
"""
Extract a text corpus from a Parquet file into shard .txt files and train a ByteLevel-BPE tokenizer.

Usage example:
  python scripts/build_corpus_and_tokenizer.py \
    --parquet mixed_dataset_large.parquet \
    --text_column text \
    --corpus_dir corpus_shards \
    --shard_size 200000 \
    --train_tokenizer \
    --tokenizer_out mixed_dataset_hf_tokenizer_new \
    --vocab_size 32768 --min_freq 2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pyarrow.parquet as pq


def write_corpus_shards(parquet_path: str, text_column: str, corpus_dir: str, shard_size: int = 200_000,
                        sample_ratio: float = 1.0) -> List[str]:
    pf = pq.ParquetFile(parquet_path)
    out_dir = Path(corpus_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    lines_in_shard = 0
    shard_paths: List[str] = []
    f = None
    total_docs = 0
    kept_docs = 0

    def _open_new_shard():
        nonlocal shard_idx, lines_in_shard, f
        shard_idx += 1
        lines_in_shard = 0
        shard_path = out_dir / f"shard_{shard_idx:05d}.txt"
        shard_paths.append(str(shard_path))
        f = open(shard_path, 'w', encoding='utf-8', buffering=1024*1024)

    _open_new_shard()
    rng = np.random.default_rng(1337)

    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=[text_column])
        col = tbl.column(text_column)
        # Convert to Python strings
        arr = col.to_pylist()
        for txt in arr:
            total_docs += 1
            if sample_ratio < 1.0 and rng.random() > sample_ratio:
                continue
            if not txt:
                continue
            s = str(txt).strip()
            if not s:
                continue
            kept_docs += 1
            f.write(s.replace('\r\n', '\n').replace('\r', '\n'))
            f.write('\n')
            lines_in_shard += 1
            if lines_in_shard >= shard_size:
                f.close()
                _open_new_shard()

    if f:
        f.close()
    print(f"Total docs: {total_docs:,} | Kept: {kept_docs:,} | Shards: {len(shard_paths)}")
    return shard_paths


def train_tokenizer_from_corpus(files: List[str], out_dir: str, vocab_size: int = 32768, min_freq: int = 2,
                                add_prefix_space: bool = False):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast

    special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=add_prefix_space)

    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_freq, special_tokens=special_tokens)
    print(f"Training tokenizer on {len(files)} files ...")
    tokenizer.train(files=files, trainer=trainer)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Wrap as HF fast tokenizer and save
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>",
    )
    fast.save_pretrained(out)
    print(f"Saved tokenizer to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', type=str, required=True)
    ap.add_argument('--text_column', type=str, default='text')
    ap.add_argument('--corpus_dir', type=str, default='corpus_shards')
    ap.add_argument('--shard_size', type=int, default=200_000)
    ap.add_argument('--sample_ratio', type=float, default=1.0)

    ap.add_argument('--train_tokenizer', action='store_true')
    ap.add_argument('--tokenizer_out', type=str, default='mixed_dataset_hf_tokenizer_new')
    ap.add_argument('--vocab_size', type=int, default=32768)
    ap.add_argument('--min_freq', type=int, default=2)
    ap.add_argument('--add_prefix_space', action='store_true')
    args = ap.parse_args()

    shard_files = write_corpus_shards(
        args.parquet, args.text_column, args.corpus_dir,
        shard_size=args.shard_size, sample_ratio=args.sample_ratio,
    )

    if args.train_tokenizer:
        train_tokenizer_from_corpus(
            shard_files, args.tokenizer_out, vocab_size=args.vocab_size,
            min_freq=args.min_freq, add_prefix_space=args.add_prefix_space,
        )


if __name__ == '__main__':
    main()

