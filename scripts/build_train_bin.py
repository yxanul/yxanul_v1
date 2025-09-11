#!/usr/bin/env python3
"""
Build train.bin from a Parquet file using a Hugging Face tokenizer (fast).

Example:
  python scripts/build_train_bin.py \
    --parquet mixed_dataset_large.parquet \
    --text_column text \
    --tokenizer_dir mixed_dataset_hf_tokenizer_new \
    --out_dir data_new \
    --max_length 16384 --num_workers 12
"""

from __future__ import annotations

import argparse
import os
import threading
import queue
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from tqdm import tqdm


def tokenize_batch(texts, tokenizer, bos_id, max_length, vocab_size):
    enc = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length - 1,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    out = []
    for ids in enc['input_ids']:
        if not ids:
            continue
        ids = [bos_id] + ids
        arr = np.array(ids, dtype=np.uint16)
        if np.any(arr >= vocab_size):
            arr = np.clip(arr, 0, vocab_size - 1)
        out.append(arr.tobytes())
    return out


def writer_thread(write_q: queue.Queue, out_path: Path, stats: dict, flush_every: int = 0):
    """Write token bytes to disk, with optional periodic flush+fsync.

    - If flush_every > 0, flush+fsync every N documents written.
    - If flush_every == 0 (default), flush once per queue item (batch), like before.
    """
    with open(out_path, 'wb', buffering=8192) as f:
        docs_since_flush = 0
        while True:
            item = write_q.get()
            if item is None:
                break

            # Support both (bytes_list, docs_in_batch) and just bytes_list
            if isinstance(item, tuple) and len(item) == 2:
                token_bytes_list, docs_in_batch = item
            else:
                token_bytes_list = item
                try:
                    docs_in_batch = len(token_bytes_list)
                except Exception:
                    docs_in_batch = 1

            for token_bytes in token_bytes_list:
                f.write(token_bytes)
                stats['tokens'] += len(token_bytes) // 2

            if flush_every > 0:
                docs_since_flush += int(docs_in_batch)
                if docs_since_flush >= flush_every:
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass
                    docs_since_flush = 0
            else:
                # original behavior: flush at each batch
                f.flush()

            write_q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', type=str, required=True)
    ap.add_argument('--text_column', type=str, default='text')
    ap.add_argument('--tokenizer_dir', type=str, required=True)
    ap.add_argument('--out_dir', type=str, default='data_new')
    ap.add_argument('--max_length', type=int, default=16384)
    ap.add_argument('--batch_size', type=int, default=1000)
    ap.add_argument('--num_workers', type=int, default=12)
    ap.add_argument('--flush_every', type=int, default=0, help='Flush+fsync every N documents (0=flush each batch)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'train.bin'

    print(f"Loading tokenizer from {args.tokenizer_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True, trust_remote_code=False)
    vocab_size = len(tokenizer)
    bos_id = tokenizer.bos_token_id
    assert bos_id is not None, 'Tokenizer must define a BOS token.'
    assert vocab_size <= 65535, f'Vocab too large for uint16: {vocab_size}'
    print(f"Tokenizer loaded. Vocab size: {vocab_size}, BOS id: {bos_id}")

    pf = pq.ParquetFile(args.parquet)
    total_rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
    print(f"Opening {args.parquet} | total rows: {total_rows:,}")

    write_q: queue.Queue = queue.Queue(maxsize=args.num_workers * 2)
    stats = {'tokens': 0}

    wt = threading.Thread(target=writer_thread, args=(write_q, out_path, stats, args.flush_every))
    wt.start()

    with tqdm(total=total_rows, unit='docs') as pbar:
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=[args.text_column])
            texts = tbl.column(args.text_column).to_pylist()
            # chunk into batches
            for i in range(0, len(texts), args.batch_size):
                batch_texts = [t for t in texts[i:i+args.batch_size] if t and str(t).strip()]
                if not batch_texts:
                    pbar.update(min(args.batch_size, len(texts) - i))
                    continue
                token_bytes = tokenize_batch([str(t) for t in batch_texts], tokenizer, bos_id, args.max_length, vocab_size)
                if token_bytes:
                    # include docs count to support periodic flushing
                    write_q.put((token_bytes, len(token_bytes)))
                pbar.update(min(args.batch_size, len(texts) - i))
                pbar.set_postfix({'Tokens': f"{stats['tokens']:,}"})

    write_q.join()
    write_q.put(None)
    wt.join()

    print(f"\nDone. Wrote train.bin to {out_path}")
    print(f"Total tokens: {stats['tokens']:,}")
    print(f"File size: {os.path.getsize(out_path)/(1024**3):.2f} GB")


if __name__ == '__main__':
    main()
