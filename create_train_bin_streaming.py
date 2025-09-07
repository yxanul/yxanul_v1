#!/usr/bin/env python3
"""Convert parquet dataset to binary token file with true streaming."""

import os
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading

def tokenize_batch(texts, tokenizer, BOS_TOKEN_ID, MAX_LENGTH, vocab_size):
    """Tokenize a batch of texts."""
    if not texts:
        return []
    
    batch_encodings = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH - 1,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False
    )
    
    result = []
    for tokens in batch_encodings['input_ids']:
        if not tokens:
            continue
        tokens = [BOS_TOKEN_ID] + tokens
        tokens_array = np.array(tokens, dtype=np.uint16)
        if np.any(tokens_array >= vocab_size):
            tokens_array = np.clip(tokens_array, 0, vocab_size - 1)
        result.append(tokens_array.tobytes())
    return result

def writer_thread(write_queue, output_file, stats):
    """Dedicated thread for writing to disk."""
    with open(output_file, 'wb', buffering=8192) as f:
        while True:
            item = write_queue.get()
            if item is None:  # Sentinel for shutdown
                break
            for token_bytes in item:
                f.write(token_bytes)
                stats['tokens'] += len(token_bytes) // 2
            f.flush()
            write_queue.task_done()

def main():
    # Configuration
    PARQUET_FILE = "mixed_dataset_large.parquet"
    OUTPUT_FILE = "train.bin"
    TOKENIZER_DIR = "mixed_dataset_hf_tokenizer"
    BATCH_SIZE = 1000  # Smaller batches for smoother processing
    MAX_LENGTH = 16384
    BOS_TOKEN_ID = 2
    NUM_WORKERS = 12  # Match your CPU cores
    
    print(f"Loading tokenizer from {TOKENIZER_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR,
        use_fast=True,
        trust_remote_code=False
    )
    
    vocab_size = len(tokenizer)
    assert vocab_size == 32768, f"Expected vocab_size=32768, got {vocab_size}"
    print(f"Tokenizer loaded. Vocab size: {vocab_size}")
    
    print(f"\nOpening {PARQUET_FILE}...")
    parquet_file = pq.ParquetFile(PARQUET_FILE)
    total_rows = parquet_file.metadata.num_rows
    print(f"Total documents: {total_rows:,}")
    print(f"Using {NUM_WORKERS} worker threads")
    
    # Setup write queue and stats
    write_queue = queue.Queue(maxsize=NUM_WORKERS * 2)
    stats = {'tokens': 0}
    
    # Start writer thread
    writer = threading.Thread(target=writer_thread, args=(write_queue, OUTPUT_FILE, stats))
    writer.start()
    
    print("\nProcessing documents...")
    
    with tqdm(total=total_rows, unit="docs") as pbar:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit initial batches
            batch_iter = parquet_file.iter_batches(batch_size=BATCH_SIZE)
            active_futures = {}
            
            # Keep pipeline full
            for _ in range(min(NUM_WORKERS * 2, total_rows // BATCH_SIZE)):
                try:
                    batch = next(batch_iter)
                    df_batch = batch.to_pandas()
                    valid_texts = [text for text in df_batch['text'].tolist() 
                                  if text and text.strip()]
                    
                    if valid_texts:
                        future = executor.submit(
                            tokenize_batch,
                            valid_texts,
                            tokenizer,
                            BOS_TOKEN_ID,
                            MAX_LENGTH,
                            vocab_size
                        )
                        active_futures[future] = len(df_batch)
                    else:
                        pbar.update(len(df_batch))
                except StopIteration:
                    break
            
            # Process as futures complete
            while active_futures:
                # Wait for any future to complete
                done, pending = as_completed(active_futures, timeout=None).__next__(), None
                
                if done in active_futures:
                    batch_len = active_futures.pop(done)
                    try:
                        token_bytes_list = done.result()
                        if token_bytes_list:
                            write_queue.put(token_bytes_list)
                        pbar.update(batch_len)
                        pbar.set_postfix({"Tokens": f"{stats['tokens']:,}", "Queue": write_queue.qsize()})
                    except Exception as e:
                        print(f"\nError: {e}")
                        pbar.update(batch_len)
                    
                    # Submit next batch
                    try:
                        batch = next(batch_iter)
                        df_batch = batch.to_pandas()
                        valid_texts = [text for text in df_batch['text'].tolist() 
                                      if text and text.strip()]
                        
                        if valid_texts:
                            future = executor.submit(
                                tokenize_batch,
                                valid_texts,
                                tokenizer,
                                BOS_TOKEN_ID,
                                MAX_LENGTH,
                                vocab_size
                            )
                            active_futures[future] = len(df_batch)
                        else:
                            pbar.update(len(df_batch))
                    except StopIteration:
                        pass
    
    # Shutdown writer
    write_queue.join()
    write_queue.put(None)
    writer.join()
    
    # Print statistics
    print(f"\nConversion complete!")
    print(f"Total tokens written: {stats['tokens']:,}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"File size: {os.path.getsize(OUTPUT_FILE) / (1024**3):.2f} GB")
    
    # Verify
    print("\nVerifying output file...")
    with open(OUTPUT_FILE, 'rb') as f:
        first_tokens = np.frombuffer(f.read(200), dtype=np.uint16)
        print(f"First 20 tokens: {first_tokens[:20].tolist()}")
        print(f"BOS tokens in first 100: {np.sum(first_tokens == BOS_TOKEN_ID)}")

if __name__ == "__main__":
    main()