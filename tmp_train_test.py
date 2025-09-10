import torch
from model_experimental import TinyMoETransformer, TrainConfig, train

# Quick dry-run one iteration CPU
cfg = TrainConfig(device='cpu', max_iters=2, eval_interval=2, eval_iters=1, batch_size=2, gradient_accumulation_steps=1, block_size=32, d_model=128, n_head=4, n_experts=2, n_layer=2, data_path='train.bin', compile=False)

# Create a tiny fake train.bin/val.bin next to CWD
import numpy as np, os
n=5000
arr = (np.random.randint(0, cfg.vocab_size, size=(n,), dtype=np.uint16))
open('train.bin','wb').write(arr.tobytes())
open('val.bin','wb').write(arr[-1000:].tobytes())

train(cfg)
