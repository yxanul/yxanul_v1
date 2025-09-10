import torch
from model_experimental import TinyMoETransformer
m = TinyMoETransformer(vocab_size=32768, n_layer=2, n_head=4, d_model=256, n_experts=2, block_size=64)
print('params', m.num_parameters())
idx = torch.randint(0, 32768, (2, 32))
logits, loss = m(idx, idx)
print('ok', logits.shape, float(loss.item()))
