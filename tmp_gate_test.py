import torch
from model_experimental import TinyMoETransformer

# test plain
m = TinyMoETransformer(vocab_size=1024, n_layer=1, n_head=4, d_model=128, n_experts=1, block_size=32, attn_gate='none')
idx = torch.randint(0, 1024, (2, 16))
logits, loss = m(idx, idx)
print('plain ok', logits.shape, float(loss))

# test gated
m2 = TinyMoETransformer(vocab_size=1024, n_layer=1, n_head=4, d_model=128, n_experts=1, block_size=32, attn_gate='sigmoid_head')
logits2, loss2 = m2(idx, idx)
print('gated ok', logits2.shape, float(loss2))
