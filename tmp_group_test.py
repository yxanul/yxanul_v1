import torch
from model_experimental import TinyMoETransformer

cfg = dict(vocab_size=1024, n_layer=2, n_head=4, d_model=128, n_experts=3, block_size=32)

# capacity mode, grouped off
m1 = TinyMoETransformer(**cfg, moe_grouped=False)
idx = torch.randint(0, 1024, (2, 16))
logits, loss = m1(idx, idx)
print('nogroup ok', logits.shape, float(loss))

# capacity mode, grouped on (force no-dropless for code path)
m2 = TinyMoETransformer(**cfg, moe_grouped=True)
# Simulate capacity path by setting router capacity < avg tokens
# (we will run forward; the grouped path is used only when dropless=False; here default dropless=True, so set to False)
for blk in m2.h:
    blk.moe.router.dropless = False
logits2, loss2 = m2(idx, idx)
print('grouped ok', logits2.shape, float(loss2))
