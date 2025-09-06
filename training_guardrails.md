# Training Guardrails & Quick Checks

## Early Training Issues (Steps 0-400)

### Loss Spikes
**Symptom:** Loss oscillates ≥0.3 in first 200-400 steps  
**Fix:** 
- Drop Muon LR: `0.028 → 0.027`
- OR increase momentum warmup: `300 → 400` steps

### Logit Saturation
**Symptom:** Head logits hit ~15 cap early (saturated)  
**Fix:**
- Keep the cap at 30.0
- Reduce head LR: `1/384 → 1/416` (≈0.00240)

## Mid-Training Issues

### Embedding Lag
**Symptom:** Training loss flattens early while validation loss remains high  
**Fix:**
- Increase embed/value_embed LR: `0.22 → 0.24`
- **Warning:** Do NOT exceed 0.25 at this model size

### Dead Top Layers
**Symptom:** Top layer activations stay near zero late in training  
**Fixes (try in order):**
1. Move long-window assignment closer to top layers
   - Current: Every 4 layers
   - New: Add more long layers near top
2. Increase window cap: `1792 → 2048` tokens

## Resource Management

### GPU Memory
**Constraint:** Single GPU training  
**Guidelines:**
- Keep batch size `B=1`
- Start with `T=16k` sequence length
- Let window grow progressively (128 → 1792)
- Do NOT increase batch size

## Quick Debug Commands

```python
# Check loss variance
if step < 400 and (max_loss - min_loss) > 0.3:
    print("WARNING: Loss spikes detected, consider Muon LR adjustment")

# Monitor top layer gradients
layer_grads = per_layer_grad_norms(model, monitor_layers)
if any(g < 1e-6 for g in layer_grads.values()):
    print("WARNING: Dead layers detected")

# Check logit saturation
if (logits.abs() > 28).any():
    print("WARNING: Logits approaching cap, reduce head LR")
```

## Key Hyperparameter Ranges

| Parameter | Current | Safe Range | Critical Limit |
|-----------|---------|------------|----------------|
| Muon LR | 0.028 | 0.025-0.030 | >0.035 unstable |
| Head LR | 1/384 | 1/416-1/320 | >1/256 unstable |
| Embed LR | 0.22 | 0.20-0.24 | >0.25 risky |
| Value Embed LR | 0.22 | 0.20-0.24 | >0.25 risky |
| Scalar LR | 0.012 | 0.010-0.015 | >0.020 unstable |
| Window Max | 1792 | 1536-2048 | >2560 OOM risk |
| Momentum Warmup | 300 | 200-500 | <150 unstable |