# PHASE 2: General Multi-Spike Training

## Goal
Make exact-gradient training work for general spiking neurons (multiple spikes per neuron, multiple resets), not just TTFS single-spike.

## Status: Steps 1, 2, 4 DONE. Step 3 deferred (grid engine already handles multi-spike).

## Completed

### Step 1: Saltation in core ✅
- `backward_layer_saltation()` in core.py
- `forward_multispike_layer_torch()` and `backward_multispike_layer_torch()` in core.py
- `loss_and_grads_saltation()` on `TTFSNetTorch`

### Step 2: Multi-spike parameter ✅
- `max_spikes` parameter in `TTFSNetTorch.__init__()` (default=1)
- `loss_and_grads()` auto-dispatches to saltation path when max_spikes > 1

### Step 4: Gradient verification ✅
- Single-layer dt_fire/dW: cosine = **1.000000** for both layers (machine-precision)
- Multi-spike forward matches single-spike for first spike
- Multi-spike training loop: loss 14.07 → 1.10 in 30 steps
- Backward compatibility: default=max_spikes=1 identical
- Explicit-adjoint gradient check: cosine = **1.000000**

### Step 3: Event-driven multi-spike — DEFERRED
- EventTTFSNet overrides `_forward_layer` for single-spike (no grid scan)
- For multi-spike (max_spikes>1), `loss_and_grads` dispatches to `forward_multispike()` which uses `forward_multispike_layer_torch` (grid engine)
- This is correct and fast enough; event-driven multi-spike is an optimization

## Remaining
- [ ] Step 5: Update README with multi-spike examples
- [ ] Train on CIFAR-10 (Phase 3)
- [ ] Compare multi-spike vs single-spike accuracy
