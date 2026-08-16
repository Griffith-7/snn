# Phase 5 — Benchmark Plan (Gate E: full engine vs surrogate baseline)

Status: 🟢 PASSED · Last updated: 2026-08-15

## Goal (PRD G0)

> A trainable SNN engine whose accuracy on a real benchmark (CIFAR-10 or CIFAR-10-DVS)
> is at least as good as a surrogate-gradient baseline (e.g., STBP/SLAYER), at equal or
> better energy/latency.

Gate E = benchmark passes honestly, then the main problem is declared solved (per `docs/01-main-problem.md`).

## Honest framing (what we can and cannot claim)

- The engine is currently a **feedforward TTFS MLP** (double-exponential PSP, exact
  spike-time gradients, existence channel, per-layer local loss). Verified on toy tasks
  only (2-class synthetic, 2–4 layers).
- CIFAR-10 is 32×32×3. A 3072-input TTFS MLP first layer is heavy on an RTX 3050 and is
  not where the science is. PRD §9 puts conv kernels beyond a simple front-end out of v1 scope.
- Therefore the first Gate-E attempt uses a **downsampled grayscale front-end** (e.g.,
  12×12 = 144 inputs) and an **apples-to-apples comparison**: the *same architecture, data
  pipeline, loss, and optimizer* for both the exact engine and a from-scratch
  STBP/SLAYER-style surrogate baseline. Only the learning rule differs.
- No claim to beat SOTA CIFAR-10 numbers; the claim is engine ≥ surrogate at equal/better
  energy on the same configuration, honestly measured.

## Components to build

### 1. Data pipeline (`engine/experiments/exp_sp05.py` + a small `engine/cifar_io.py`)
- Load local `cifar-10-python/cifar-10-batches-py` (present). Train 50k / test 10k.
- Normalize → grayscale → downsample (nearest or mean-pool) to chosen resolution.
- TTFS encode: pixel intensity `x ∈ [0,1]` → input spike time `t = 0.5 + (t_max_in − 0.5)*(1 − x)`
  (bright = early), consistent with the existing toy encoder.
- Same images served to both engine and baseline (same tensor, same split/seeds).

### 2. Surrogate baseline (STBP/SLAYER-style, from scratch, in torch)
- Feedforward MLP over `T` discrete timesteps, membrane `v = (1−λ)v + Σ W s_in`, spike
  `s = H(v−θ)` with sigmoid surrogate `σ'(v−θ)` in the backward pass.
- Same size list, same weight init scale, same loss family (latency CE on first-spike
  times / rate CE), same optimizer + seed.
- Implemented in `engine/baseline_stbp.py` (no new deps; snntorch is not installed and a
  from-scratch baseline keeps the comparison controllable).

### 3. Metrics
- **Accuracy:** test top-1, engine vs baseline (same data).
- **Energy (SynOps):** Σ over layers of (spikes fired in layer l) × (fan-in of layer l+1),
  counted on the same test set for both models.
- **Latency (timesteps):** TTFS = 1 event per neuron (sparse); surrogate = T timesteps.
  Also report wall-clock train time for transparency.
- **Configs + seeds published** in the results doc; single-file reproduction.

### 4. Benchmark procedure (Gate E acceptance)
1. Train exact engine (modes: `ref` = SP-01+SP-02 exact, and `deep` = SP-04 local) on the
   chosen resolution/subset.
2. Train surrogate baseline on the identical data/arch/loss.
3. Record accuracy, SynOps, latency for both; verdict = engine ≥ baseline at ≤ baseline energy.
4. Full 50k / full-resolution is the stretch goal if the first pass is feasible on the GPU.

## Open scope decisions (confirm before building)

1. **Resolution/front-end:** 8×8 (64), 12×12 (144), or 16×16 (256) grayscale inputs.
2. **Training budget:** subset (e.g., 10k–20k train samples, ~30–60 epochs) vs full 50k.
3. **CIFAR-10-DVS:** defer to a second Gate-E attempt (needs download + event→TTFS encoding) —
   recommended, since CIFAR-10 suffices for G0 ("CIFAR-10 OR CIFAR-10-DVS").
4. **Baseline:** from-scratch STBP in `engine/baseline_stbp.py` (recommended) vs installing snntorch.

## Gate E checklist (mirrors GATES.md)

- [ ] Benchmark vs STBP/SLAYER on CIFAR-10 (apples-to-apples config)
- [ ] Energy (SynOps) + latency (timesteps) measured for both
- [ ] Configs + seeds published; single-file rerun
- [ ] Verdict: engine ≥ surrogate at equal/better energy

## Change log

- 2026-08-15: **Gate E PASSED** — full 15k/40 run: ref 0.273 / deep 0.250 / stbp 0.238; engine ≥
  baseline at ~equal SynOps and 1-event/neuron vs T=160 latency. Debugging fixed two root causes
  (dead-output init w_scale=0.1→0.4; weak latency-CE signal beta=1→3). Results in
  `docs/results/SP-05-experiments.md`.
- 2026-08-14: Plan written after the full audit; scope decisions pending user confirmation.
