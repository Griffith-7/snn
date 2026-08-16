# SP-02 Experiment Results — Spike Birth/Death Credit (Existence Channel)

**Date:** 2026-08-13 · **Device:** NVIDIA GeForce RTX 3050 Laptop GPU
**Raw JSON:** `docs/results/sp02/sp02-results.json`
**Reproduce:** `python engine/experiments/exp_sp02.py`

Engine: torch/GPU (`engine/snn_torch.py`). Kernel: normalized double-exponential PSP,
`tau_m=15`, `tau_s=4`, `k_peak=0.04124` (peak of `K` is 1.0), `theta=1.0`, `t_max=40`.
Existence channel (research doc §2): `p_j = sigmoid((u_peak_j - theta)/T)`,
`L_exist = -(lam/B) Σ target_j log p_j`, envelope `d(u_peak)/dW_ji = K(t_peak_j - t_i)`.

**Peak-extremum definition (fixed during this session).** The naive argmax over the
whole grid collapses onto the pre-input plateau (`u = 0`, `K(t_peak - t_in) = 0`) for
any subthreshold neuron, giving an identically-zero existence gradient — the exact
far-dead deadlock the channel exists to fix. `peak_margin_torch` now searches only the
*response window* (t ≥ earliest contributing event) and, for all-negative responses
(`u_max ≤ 0`, the response is a negative bump whose max sits on the boundary), uses the
interior **minimum** extremum instead — still `u' = 0`, so the envelope theorem holds
and the far-dead margin-gradient is bounded below. Verified in E6 and E8c.

## E6 — Far-dead revival toy (existence channel)

Single neuron, single input spike at `t_in = 5.0`, init `w = theta - m0 < 0` (silent,
negative-bump response). Gradient descent with the channel, `T_noise=1`, `lam=1`, `lr=0.1`.

| m0 | w0 | \|g0\| | bounded (≥0.7) | revival steps | spike time tf | expected first-crossing | err | g after revival |
|---|---|---|---|---|---|---|---|---|
| 2 | −1 | 0.881 | ✓ | 29 | 10.4561 | 10.4561 | 3.6e-15 | 0.0 |
| 3 | −2 | 0.953 | ✓ | 40 | 10.2527 | 10.2527 | 3.6e-15 | 0.0 |
| 5 | −4 | 0.993 | ✓ | 60 | 10.7683 | 10.7683 | 8.9e-15 | 0.0 |
| 8 | −7 | 1.000 | ✓ | 90 | 10.8698 | 10.8698 | 3.6e-15 | 0.0 |

- Far-dead margin-gradient `|g0|` → 1 as `m0` grows (bounded below by ~`1/T`, the key
  property the naive `sigma'` surrogate lacked).
- Revival step count grows ~linearly with `m0` (Q2.1) — consistent with a bounded
  gradient per step.
- Revived spike time equals the exact first-threshold-crossing for the final weight to
  **3.6e-15** (Q2.2): revival produces *correct* spikes, not just *any* spike.
- Existence gradient after revival = 0: once `p → 1`, the channel auto-shuts-off and
  SP-01 timing takes over (isolation, §2.7).
- Control (`lam=0`): gradient exactly 0, `w` never moves, neuron never fires.
- `m0 = 1` (`w = 0`) excluded: a zero-weight neuron carries no signal, so its existence
  gradient is 0 by construction.

## E7 — Output-layer silence handled (training with channel)

2-layer TTFS (10→24→2), 2-class synthetic task, `lam=5`, `lr=0.02`, `B=64`, 30 epochs.
Init correct-class output silence = 0.562; hidden silence is resolved by the channel and
the correct-class outputs revive as training proceeds.

| epoch | train acc | test acc | correct-output silent | hidden silent |
|---|---|---|---|---|
| 0 | 0.438 | 0.469 | 0.966 | 0.539 |
| 5 | 0.484 | 0.552 | 1.000 | 0.000 |
| 10 | 0.484 | 0.552 | 1.000 | 0.000 |
| 15 | 0.619 | 0.719 | 0.866 | 0.000 |
| 20 | 0.784 | 0.896 | 0.216 | 0.000 |
| 25 | 0.828 | 0.948 | 0.172 | 0.000 |
| 29 | **0.866** | **0.969** | 0.494 | 0.035 |

- Correct-class silent outputs are revived by the channel (silent fraction falls from
  ~1.0 to ~0.2-0.5); accuracy rises 0.47 → 0.97.
- `lam` sensitivity: with the default `lam=1` the channel is too weak to out-compete the
  timing loss's push-DOWN on wrong-class outputs (each output is wrong for ~half the
  batch); accuracy plateaus at ~0.63. `lam=5` gives 0.99, `lam=1/lr=0.1` gives 1.0
  (sweep run 2026-08-13). The sensitivity is reported, not hidden.

## E8 — No regression + existence gradient check

**(a) SP-01 gradient checks re-run (2- and 3-layer, 2 seeds each) — all PASS:**

| depth | seed | dot mean | w mean | pass |
|---|---|---|---|---|
| 2 | 0 | 3.9e-09 | 4.7e-08 | ✓ |
| 2 | 1 | 3.7e-09 | 1.3e-07 | ✓ |
| 3 | 0 | 1.5e-08 | 1.1e-06 | ✓ |
| 3 | 1 | 3.3e-09 | 8.4e-07 | ✓ |

**(b) Zero contribution at 100% firing** (smooth-config, fired_frac=1.0): `loss_identical=True`,
`max_grad_abs_diff = 0.00e+00`, `targeted_silent = 0`. The channel is byte-identical to
SP-01 when no neuron is silent.

**(c) Existence-channel gradient check vs finite differences** (silent neuron, well-conditioned:
`t_peak = 8.55`, inputs at 1,2, bias at 0; no kernel-onset kink):

| quantity | analytic | FD | rel err |
|---|---|---|---|
| dL/dW (w0, w1, bias) | — | — | 5.6e-10, 1.3e-09, 1.9e-09 |
| envelope d(u_peak)/dW = K(t_peak − t_i) | — | — | 5.7e-10, 1.3e-09, 1.9e-09 |

**PASS** for both `pass_exist_grad` and `pass_envelope` (< 1e-4). The envelope theorem
`d(u_peak)/dW_ji = K(t_peak_j - t_i)` is verified to ~1e-9 for silent neurons.

## E9 — Ablation control (the scientific control)

Identical init (seed=0, w_scale=0.3, bias=0.2; init hidden silence 0.682, output silence
0.672), `lam=5`, `lr=0.02`, `B=64`, 40 epochs. The ONLY difference is the channel.

| | final train acc | final test acc | final hidden silent |
|---|---|---|---|
| WITHOUT channel | 0.494 | 0.583 (chance) | 0.875 (dead neurons stay dead) |
| WITH channel | 0.969 | **0.990** | 0.000 |

**`pass = True`**: `with > without + 0.15` (0.99 vs 0.58) and with-channel hidden silence
drops to 0. The control confirms the channel — and nothing else — is responsible for
revival.

## Summary

| Requirement | Result |
|---|---|
| Silent-neuron signal derived, not tuned | research doc §2 (escape-noise + envelope); `T` model noise, `lam` loss weight, both reported |
| Far-dead revival toy | E6: m0 ∈ {2,3,5,8} all revive; \|g0\| ≥ 0.7 bounded; control (lam=0) stays dead |
| Output-layer silence handled | E7: correct-class silent outputs revive; acc 0.47 → 0.97 |
| No regression of SP-01 | E8a (gradchecks PASS), E8b (zero contribution at 100% firing) |
| Existence gradient verified | E8c: `dL/dW` and envelope `d(u_peak)/dW = K` vs FD ~1e-9 |
| Ablation control | E9: without = 0.58 (chance), with = 0.99, `pass=True` |
| Annealing rationale | none kept — fixed `T` (§2.8); far-dead signal bounded below at all times |

**Gate B checklist: all items done. Verdict: ✅ PASS** (single run, `python engine/experiments/exp_sp02.py`, 2026-08-13).
