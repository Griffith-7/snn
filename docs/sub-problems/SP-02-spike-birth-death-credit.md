# SP-02 — Spike Birth/Death: Silent Neuron Credit

**Status:** ✅ Complete · **Phase:** 2 · **Gate:** B (PASS, 2026-08-13)

## Problem statement

A neuron that *should* fire but doesn't has no spike time to differentiate — its gradient is exactly 0, so it can never be updated to fire (the **death attractor**). A neuron that fires but *shouldn't* cannot be silenced by a timing gradient. Classical calculus has no answer for this discrete existence decision.

This sub-problem is: give **dead** neurons a learning signal that is (a) *informative* (does not vanish when far from threshold), (b) *unbiased* or provably correct w.r.t. a well-defined objective, and (c) *principled* — derived, not hand-tuned.

This is THE core research risk of the project and the reason exact spike-time methods historically fail on real tasks.

## Why the current "revival" draft fails (do not repeat)

From the skeleton review (MEMORY.md):

1. **Signal vanishes far from threshold.** `ds = s(1-s)/T` where `s = sigmoid((u_peak - theta)/T)`. A neuron 10 units below threshold has `ds ~ 4.5e-5`. Far-dead neurons get nothing.
2. **Annealing shrinks it further.** T decreases over training → the rescue window narrows. The "death attractor" is not avoided, just nudged near threshold.
3. **No output-layer handling.** Output neurons that are silent for the correct class have no `dL/dt_out` at all — the loss itself is undefined for them.
4. **It is a heuristic**, with free knobs (`alpha`, `T`) and no derivation.

## The candidate solution (to be derived, then chosen)

### Option A — Escape-noise / stochastic neuron equivalence (leading candidate)

Replace the deterministic threshold with a stochastic neuron: the neuron fires with probability `p(v) = sigma((v - theta)/T)` (escape rate). The *expected* spike count / spike rate is a **smooth** function of weights, so it has an exact gradient:

```
dE[spikes]/dW  is well-defined and nonzero for ALL neurons, including far-dead ones.
```

Key result (Gygax & Zenke 2025): the surrogate gradient used in practice is provably equal to this escape-noise gradient in a specific noise limit. This turns the "quarantined surrogate" into an exact gradient of a well-defined stochastic model.

Far-dead reach: a logistic escape rate never vanishes — `sigma` has nonzero derivative everywhere. Whether the *expected* gradient has enough magnitude at extreme distance is exactly what SP-02 must test (it depends on variance/baseline design, not just the sigmoid).

### Option B — REINFORCE-style estimator with spike-timing baseline

`dL/dW = E[ L * d log p(spikes)/dW ]` via the likelihood-ratio trick. Unbiased for the stochastic objective, works at any distance. Known issues: high variance; needs a baseline (e.g., eligibility-trace / value baseline) to be usable — the baseline design IS the research.

### Option C — E-prop style eligibility traces as a complementary channel

Three-factor rule with a per-neuron eligibility trace. Good hardware properties; used as the *local* mechanism that SP-04 can adopt.

**Decision D2 (resolved 2026-08-13):** Option A — escape-noise peak-margin channel, per
`docs/research/SP-02-research.md` §3. Derivation: Bernoulli existence `p_j = sigmoid((u_peak - theta)/T)`
at the potential peak; MLE objective `L_exist = -(lam/B) Σ target log p`; the `-log p` form
gives far-dead margin-gradient bounded below by ~`1/T` (the naive `sigma'` draft vanishes);
exact envelope `d(u_peak)/dW_ji = K(t_peak_j - t_i)`. B (REINFORCE), C (e-prop → SP-04),
D (pseudospikes) rejected with reasons recorded.

**Peak-extremum fix (empirically forced, 2026-08-13):** the naive argmax over `[0, t_max]`
collapses onto the pre-input plateau for any subthreshold neuron → zero gradient (E6 failed).
The peak is now searched over the **response window** (t ≥ earliest contributing event), and
for all-negative responses the interior **minimum** is used (u' = 0 → envelope still exact).
Without this the channel cannot revive a neuron whose potential never exceeds 0.

## Requirements / acceptance (GATE B)

- [x] The silent-neuron signal is **derived** (research doc §2, written derivation + assumptions), not tuned.
- [x] **Far-dead test:** E6 — single neuron initialized `m0 ∈ {2,3,5,8}` below threshold revives; margin-gradient bounded below (~1/T); control (`lam=0`) stays dead. (`m0=1`/`w=0` excluded: no signal by construction.)
- [x] **Output-layer silence handled:** correct-class silent outputs targeted (E7: acc 0.47 → 0.97).
- [x] Verify the signal does NOT regress SP-01's exact gradients for fired neurons (E8a gradchecks PASS; E8b zero contribution at 100% firing; structural isolation §2.7).
- [x] Ablation: training WITHOUT the mechanism fails (E9: 0.58 chance, dead stay dead); WITH it succeeds (0.99). Scientific control.
- [x] Record annealing design rationale: **no annealing** — `T` is a fixed model noise level (§2.8).

## Open questions

- Q2.1: Revival step-count grows ~linearly with `m0` (measured in E6) — consistent with the bounded-gradient theory. Fine-grained `steps vs m0/T` scaling curve is future work.
- Q2.2: Revived neurons produce *correct* first-crossing spike times (E6: err 3.6e-15), and the SP-01 timing gradient takes over smoothly (E8b zero contribution once firing).
- Q2.3: Separate additive channel (current, chosen for Gate-B isolation) vs a unified stochastic forward (Gygax-Zenke) — unified model is future work.
- Q2.4: Full escape-rate survival-integral gradient (`P_fire = 1 - exp(-∫rho)`) vs the peak saddle-point form — the integral is the rigorous limit; future refinement.
- Q2.5: Uniform existence target (all silent hidden fire) vs per-neuron targets from downstream error — default prior now; future work.
- Q2.6 (new): `lam` sensitivity — at `lam=1` the channel competes with the timing loss's wrong-class push-DOWN and revival is slow (E7: 0.63 plateau); `lam=5` revives and learns (0.99). Reported, not hidden.

## Change log

- 2026-08-13: **Complete.** Implemented `peak_margin_torch` + `TTFSNetTorch.existence_grads` in `engine/snn_torch.py`; experiment suite E6–E9 in `engine/experiments/exp_sp02.py`; all pass. Results in `docs/results/SP-02-experiments.md`; derivation in `docs/research/SP-02-research.md`.
- 2026-08-13: Fixed the peak-extremum definition (response window + all-negative → interior minimum) after E6 exposed the zero-gradient plateau deadlock.
- 2026-08-13: D2 decided (Option A); research doc written; E6–E9 written (first run revealed the plateau bug).
