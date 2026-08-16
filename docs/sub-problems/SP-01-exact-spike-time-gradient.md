# SP-01 — Exact Gradient w.r.t. Spike Time (Fired Neurons)

**Status:** ✅ Complete · **Phase:** 1 · **Gate:** A (PASS, see `docs/tracking/GATES.md`)

## Problem statement

For a neuron that **fires** at time `t_f`, the loss is a function of that spike time: `L = L(t_f)`. We need `dL/dW`. The membrane potential at firing satisfies the threshold condition:

```
v(t_f; W) = theta
```

The derivative of the spike output `H(v - theta)` is useless (0 or Dirac). But the *spike time* `t_f` is a smooth function of `W` when the crossing is transversal (`dv/dt != 0`). This sub-problem is: compute `dL/dW` exactly for all fired neurons, in a deep network, and verify it numerically.

## The math (derivation)

### Spike time as an implicit function

For a post-synaptic neuron `n` receiving pre-synaptic spikes at times `{t_e}`, the membrane potential (double-exponential PSP, kernel `K`) is:

```
v_n(t) = bias term + sum_e w_ne * K(t - t_e),   K(d) = (e^(-d/tau_m) - e^(-d/tau_s)) / (tau_m - tau_s)
```

Firing time `t_f` solves `v_n(t_f) = theta`. Implicit differentiation:

```
0 = dv_n/dt (t_f) * dt_f/dW + dv_n/dW (t_f)
=> dt_f/dW = - (dv_n/dW) / (dv_n/dt)(t_f)
```

`dv_n/dW_ne = K(t_f - t_e)` (the kernel evaluated at the spike time). `dv_n/dt = du_n/dt` at `t_f` — computable analytically.

### Chain rule into the loss

If the output-layer loss depends on spike times as `L(t_out)`, then:

```
dL/dW = sum over fired post neurons of  (dL/dt_f) * dt_f/dW
```

The factor `dL/dt_f` is the **adjoint** `lam`, propagated from the loss.

### Adjoint propagation (layer to layer)

A post-synaptic spike at `t_post` contributes to a pre-synaptic neuron's membrane via `w * K(t_post - t_pre)`. The adjoint `lam` at a pre-synaptic spike `t_pre` accumulates:

```
lam_pre += sum over postsynaptic fired neurons f of
    lam_f * w_f * dK/dd (t_post - t_pre) * dt_post/d...  (careful: includes dt_post/dW_pre chain)
```

Concretely, the engine's `backward_layer` computes:

```
alpha = lam_f / (du/dt)(t_post)        # = lam_f * dt_post/du
lam_prev[e] += alpha * W[nf, c] * K'(t_post - t_ev[e])
```

where `K'` is the derivative of the kernel w.r.t. delay. This is the standard TTFS/SpikeProp adjoint scheme.

### Weight gradient

```
dL/dW[nf, e] = -alpha * K(t_post - t_ev[e])   for each fired post neuron nf and pre-spike e
```

## Requirements / acceptance (GATE A)

- [x] Analytic `dt/dW` written down and implemented for a 2-layer TTFS network.
- [x] **Bias:** modeled as a fixed input spike at `t_b=0` with a trainable weight column (documented choice). With a bias *impulse* the bias gradient is the impulse kernel `K(t_f - t_b)`, NOT the step response — the skeleton-review suspicion applied to a constant-current bias and is moot for this model. Verified by E2/E3.
- [x] Adjoint accumulation `lam_prev` correct across layers.
- [x] Output loss head produces `dL/dt_out` for fired output neurons (latency cross-entropy, `dL/dt = beta*(onehot - p)`).
- [x] **Gradient check passes:** analytic `dL/dW` vs central finite differences, relative error < 1e-4. 2-layer: dot ~5e-9 / w ~1e-7. 3-layer: dot ~4e-8 / w ~2e-6. Near-zero gradients tracked by absolute error (float64 floor); status-flip and grazing perturbations skipped by design (see E5b/E5d).
- [x] Silent neurons in forward (infinite `t_f`) are routed to SP-02 logic; no NaN. Verified: silent gradient rows exactly zero (E5c).
- [x] Regression harness: gradient-check suite saved as permanent tests in `engine/experiments/exp_sp01.py`.

## Open questions

- Q1.1: What loss function is best for TTFS classification, and how does it behave when output neurons are silent? (links to SP-02)
- Q1.2: How to handle near-grazing crossings (`dv/dt ~ 0`), where `dt/dW` blows up? Options: clip, saltation-free floor, or noise regularization (link to SP-02's escape noise).

## Change log

- 2026-08-13: Implemented and verified. Results in `docs/results/SP-01-experiments.md` (E1/E1b forward, E2/E3 gradient checks, E4 training, E5a–E5d edge cases).
- 2026-08-13: Key finding — two-sided finite differences are **invalid** when a firing time sits at or within ~`eps*|dt/db|` of an input-arrival time (clamped kernel-onset kink `K(d<=0)=0`) or a bias-only crossing; the analytic gradient is exact there. FD-based tests must use input-driven, well-separated firing (`_build_smooth_net`, min margin ≥ 0.13). See E5d.
