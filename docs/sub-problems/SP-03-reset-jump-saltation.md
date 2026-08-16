# SP-03 — The Reset Jump: Saltation Matrices

**Status:** ⚪ Not Applicable — **D1 (2026-08-13): single-spike (TTFS)** · **Phase:** 3 (skipped) · **Gate:** C (N/A, see `docs/tracking/GATES.md`)

## Precondition (D1) — RESOLVED

This sub-problem is only needed if the engine uses **multi-spike** neurons (a neuron can fire more than once per trial).

- ✅ **D1 decided 2026-08-13: single-spike (TTFS)** — neurons are dead after their first spike, the reset never perturbs later dynamics, and SP-03 is declared **Not Applicable**. The phase is skipped and SP-04 proceeded (Gate D PASS, 2026-08-14).
- If D1 is ever revisited to multi-spike: SP-03 is required. The full derivation and acceptance criteria below remain valid and ready.

## Problem statement

When a neuron fires, its membrane potential is reset (`u -> u_reset` for hard reset, or `u -> u - theta` for additive reset). This is a discontinuity in an otherwise continuous dynamical system. Standard backprop — which assumes smooth flow — misses the coupling this jump creates between the pre-jump state and everything that happens after.

Concretely: if a neuron fires twice, the second spike time `t_f2` is a function of the reset state, which is itself a function of `t_f1` and the pre-jump state. Perturbing a weight shifts `t_f1`, and that shift is *transported* through the reset jump into `t_f2`. The correct sensitivity must include the jump's linearization.

## The math (saltation matrix)

This is solved, classical math from hybrid dynamical systems (Aizerman–Gantmakher).

### Setup

- Flow before crossing: `du/dt = f^-`
- Discontinuity surface: `h(u) = u - theta = 0`, normal `n = h_u = 1` (scalar case)
- Jump map: `g(u)` — `u_reset` (hard) or `u - theta` (additive)

### Saltation matrix

```
Sigma = g_u + (f^+ - g_u * f^-) * n^T / (n^T * f^-)
```

For the scalar membrane with hard or additive reset (`g_u = 0` or `1`), both reduce to the same scalar:

```
Sigma = du/dt^+ / du/dt^-
```

A perturbation `delta u^-` just before the crossing becomes `delta u^+ = Sigma * delta u^-` just after.

### The subtle part (the "state-dependent limit")

`du/dt^-` and `du/dt^+` are NOT "before/after" of the same trajectory:

- `du/dt^-` is the derivative of the pre-jump trajectory evaluated AT `t = t_f` as it crosses threshold.
- `du/dt^+` must be evaluated AFTER applying the jump map, with the reset initial condition `u_reset` — i.e., the derivative of the *post-reset* trajectory (same incoming input current, new initial membrane state) evaluated at `t = t_f`.

For the double-exponential PSP kernel, `du/dt^+` is analytic: it's the derivative of the membrane response starting from `u = u_reset` at `t = t_f`, which includes an explicit decay term from the reset initial condition. This formula must be written down in full before implementing.

### Adjoint through the jump

In the backward pass, the adjoint `lambda` maps through the jump as:

```
lambda^- = Sigma^T * lambda^+
```

For a neuron with multiple spikes, iterate backward over the neuron's OWN spike times, applying `Sigma` between consecutive spikes, THEN accumulate into pre-synaptic `lambda` via `lam_prev += alpha * W * K'`.

**Known failure mode (grazing):** if `du/dt^- ~ 0` (tangential crossing), `Sigma` blows up. Handle via a floor (like the engine's `denom` clamp) or via noise regularization from SP-02. Must be tested, not assumed.

## Requirements / acceptance (GATE C)

- [ ] Full analytic derivation of `Sigma` for the chosen reset map, including the post-reset `du/dt^+` formula.
- [ ] Implemented in the backward pass (adjoint jump application between a neuron's own spikes).
- [ ] **Jump-map gradient check:** perturb a weight; compare saltation-predicted shift in the 2nd spike time vs finite differences. Must match to relative error < 1e-4.
- [ ] Grazing case (`du/dt^- ~ 0`) tested and behavior documented (no NaN, no divergence).
- [ ] No regression of SP-01 and SP-02 gradient checks.

## Open questions

- Q3.1: For additive reset (`u -> u - theta`), `du/dt` is continuous across the jump (membrane derivative doesn't change, only the value drops). Does `Sigma = du/dt^+ / du/dt^- = 1` then? If so, additive reset needs NO saltation correction and the phase collapses. This MUST be checked — the "state-dependent limit" matters only if the *derivative* jumps, which happens for hard reset to a fixed value, not additive reset.

## Change log

- 2026-08-14: Declared **Not Applicable** — D1 (single-spike TTFS) resolved 2026-08-13; Gate C marked N/A in GATES.md; derivation retained for a possible multi-spike revisit.
- (prior: none)
