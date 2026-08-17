# SP-02 Rigor Re-Validation — Escape-Noise Survival-Integral Gradient (2026-08-17)

Runs `python engine/experiments/exp_sp02_rigor.py` (GPU, float64).
Code: `engine/escape_rate.py` (new), `engine/snn_torch.py` (`edge_peak_guard`),
`engine/experiments/exp_sp02_rigor.py` (new). JSON: `docs/results/sp02-rigor/sp02-rigor-results.json`.

## Background
The SP-02 existence channel uses the deterministic peak-margin surrogate
`p_j = sigmoid((u_peak_j - theta)/T_noise)` — the saddle-point approximation of
the escape-noise firing probability that research doc Q2.4 deferred as future
work. `escape_rate.py` implements the deferred form **exactly** (trapezoid
quadrature on the engine grid):

    P_fire = 1 - exp(-int_0^tmax rho(u(t)) dt),   rho(u) = rho0 * sigmoid((u-theta)/T)
    dL/dW_ji = -(S/P_fire) * int_0^tmax rho'(u(t)) K(t - t_in_i) dt,   L = -log P_fire

## Check A — channel vs exact escape-noise gradient
FD-validated to **rel err ~1e-9..1e-6** (float64, central diff on the dominant
input): the quadrature gradient `dP/dW` and loss gradient `dL/dW` are exact.

For the dominant input the gradient ratio admits the **closed-form
decomposition** (exact algebra, residual ~1e-12):

    g_esc / g_chan = f(S) * C,     f(S) = -S*log(S)/(1-S),    C = kernel-geometry term

measured over margins m0 in {-0.1..-0.9} and escape rates rho0 in {1e-3..1}:

| m0 | S~0.99 | C_emp | C_theory |
|----|--------|-------|----------|
| -0.10 | 0.985 | 0.692 | 0.692 |
| -0.25 | 0.986 | 0.670 | 0.670 |
| -0.50 | 0.987 | 0.635 | 0.635 |
| -0.75 | 0.988 | 0.605 | 0.605 |
| -0.90 | 0.989 | 0.588 | 0.588 |

Key verified facts (gates A pass):
- **S-dependence is exactly f(S)** (C_emp constant across the 1e-3..1.0 rate
  sweep, spread < 0.02; decomposition residual ~1e-12).
- **Saddle limit**: C -> 1 as T_esc -> 0 (C_emp = 0.965/0.934/0.819/0.718 at
  T_esc = 0.05/0.1/0.25/0.5). The channel is therefore the **exact expected
  gradient** in the double limit far-dead (S->1) AND narrow spike (T_esc->0).
- **Direction is exact**: cosine similarity 0.999 across all rows (the channel
  is a magnitude-scaled copy of the true gradient).
- **Magnitude bias**: at T_esc = 1 the channel over-pushes silent neurons by
  `1/(f(S) C)` ~ 1.4-1.7x (far-dead) up to ~10x (near-threshold). This is the
  first quantitative statement of how far the sigmoid channel is from the true
  escape-noise expected gradient; it is a monotone, bounded, closed-form bias —
  the channel is safe as a revival prior but not the true gradient at finite
  temperature.

## Check B — envelope theorem at boundary extrema
Brute-force FD `d(u_peak)/dW_ji` vs the envelope `K(t_peak - t_in_i)`:
- interior-max extremum: rel err **7.3e-9** (matches E8);
- strictly-all-negative interior-min extremum: rel err **1.1e-9** (the doc's
  min-branch claim is verified once the potential is strictly below zero);
- **cross-branch discontinuity (u_max = 0 exactly)**: perturbing the bias of a
  neuron whose u_max == 0 flips the all-negative/min branch to the max branch;
  FD rel err ~1.1e5. This is a *documented channel non-smoothness* (extremum
  selection is not a smooth function of W at u_max = 0), not an envelope
  failure within a branch.

The boundary-extremum analysis also resolves the concern about "peaks at the
window start": u(t_start) = 0 always (causal K(0) = 0), so a max at t_start has
u_peak = 0 and falls into the all-negative branch; the envelope is valid within
a branch. The only genuine deadlock is the **degenerate u(t) = 0 plateau**.

## Check C — degenerate plateau guard
All-zero weights => u(t) = 0: channel gradient max-abs 3.0e-3 (effectively
deadlocked; the residual is golden-section plateau drift) vs exact escape
gradient max-abs **0.223 (76x larger)** — the escape model revives these
neurons, the channel cannot. `edge_peak_guard` flags them (guard=True), so the
channel never injects a dead/zero signal. `n_edge_guarded` is recorded per layer
in existence stats; smoke-tested in both `existence_grads` and
`local_learning_grads` (ref and deep paths).

## Gates
All 6 pass:
- `A_fd_gradient` true (max fd rel err < 1e-4)
- `A_s_dependence` true (C rho0-independent; decomposition exact)
- `A_saddle_limit_C_to_1` true (C -> 1 as T_esc -> 0)
- `A_far_dead_bounded` true (cos > 0.99, ratio < 1)
- `B_envelope_within_branch` true (rel err < 1e-6)
- `B_cross_branch_discontinuity_documented` true

## Interpretation for the repo
1. The SP-02 channel is **not wrong**, it is a *magnitude-biased* version of the
   exact escape-noise expected gradient, with the bias now in closed form
   (f(S) * C, both computable). Direction exact to cos 0.999.
2. The exact escape gradient is available (escape_rate.py) for future
   loss-corrected training (L_exist = -log P_fire with per-layer T_esc), and
   its survival-integral form is the defensible "Option A stochastic forward"
   deferred in the SP-02 research doc.
3. Boundary extrema are sound within a branch; the degenerate plateau is now
   guarded; the cross-branch u_max=0 discontinuity is documented as a known
   non-smoothness.
