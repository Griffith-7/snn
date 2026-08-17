# SP-03 Saltation Jump-Map Re-Validation (2026-08-17)

Runs `python engine/experiments/exp_sp03_saltation.py` (CPU, float64 — closed-form
math, no ODE-solver error). Code: `engine/reset_lif.py` (new),
`engine/experiments/exp_sp03_saltation.py` (new).
JSON: `docs/results/sp03-saltation/sp03-saltation-results.json`.

## Background
Decision D1 (2026-08-13) fixed the production engine to **single-spike TTFS**, so
Gate C (reset jump / saltation) was marked N/A — there is no reset jump to
saltate. This session re-opens Gate C on the **minimal multi-spike model** that
makes the reset jump real:

    u' = (i - u)/tm,   i' = -i/ts,   hard reset u(t_f+) = u_reset at firing,
    inputs = delta-current impulses (t_k, w_k);  multiple spikes per neuron.

All propagation between events is closed-form, so spike times are exact roots of
`u(t) = theta` (bracketing scan + bisection + Newton polish); no solver error
contaminates the FD comparison. Config: tm=15, ts=4, theta=1, u_rest=u_reset=0,
inputs [(0,8),(1,5),(2,3)] → fires at 1.873, 3.506, 6.426.

## Saltation matrix (derived, then verified)
For state `x=(u,i)`, event `g = u - theta`, reset `R(u,i) = (u_reset, i)`:

    Xi = dR/dx + (f^+ - dR/dx f^-) dg/dx / (dg/dx f^-)

With hard reset to u_rest=0 the i-component is reset-immune, so its row is the
identity and **there is no u→i coupling** (the first version of this module
claimed an `Xi_iu = i_f*tm/(ts*(i_f-theta))` term — that is wrong; the FD check
caught it: s_i is continuous through the reset because i is untouched by it).
The u-row reduces to the scalar EventProp factor:

    Xi_uu = u'_f+ / u'_f- = i_f/(i_f - theta)

    s_u^+ = Xi_uu s_u^-,   s_i^+ = s_i^-,   dt_f/dw = -s_u(t_f^-)/u'(t_f^-).

## E1 — forward invariants
3 event-driven fires. For each fire `t_f`: `u(t_f - 1e-9) = theta` (pre-fire root
exactness) and `u(t_f + 1e-9) = u_reset` (reset applied) — both within 1e-6
(actually ~1e-9). Fires strictly increasing. **PASS.**

## E2 — fixed-time saltation vs central FD
At `t_eval = t_f1 + 0.5*(t_f2 - t_f1)` (between first and second spike, i.e. with
one reset behind us), perturb `w0` by ±1e-5 and central-difference the state:

| sensitivity | analytic | FD | rel err |
|-------------|----------|-----|---------|
| s_u = du/dw0 | — | — | **1.83e-10** |
| s_i = di/dw0 | — | — | **6.80e-11** |

**No-jump control** (propagate s through the reset as the identity): rel err
**8.51e-2** on u — the control FAILS by four orders of magnitude, proving the
saltation is necessary, while s_i is unaffected (rel 6.8e-11) — the independent
confirmation that the i-row is the identity. **PASS.**

## E3 / E3b — spike-time sensitivity vs central FD
`d(t_f2)/dw0`: analytic −0.312141 vs FD −0.312141, **rel 2.65e-10**.
Full sweep (every spike × every weight, w0 and w1): **max rel 3.39e-10** ≪ 1e-4.
**PASS (Gate C criterion).**

## E4 — grazing
Bisected the exact-graze weight `w_graze = 6.064154` (single input: the weight
whose peak u exactly touches theta). At `w_graze + 1e-7` the analytic
`dt_f/dw` is finite but large (−1.021e3), **no NaN**; the exact graze is flagged
`±inf` (not NaN). **PASS.**

## E6 — general reset map u -> u_reset (control)
At an *exact* threshold crossing `u(t_f) = theta`, additive reset `u -> u - theta`
coincides with hard reset `u -> 0`, so the meaningful generalization is a nonzero
`u_reset`, with `Xi_uu = u'_f+/u'_f- = (i_f - u_reset)/(i_f - theta)`:

| u_reset | rel_u (fixed-time) | rel_i | rel dt2 (spike-time) |
|---------|--------------------|-------|----------------------|
| −1.0 | 1.44e-10 | 6.99e-11 | 3.09e-11 |
| +0.5 | 2.02e-10 | 4.16e-12 | 3.17e-10 |

**PASS.** (This control caught a real bug: the first version hardcoded
`Xi_uu = i_f/(i_f-theta)`, the u_reset=0 special case; the general formula
`(i_f-u_reset)/(i_f-theta)` restores FD agreement — the second bug this check
has caught.)

## E5 — forward oracle
Fixed-step explicit-Euler integration of the same ODE + reset (step 1e-4) finds
crossings at 1.873, 3.506, 6.426 — every event-driven fire matches the nearest
oracle crossing within 5e-3 (a few Euler steps). **PASS.**

## Gates
| gate | check | result |
|------|-------|--------|
| E1 | forward invariants (root exactness, reset applied) | ✅ |
| E2 | fixed-time saltation FD rel < 1e-6 | ✅ 1.8e-10 / 6.8e-11 |
| E2b | no-jump control must fail (rel_u > 1e-3) + i-row identity | ✅ 8.5e-2 / 6.8e-11 |
| E3 | spike-time dtdw FD rel < 1e-4 | ✅ 2.65e-10 |
| E3b | all-spike × all-weight max rel < 1e-4 | ✅ 3.4e-10 |
| E6 | general u_reset (nonzero) fixed-time + spike-time | ✅ ~1e-10 |
| E4 | grazing documented, no NaN | ✅ |
| E5 | forward oracle agreement | ✅ |

## Interpretation
Gate C is **CONFIRMED for the minimal multi-spike hard-reset LIF**: the saltation
jump map is exact and FD-verified at machine-precision-plus (1e-10 level), the
scalar EventProp factor `u'_f+/u'_f- = (i_f-u_reset)/(i_f-theta)` is recovered
for general u_reset, the i-row identity is confirmed (it corrected the initial
wrong coupling term), and the no-jump control fails at 8.5e-2 — i.e. for
multi-spike LIF with reset, **ignoring the jump map is wrong**. At exact
threshold crossings the additive reset `u -> u-theta` is identical to hard reset
to 0 (both give u+ = 0), so Q3.1 collapses into the u_reset generalization
tested here. Production engine note: D1 stands, so the single-spike engine has
no reset jump; this closes the rigor gap for the general multi-spike case that
D1 intentionally deferred.
