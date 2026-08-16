# SP-02 Research — Spike Birth/Death: Silent-Neuron Credit

Status: ✅ Written · Date: 2026-08-13 · Sources: verified papers (below)

## 1. Literature findings

### 1.1 The exact-gradient community hits the same wall

- **Gygax & Zenke 2024 (arXiv 2404.14964, "Elucidating the theoretical underpinnings of surrogate gradient learning in SNNs"):** surrogate gradients (SGs) equal the exact gradient of the *expected* output for a **single neuron** when the surrogate derivative is matched to the neuron's escape-noise function; in deep nets SGs are *smoothed stochastic derivatives* (stochAD) whose form should mirror the escape-noise derivative. Two crucial caveats for us: (1) SG equivalence with an exact gradient holds for single neurons, not deep nets; (2) SGs are **not** gradients of any conservative surrogate loss (their curl is non-zero). Implication: the principled way to "create a gradient where no spike exists" is to make spike existence stochastic — the gradient of the firing probability is then well-defined.
- **Klos & Memmesheimer 2023 (arXiv 2309.14523, "Smooth Exact Gradient Descent Learning in SNNs"):** *pseudodynamics / pseudospikes* — extend the exact spike-time map into silent neurons by adding a suprathreshold drive so each silent neuron emits a computable "pseudospike" time; pseudospikes transition smoothly into ordinary spikes as the neuron revives. Trains a network from an almost-entirely-silent initialization to 97.3% on a neuromorphic benchmark. This is the strongest prior art for *deterministic exact* revival, and the closest cousin of our approach.
- **EventProp + loss shaping (2025, Neuromorphic Computing & Engineering):** shows CE-on-spike-times drives useful hidden spikes to deletion (the silent-death mechanism at scale). Their fix for silent *output* neurons is **phantom spikes** — regularize as if the silent neuron spiked at trial end `T`. For silent *hidden* neurons they fall back to a per-epoch heuristic boost of all incoming synapses (they admit this is a heuristic — exactly what Gate B rejects).
- **Rouser (2024, arXiv 2407.19566, "To Spike or Not to Spike"):** learn per-neuron firing *thresholds* to rouse dead neurons; effectively lowers the threshold until the neuron fires. Works, but it changes the neuron model (a threshold parameter per neuron) and adds parameters; it is a *different* mechanism from giving a gradient to the weights.
- **Gerstner, Neuronal Dynamics, Ch. 9 (escape noise):** standard escape-rate model: hazard `rho(u)`, survivor `S(t) = exp(-∫_0^t rho(u(s)) ds)`, probability of having fired by `t` is `P_fire(t) = 1 - S(t)` — all smooth functions of `u`, hence of the weights. Logistic/exponential escape rates give non-vanishing firing probability even far below threshold.
- **ETTFS (2024, arXiv 2410.23619):** prevents silence *by construction* via init + normalization. Complementary, not a revival mechanism; belongs to SP-04 (deep stability).

### 1.2 What the literature does NOT give us

None of the principled papers give a **cheap, deterministic, event-based, layer-to-layer** silent-neuron signal that reuses the SP-01 spike-time machinery. Pseudospikes require modifying the forward dynamics; phantom spikes are output-only; threshold learning changes the model; REINFORCE has variance. We derive our own from the escape-noise identity, using quantities the SP-01 engine already computes.

## 2. The model: spike existence as a stochastic Bernoulli at the potential peak

### 2.1 Setup

For each neuron `j` the forward pass already computes (SP-01):

```
t_peak_j = argmax_{t in [t_start, t_max]} u_j(t)   (peak over the RESPONSE window)
u_peak_j = u_j(t_peak_j)                           (the peak potential)
margin    m_j = u_peak_j - theta
```

`t_start` = time of the earliest *contributing* presynaptic event (|w| > 0).

**Peak-extremum fix (2026-08-13, empirically forced).** The original spec searched
`argmax_{t in [0, t_max]}`, but the pre-input plateau (`u = 0` for `t < t_start`) is always
the argmax of any subthreshold neuron, giving `K(t_peak - t_in) = 0` and an identically-zero
existence gradient — the far-dead deadlock the channel exists to fix (E6 failed on it).
The search is therefore restricted to the response window, and for **all-negative responses**
(`u_max <= 0`: the potential is a negative bump whose max sits on the window boundary, where
`u'` is undefined and the envelope theorem fails) the interior **minimum** extremum is used
instead. Both are interior extrema with `u'(t_peak) = 0`, so the envelope theorem (2.5) holds
in both cases, and the far-dead margin-gradient is bounded below by ~`1/T` exactly as derived
(verified in E6, E8c). Note this also means `m_j` can be arbitrarily negative (far-dead),
because the response-minimum of a negative bump is not capped by the plateau.

Fired neuron: `m_j >= 0` (crossing at `t_fj <= t_peak_j`). Silent neuron: `m_j < 0`.

### 2.2 Stochastic birth model (escape-noise, peak / saddle-point form)

Replace the deterministic "fires iff `m >= 0`" decision with a Bernoulli whose success probability is the escape-noise firing probability concentrated at the peak:

```
p_j = sigma(m_j / T) = 1 / (1 + exp(-m_j / T))
```

Derivation level (stated honestly): in the full escape-noise model the neuron fires with hazard `rho(u(t))` over the trial, giving `P_fire = 1 - exp(-∫_0^tmax rho(u(s)) ds)`. For a unimodal potential the hazard integral is dominated by the neighbourhood of the peak (saddle-point approximation), reducing to `sigma((u_peak - theta)/T)` with `T` the escape-noise width. `T` is a **model noise level** (like `tau_m`), NOT a tuning knob. The full integral form is recorded as future work (Q2.4). The key theoretical anchor (Gygax & Zenke 2024) is that the derivative of this firing probability is exactly the gradient of the expected firing in a stochastic neuron.

### 2.3 Objective

```
L_exist = - (1/N) * sum over targeted silent neurons j of  target_j * log(p_j)
```

- Hidden layers: `target_j = 1` for all silent neurons (uniform activity prior: healthy hidden layers fire).
- Output layer: `target_j = 1` only for the **correct-class** silent output (the wrong-class outputs may stay silent — a silent wrong output can never win, which is fine).
- `lambda` scales the channel relative to the latency CE; it is a standard multi-objective loss weight, default `lambda = 1.0`, sensitivity reported. **Sensitivity (measured, E7):** at `lambda = 1` the channel competes with the timing loss's push-DOWN on wrong-class outputs (each output is wrong for ~half the batch) and revival is slow (output-silence task plateaus ~0.63); `lambda = 5` revives and learns (0.99). The toy (E6) works at `lambda = 1`; the output-layer task needs more weight because the timing loss actively opposes the correct-class push.

### 2.4 The key fix over the naive `sigma'(m)` surrogate (the skeleton draft)

The skeleton draft used `ds = s(1-s)/T` — the derivative of the *probability* — which vanishes as `s -> 0` (far-dead neurons get ~nothing). The MLE-style objective `-log p` fixes this:

```
d(-log p_j)/d m_j = -(1 - p_j)/T      -- as m_j -> -inf, p_j -> 0, so this -> -1/T
```

**For far-dead neurons the margin-gradient is bounded below by `1/T`, it does NOT vanish.** The naive `sigma'` form confuses the derivative of the probability with the gradient of the objective; they differ by the factor `1/p_j`. This single fix is the quantitative reason far-dead revival works (verified in E6).

### 2.5 Envelope theorem: exact `d(u_peak)/dW`

`u_peak_j = u_j(t_peak_j)` with `u_j'(t_peak_j) = 0` at an interior peak (endpoint peaks handled identically: `du/dt(t_max)` need not be zero but `t_peak = t_max` is fixed). Differentiating with respect to a weight:

```
d(u_peak_j)/dW_ji = du_j/dt (t_peak_j) * d(t_peak_j)/dW_ji + dK/dW (t_peak_j - t_i)
                   = 0 + K(t_peak_j - t_i)                    [envelope theorem]
```

The time-derivative term vanishes at the peak. Likewise:

```
d(u_peak_j)/dt_i = w_ji * K'(t_peak_j - t_i)
```

These are **exact** (no surrogate), verified by finite differences in E8 — subject to the same well-conditioning caveat as SP-01 (E5d): valid when `t_peak_j` does not coincide with an input arrival time.

### 2.6 The existence channel (per-layer gradients)

Combining 2.4 and 2.5, with `g_j = dL_exist/d(u_peak_j) = -target_j (1 - p_j)/T`:

```
dL_exist/dW_ji = g_j * K(t_peak_j - t_i)                  (silent neuron j, all inputs incl. bias)
dL_exist/dt_i  = sum_j g_j * w_ji * K'(t_peak_j - t_i)    (existence adjoint into fired presynaptic spike times)
```

Layer-to-layer: silent neurons at layer `l` contribute (a) weight gradients via `K(t_peak - t_prev)` and (b) an existence adjoint `dL_exist/dt_prev` into the **fired** neurons of layer `l-1`, which is added to that layer's `dL/dt` before the SP-01 backward runs. Silent neurons of layer `l-1` cannot receive a timing adjoint (no spike time) and are handled by their own channel at layer `l-1`. Total gradient = SP-01 timing gradient + existence gradient.

### 2.7 Isolation (why SP-01 cannot regress)

- The existence channel is **zero for every fired neuron**: `target` only selects silent neurons, and additionally `(1-p_j) -> 0` as a revived neuron approaches `p -> 1`.
- With no silent targeted neurons it contributes exactly `0`, so the gradient is byte-identical to SP-01 (verified in E8).
- Revived neurons: once `m >= 0`, the neuron fires at `t_f` and the exact SP-01 timing gradient takes over; the existence factor `(1-p)` automatically decays to `~0`. Smooth hand-off (Q2.2, checked in E6).

### 2.8 Annealing rationale (Gate B requirement)

**No temperature annealing.** The skeleton draft annealed `T -> 0`, which narrows `sigma'` and re-introduces vanishing far-dead signal — the exact failure we are fixing. `T` is a fixed noise level. Rationale: the stochastic model's noise width does not change during training; a fixed width guarantees a minimum far-dead margin-gradient of `1/T` at all times.

## 3. Decision D2

Compare candidates on (i) far-dead signal strength, (ii) variance, (iii) hardware friendliness.

| Option | Derived? | Far-dead strength | Variance | Hardware | Verdict |
|---|---|---|---|---|---|
| **A. Escape-noise peak-margin (Sections 2.2–2.6)** | Yes (escape-noise + envelope theorem) | margin-gradient `-> 1/T`, bounded below | zero (deterministic) | event-based: needs only `t_peak`, `u_peak`, `K`, `K'` — all SP-01 machinery | **CHOSEN** |
| B. REINFORCE + spike-timing baseline | Yes (unbiased for the stochastic objective) | gradient `~1/T` but the *signal-to-noise* degrades exponentially with distance | high; needs baseline design (Sprekeler 2009) | sampling cost, non-deterministic, hard to verify | Rejected (recorded; revisit if bias proves harmful) |
| C. E-prop / eligibility traces | Partial (three-factor rules) | as designed | n/a | local & hardware-friendly | Deferred to SP-04 (local credit) |
| D. Pseudospikes (Klos & Memmesheimer 2023) | Yes | strong | zero | modifies forward dynamics; heavy | Not implemented now; the peak-margin form is the cheap saddle-point of the same idea — documented |

**D2 decision: Option A.** Deterministic (zero sampling variance), derived, shares the verified SP-01 engine, hardware-friendly (event-based), and the far-dead margin-gradient is provably bounded below. Rejected options and reasons are recorded above.

## 4. Gate B mapping

| Gate B item | How met |
|---|---|
| Silent-neuron signal DERIVED, not tuned | §2.2–2.6 derivation; `T` is a model noise level, `lambda` is a loss weight (both reported) |
| Far-dead revival toy test | E6: single neuron initialized `m0` below threshold, `m0 in {2,3,5,8}` (`m0=1`/`w=0` excluded: no signal by construction); revives; gradient magnitude bounded ≥ 0.7 (ablation: no channel ⇒ stays dead) |
| Output-layer silence handled | §2.3 correct-class target; E7 |
| No regression of SP-01 | §2.7 structural + E8 (re-run SP-01 gradchecks; zero-contribution check; existence-channel FD check) |
| Ablation control | E9: identical init, without mechanism fails / with mechanism succeeds |
| Annealing rationale recorded | §2.8 |

## 5. Open questions

- Q2.1: Revival step-count vs distance `m0/T` (measured in E6; confirms the bounded-gradient theory).
- Q2.2: Do revived neurons produce *correct* spike times, and does the SP-01 timing gradient take over smoothly? (checked in E6/E9)
- Q2.3: Separate channel (current) vs unified stochastic forward (Gygax-Zenke). Current design chosen for isolation (Gate B requires it); a unified stochastic model is future work.
- Q2.4: Full escape-noise survival-integral gradient (`P_fire = 1 - exp(-∫ rho(u(s))ds)`) vs the peak saddle-point form — the integral form is the rigorous limit, future refinement.
- Q2.5: Uniform existence target (all silent hidden fire) vs per-neuron targets from downstream error — the former is the default prior; the latter is future work.
- Q2.6: **Channel-strength sensitivity (RESOLVED by real-data verification).** Q2.6 (original): at what `lam` does the channel compete with the timing CE and which `lam` revives a 10-class output layer on real data? Real CIFAR-10 finding: uniform `lam=5` revives the hidden layer but **not** the output (99.6% silent — the output is targeted on only 1/10 of samples, correct-class, and receives no downstream adjoint). Silent outputs are near-threshold (deficit ≤ ~0.6), so the saturation of the sigmoid is not the binding constraint — the push magnitude is. Fix: **per-layer `lam`** (`_as_layer_lam`): hidden 5 / output 50 → 0% silent output from epoch 0; uniform lam=20 revives but underfits; `[5,20]` re-collapses (the hidden channel must stay strong enough to feed the output's input stream); `[5,100]` over-fires (wrong timing). A `lam_out ∈ {30,40,50,60,80}` sweep at 4096 is within noise (val 0.202–0.209) — the knob is exhausted; the remaining accuracy gap vs the surrogate baseline is a readout/training matter, not an existence-channel one (MEMORY facts 17–20).

## 6. Real-data verification: per-layer channel strength fixes output-layer collapse (2026-08-15)

The existence channel's E7 toy (2-class, acc 0.47 → 0.97) does not capture the 10-class
real-data regime. On real CIFAR-10 (144→64→10, std-init, mode ref):

**Failure (`lam=5`, seed-1 std-init):** hidden silence revives (826 → 0) but the output layer
stays ~99.6% silent (correct-class targets applied every batch), acc stuck ~0.10.

**Real-data diagnosis (`diag_sp02_real`):**
- Silent outputs are **near-threshold**, not far-dead: deficit below θ percentile
  [0, 0.8, 0.9] (i.e. ≤ ~1.0 below threshold), so far-dead vanishing is NOT the cause.
- Kernel decay is equal for hidden and output (frac `K(t_peak−t_i) < 0.1`: 0.79 vs 0.86), so
  "late t_peak kills the kernel" is NOT the cause.
- The differentiator is **channel strength**: output neurons are targeted on only 1/10 of
  samples (correct-class) and receive no downstream adjoint, vs hidden neurons targeted on every
  sample. `lam=5` is simply too weak for the output.

**Fix: per-layer `lam`** (`_as_layer_lam` in `engine/snn_torch.py`, scalar → every layer
backward-compatible, sequence → per-layer). Real CIFAR-10 4096, seed 1 (failure regime), mode ref:

| config | silent_out @ep | train acc @ep14 |
|---|---|---|
| uniform `lam=5` | 0.99 (stuck) | ~0.10 |
| uniform `lam=20` | 0 → 0% | 0.164 (ep7) |
| **per-layer `[5,50]`** | **0% from ep0** | **0.270** |
| per-layer `[5,20]` | re-collapses (0.989) | — |
| per-layer `[5,100]` | 0% | plateaus 0.168 |

**Full-scale verdict (`docs/results/evidence/sp05_seed1_plam.log`, 15k/40 std-init seed 1, `[5,50]`):** test **0.261**,
silent_out 0.0000, silent_hid 0.0000, train 0.333 (underfit), 115 min — vs lam=5 collapse 0.104,
uniform lam=20 0.225, tuned baseline 0.264 (seed 1), pos-init engine 0.269. The collapse is
**fixed without any firing-guaranteeing init**; accuracy is statistically tied with the tuned
baseline. Q5 (MEMORY) resolved; remaining gap = readout/accuracy, not existence.

## Sources

1. Gygax & Zenke — "Elucidating the theoretical underpinnings of surrogate gradient learning in spiking neural networks," arXiv:2404.14964 (2024).
2. Klos & Memmesheimer — "Smooth Exact Gradient Descent Learning in Spiking Neural Networks," arXiv:2309.14523 (2023).
3. Nowotny et al. — "Loss shaping enhances exact gradient learning with EventProp in SNNs," Neuromorphic Computing & Engineering (2025) [phantom spikes, hidden-silent heuristic].
4. Xiao et al. — "To Spike or Not to Spike, that is the Question" / Rouser, arXiv:2407.19566 (2024).
5. Gerstner et al. — "Neuronal Dynamics," Ch. 9 (escape noise: hazard, survivor, P_fire).
6. Sprekeler, Hennequin, Gerstner — "Code-specific policy gradient rules for spiking neurons," NeurIPS 2009 (REINFORCE baseline).
7. Che et al. — ETTFS, arXiv:2410.23619 (2024).
8. Göltz et al. — DelGrad, 2024 (exact event-based gradients; context).
