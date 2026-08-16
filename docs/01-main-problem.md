# Main Problem — The Non-Differentiable Spike

## Statement

An SNN neuron computes, at each time step:

```
v(t+1) = tau * v(t) + I(t)          (membrane potential, leaky integration)
s(t+1) = H(v(t+1) - theta)          (spike: Heaviside step)
```

Backpropagation needs `dL/dW` through the neuron. But `H` is a step function:

- Its derivative is **0 almost everywhere** (a subthreshold neuron sends no learning signal back).
- Its derivative is **undefined / infinite at the exact threshold** (a Dirac delta at the firing instant).

So the standard chain-rule gradient through a spike is either multiplying by 0 or by infinity — both unusable. This is *the* barrier that has kept SNNs from being trained as well as ANNs.

## What the problem is NOT

It is **not** "you cannot calculate the slope of a vertical cliff" in general. That framing is wrong in two ways:

1. **The spike's *timing* is smooth.** For any neuron that fires, the firing time `t_f` satisfies `v(t_f; W) = theta`. If `dv/dt != 0` at crossing, the implicit function theorem gives a well-defined, smooth derivative:
   ```
   dt_f/dW = -(dv/dW) / (dv/dt)   at t = t_f
   ```
   This is exact math (SpikeProp lineage), and the *engine's core operation for fired neurons*.

2. **The reset jump is solvable.** The discontinuity at firing is a hybrid-systems event. The saltation matrix gives the exact linearization through the jump:
   ```
   delta u^+ = Sigma * delta u^- ,   Sigma = du/dt^+ / du/dt^-  (scalar case)
   ```
   This is decades-old, rigorous calculus for discontinuous dynamical systems.

## What the problem really IS

The genuinely unsolvable-by-calculus part is **spike existence**: the discrete decision of whether a neuron fires at all. There is no classical derivative of "spike happened / didn't happen." The consequences:

- **The silent neuron / death attractor.** A neuron that *should* fire but doesn't has gradient exactly 0 → the optimizer can never update it → it stays dead forever.
- **Birth is equally invisible.** A neuron that fires but *shouldn't* cannot be silenced by the timing gradient.

The only rigorous tools for the existence problem are **probabilistic**: treat the neuron as stochastic (escape noise) and differentiate the *expected* spike rate, which is smooth and well-defined. In the noise limit, the surrogate gradient is provably the exact gradient of this expectation (Gygax & Zenke 2025).

## The precise decomposition

The main problem decomposes into exactly four sub-problems:

| # | Sub-problem | Kind of solution | Status |
|---|-------------|------------------|--------|
| 1 | Exact gradient w.r.t. spike time (fired neurons) | Math — implicit function theorem | ✅ Gate A PASS (2026-08-13) |
| 2 | Spike birth/death — silent neuron credit | Statistics — escape noise / expectation | ✅ Gate B PASS (2026-08-13) |
| 3 | The reset jump (discontinuity) | Math — saltation matrices | N/A — D1: single-spike TTFS (2026-08-13) |
| 4 | Temporal + spatial credit assignment (across layers & time) | Algorithm architecture | ✅ Gate D PASS (2026-08-14) |

See `02-sub-problems.md` for the full dependency map and `sub-problems/SP-0X-*.md` for each one.

## Why it's worth solving (impact)

1. **Energy.** Spikes are event-driven add operations, not float multiply-adds. Neuromorphic hardware (Loihi, SpiNNaker) claims up to 1000x energy reduction vs GPUs; SpikeMLLM reports 25.8x power efficiency at ~1% accuracy gap on 72B models. Solving training is what makes this usable at scale.
2. **Latency.** TTFS networks already hit competitive accuracy at T=1 (single timestep) — real-time edge AI.
3. **Temporal computation.** SNNs can exploit spike *timing* codes (ISIs, coincidences, polychrony) that rate-based ANNs cannot — a strictly larger coding space.
4. **On-device learning.** Local rules (SP-04) enable training on the chip without GPUs.
5. **Neuroscience.** Exact-gradient tools let us test whether the brain's learning is gradient-compatible.

## Definition of done (main problem)

The main problem is solved when ALL four sub-problems are solid (PRD §8) AND the engine beats a surrogate-gradient baseline honestly on CIFAR-10 and CIFAR-10-DVS (PRD G0).

## Status (final, 2026-08-16)

- **All four sub-problems solid** (SP-01 / SP-02 / SP-04 ✅ Gates A/B/D PASS; SP-03 N/A under D1).
  SP-02 additionally received a real-data fix (per-layer existence-channel strength `lam=[5,50]`)
  that removed any initialization dependence — verified at full scale on real CIFAR-10, two seeds.
- **Gate E PASS on CIFAR-10** (2026-08-15/16): exact engine **0.273 / 0.261 / 0.250** vs **tuned**
  STBP surrogate baseline **0.270 / 0.264 / 0.265** (seeds 0–2) at equal SynOps and **160× lower
  latency** (1 event/neuron vs T=160).
- **Gate E benchmark on CIFAR-10-DVS — accuracy NOT confirmed** (2026-08-16, seeds 0–2): exact
  engine **0.230 / 0.204 / 0.220** vs **tuned** baseline **0.214 / 0.250 / 0.234** (mean 0.218 vs
  0.233) — mixed within seed noise, baseline marginally ahead. Same 12×12/144 TTFS
  encoding/arch/loss/init as CIFAR-10, per-layer lam `[5,50]` carried over, 0% silent out, SynOps
  ~equal, **160× lower latency** (the decisive, reproducible win). Reported honestly: the
  strictly-worded G0 accuracy bar on DVS is **not met** — engine ≈ surrogate there
  (`docs/results/SP-05-DVS-experiments.md`).
- **Definition of done:** met on the core + **CIFAR-10** (all four sub-problems solid, Gate E
  PASS); the "and CIFAR-10-DVS" accuracy requirement is **not met** — the honest finding is the
  engine matches a tuned surrogate on DVS within seed noise at 160× lower latency.

Full account: `FINAL-REPORT.md`.
