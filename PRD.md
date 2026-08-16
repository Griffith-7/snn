# PRD — Exact Event-Based SNN Training Engine

**Version:** 0.1 (initial)
**Author:** AI researcher
**Status:** Draft

## 1. Problem statement

Spiking Neural Networks (SNNs) are energy-efficient and event-driven, but cannot be trained as well as standard ANNs because the spike function is non-differentiable. Standard backpropagation breaks; surrogate gradients are biased ("fake math") and leave SNNs behind ANNs on complex tasks (language, high-res images).

**This project's thesis:** the problem is not "you cannot differentiate a cliff." It is a set of four solvable sub-problems. Exact math exists for the spike's timing and for the reset jump; only spike *birth/death* has no classical derivative, and probability theory handles that. Solve all four, correctly and verifiably, and SNNs become trainable.

## 2. Goals (what "done" means)

### 2.1 Functional goals

1. **G0 (North Star):** A trainable SNN engine whose accuracy on a real benchmark (CIFAR-10 or CIFAR-10-DVS) is at least as good as a surrogate-gradient baseline (e.g., STBP/SLAYER), at equal or better energy/latency.
2. **G1:** Exact gradient w.r.t. spike times for all *fired* neurons, verified to numerical precision (target: gradient-check against finite differences, relative error < 1e-4, ideally < 1e-6).
3. **G2:** A principled, non-heuristic mechanism that gives *dead* (silent) neurons a useful, unbiased learning signal — grounded in escape-noise / expectation theory, not a hand-tuned sigmoid.
4. **G3:** Exact handling of the reset discontinuity via saltation matrices (only required if the engine allows multi-spike neurons).
5. **G4:** Temporal + spatial credit assignment that is correct AND cheap enough for neuromorphic hardware (no full-trajectory BPTT storage), without losing accuracy.

### 2.2 Non-goals (explicitly out of scope)

- Claiming to "differentiate the vertical cliff" — that is mathematically impossible and we do not attempt it.
- Competing with LLM-scale training in this project's lifetime.
- Biological plausibility for its own sake (we only require it when it coincides with the technical solution).

## 3. Users / stakeholders

| Stakeholder | Need |
|---|---|
| The researcher (you) | A rigorous, incremental path with each step verifiable |
| Neuromorphic hardware people | A learning rule that runs locally, online, cheap memory |
| ML community | A method that beats surrogate baselines honestly |

## 4. Architecture summary

The engine combines (planned):

- **FastMSNN forward:** exact event-based simulation with analytic root-finding of spike times (double-exponential PSP kernel).
- **IFT backward:** implicit-function-theorem gradients for fired neurons.
- **Revival channel:** isolated mechanism for silent neurons (SP-02).
- **Saltation matrices:** exact jump linearization for resets (SP-03).
- **Adam optimizer** with the above gradient sources.

## 5. Success metrics / acceptance criteria

| Metric | Target |
|---|---|
| Gradient correctness (fired neurons) | finite-difference relative error < 1e-4 |
| Silent-neuron signal | dead neurons can be revived on a controlled toy problem (e.g., XOR with dead init), and it is principled (derived, not tuned) |
| Accuracy CIFAR-10 | within 1 point of STBP/SLAYER surrogate baseline |
| Accuracy CIFAR-10-DVS | at or above surrogate baseline |
| Training memory | O(T) memory only if T small; prefer O(1)-per-neuron eligibility traces for SP-04 |
| Reproducibility | fixed seeds, published configs, single-file runnable |

## 6. Milestones (aligned with PLAN.md)

- **M1:** SP-01 solved + gradient-verified on a 2-layer TTFS network. Gate: pass Section 2.1 G1.
- **M2:** SP-02 solved + principled revival demonstrated. Gate: pass G2 + revival toy test.
- **M3:** SP-03 solved (if multi-spike is adopted) — saltation verified. Gate: G3 + jump-map gradient check.
- **M4:** SP-04 solved — correct, local, cheap credit assignment. Gate: G4.
- **M5:** Full engine beats surrogate baseline on CIFAR-10 and CIFAR-10-DVS. Gate: G0.

## 7. Risks and open questions

| Risk | Impact | Mitigation |
|---|---|---|
| Exact spike-time gradients die on silent neurons (classic SpikeProp failure) | High | SP-02 revival must reach far-dead neurons, not just near-threshold (current draft does NOT — see SP-02 file) |
| Multi-spike regime decides whether saltation is needed | Medium | Decide single-vs-multi-spike early (PLAN gate) |
| Bias gradient uses impulse kernel instead of step response | Medium | Fix in engine; covered by gradient-check test in M1 |
| No loss head for silent output neurons | High | SP-02 must include output-layer death handling |
| Beating surrogate baselines is an empirical question | High | Benchmark from M5; do not claim victory before it |

## 8. Definition of "solid" (gate rule)

A sub-problem is **solid** when ALL of:

1. It has a written, precise statement (its SP file).
2. It has a derived solution with math written down (not just code).
3. It has a passing verification test (gradient check, toy problem, or ablation).
4. It has a documented decision + what was rejected and why (in MEMORY.md).
5. It does not regress any previously solved sub-problem.

No sub-problem N+1 starts until N is solid.

## 9. Out of scope for v1 (future)

- Spiking LLMs
- ANN-to-SNN conversion pipelines
- Convolutional spike kernels beyond a simple front-end
- Full on-chip learning hardware port
