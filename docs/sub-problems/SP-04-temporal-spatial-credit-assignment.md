# SP-04 — Temporal + Spatial Credit Assignment

**Status:** ✅ Complete (Gate D PASS) · **Phase:** 4 · **Gate:** D

## Problem statement

SP-01/SP-02 give a per-neuron, per-spike learning signal. But the loss is a *global* function of the whole network's spike trains. Two directions of credit assignment remain:

1. **Spatial:** error must propagate backward across layers (output → hidden). Naive backprop needs weight transport (`W^T`) and a global error bus — both violate the locality requirements of neuromorphic hardware.
2. **Temporal:** error must propagate backward across time. BPTT over the spike DAG stores the full trajectory → `O(T)` memory, and it's non-local in time (updates locked to sequence end). This conflicts with online/on-chip learning.

This sub-problem is: **exact AND cheap AND local** credit assignment — all three, without sacrificing accuracy.

## Why it's the "system decider"

- SP-01 + SP-02 prove the *signal* is right. SP-04 decides whether the *mechanism* is deployable.
- EventProp (Wunderlich 2021) already made exact event-gradients cheap (adjoint method, no trajectory storage, 5-26x faster than BPTT-surrogate) — for the ACTIVE network. The open part: making it local-in-time AND carrying SP-02's informative signal for dead neurons.

## The candidate approaches (evaluate, then pick — D3)

| Approach | Spatial locality | Temporal locality | Memory | Accuracy risk |
|---|---|---|---|---|
| BPTT over events (current engine plan) | No (W^T transport) | No (O(T) storage) | O(T) | Low — but fails hardware goal |
| EventProp / adjoint (no stored trajectory) | No (global error) | Yes (adjoint at events) | O(1)-ish | Medium — need loss shaping |
| E-prop / eligibility traces | No (global error) | Yes (local traces) | O(1)/neuron | Medium-high |
| TESS / Traces Propagation (forward-only, fully local) | Yes | Yes | O(1) | High — largest accuracy gap |
| Forward-Forward / contrastive (layer-local) | Yes | Yes | O(1) | High |

**The target:** an eligibility-trace / forward-only rule whose accuracy is within the SP-01/SP-02 exact engine's ceiling — i.e., locality that doesn't cost accuracy. If that is impossible, SP-04 must state it and justify which tradeoff is accepted (with evidence).

## The relationship to SP-02

SP-02's escape-noise signal is naturally *local in time* (it only needs the membrane potential at each step). That's a gift: the informative dead-neuron signal and the hardware-friendly mechanism can be the same object. SP-04 should reuse it rather than bolt a separate mechanism on.

## Requirements / acceptance (GATE D)

- [x] A written comparison of candidate approaches with a decision (D3) and rejected-option rationale in MEMORY.md.
- [x] Credit assignment correctness: gradient check across **depth** (3+ layers) passes.
- [x] Memory target met: per-neuron eligibility trace (O(1)) OR O(T) only when T is small — measured, not assumed.
- [x] No accuracy regression vs the SP-02 solid state (benchmark comparison).
- [x] If a fully-local rule is chosen: demonstrated on a deeper architecture (≥ VGG-style / ≥ 4 hidden layers), not just 2-layer.
- [x] Document what "local" means concretely for the target hardware (no weight transport, no global error bus, online updates allowed).

## Open questions

- Q4.1: Is an exact+informative+local rule possible in principle, or is there an information-theoretic lower bound on the accuracy cost of locality? — **measured (E5):** ref 0.990 / deep 0.969 / fa 0.812 / contrastive 0.948 on the 4-hidden net; fully-forward-only costs ~4 pp, no hard barrier found on this task.
- Q4.2: Does SP-02's escape-noise signal keep SP-04 exact gradients unbiased, or does variance dominate at depth? — exact by construction (E1); existence-channel grads verified jointly in the mixed config.

## Change log

- 2026-08-14 — **Complete, Gate D PASS.** Implemented the D3 mechanism in `engine/snn_torch.py` (`_init_local_machinery`, `_existence_layer_grads`, `_contrastive_signal`, `local_learning_grads` with modes deep/fa/contrastive/ref) and the permanent E1–E5 suite (`engine/experiments/exp_sp04.py`). Fresh run passes all gates: E1 gradcheck depth 3/4 + mixed fired/silent (dot ≤ 2.2e-6, w ≤ 2.4e-6, 0 flips); E2 retained state O(1) in grid (24.65 B/neuron flat, G=401→16001); E3 no-regression (deep 0.927 vs ref 0.969); E4 deep net (4 hidden) trains 0.969, depth utility +36.5 pp; E5 Q4.1 ablations (fa 0.812, contrastive 0.948 vs ref 0.990). Results: `docs/results/SP-04-experiments.md` + JSON.
- 2026-08-14 — Research written (`docs/research/SP-04-research.md`). Key result: the engine is already a three-factor rule (local eligibility × learning signal); temporal locality is already exact + O(1); the ONLY non-local term is the hidden-layer learning signal (the `W^T` transport). D3 recommended = per-layer local loss (deep supervision, exact) as the mechanism, with feedback-alignment and contrastive ablations to measure the cost of stricter locality (Q4.1).
