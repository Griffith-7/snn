# Sub-Problems — Overview and Dependency Map

The main problem (`01-main-problem.md`) decomposes into exactly four sub-problems. This file is the map; each sub-problem has its own living document under `sub-problems/`.

## The four sub-problems

| ID | Name | File | Kind of solution | Solved by | Status |
|----|------|------|------------------|-----------|--------|
| SP-01 | Exact gradient w.r.t. spike time (fired neurons) | `SP-01-exact-spike-time-gradient.md` | Math — IFT / implicit differentiation | Phase 1 | ✅ Gate A PASS (2026-08-13) |
| SP-02 | Spike birth/death — silent neuron credit | `SP-02-spike-birth-death-credit.md` | Statistics — escape noise / expectation | Phase 2 | ✅ Gate B PASS (2026-08-13) |
| SP-03 | The reset jump (discontinuity) | `SP-03-reset-jump-saltation.md` | Math — saltation matrices (hybrid systems) | Phase 3 | N/A — D1: single-spike TTFS (2026-08-13) |
| SP-04 | Temporal + spatial credit assignment | `SP-04-temporal-spatial-credit-assignment.md` | Algorithm architecture + systems | Phase 4 | ✅ Gate D PASS (2026-08-14) |

## Dependency map

```
SP-01 ──> SP-02 ──> SP-04
  \              /
   SP-03 (only if multi-spike)
```

- **SP-01 must be first.** It is the foundation: exact math for fired neurons. Everything else builds on having a correct backward pass.
- **SP-02 is the core research risk.** It addresses the silent/dead neuron problem — the reason exact spike-time gradients (SpikeProp lineage) historically failed. Without it, the engine cannot train anything where neurons start silent (all real tasks).
- **SP-03 depends on an architecture decision (D1):** single-spike TTFS vs multi-spike. Saltation is only needed for multi-spike. **D1 was decided 2026-08-13: single-spike TTFS**, so SP-03 is declared **Not Applicable** and the phase is skipped (GATES.md Gate C, N/A).
- **SP-04 can partially overlap SP-02** (both concern the learning signal's propagation), but per project rule, SP-04 starts only after SP-02 is solid.

**Status (final, 2026-08-16):** SP-01, SP-02, SP-04 all solid (Gates A, B, D PASS); SP-03 N/A
under D1. **Phase 5 / Gate E PASS on CIFAR-10** (2026-08-15/16): exact engine 0.273/0.261/0.250 vs
tuned STBP baseline 0.270/0.264/0.265 (seeds 0–2), equal SynOps, 160× lower latency; SP-02
real-data fix (`lam=[5,50]`) removed init dependence. **CIFAR-10-DVS benchmark done (2026-08-16,
seeds 0–2): accuracy NOT confirmed** — engine 0.230/0.204/0.220 vs tuned baseline 0.214/0.250/0.234
(mean 0.218 vs 0.233), mixed within seed noise, baseline marginally ahead; 160× latency win stands
(`docs/results/SP-05-DVS-experiments.md`). Strictly-worded PRD G0 "and CIFAR-10-DVS" accuracy bar:
**not met** (reported honestly; engine ≈ surrogate on DVS).

## The "solid" definition (gate rule)

A sub-problem is solid when ALL of (PRD §8):

1. Precise written statement (its SP file).
2. Solution derived in math, written down (not just code).
3. Passing verification test (gradient check / toy problem / ablation).
4. Decision + rejected alternatives documented in MEMORY.md.
5. No regression of previously solved sub-problems.

## Cross-cutting concerns

These are not sub-problems (they do not block a phase gate), but every phase must respect them:

- **Engine code must be runnable** — no placeholder kernels (this was the "other AI" skeleton's main flaw).
- **Bias gradient correctness** — bias is a step response, not the `Kv` impulse kernel (suspected bug from the skeleton review).
- **No NaN on silent neurons** — infinite spike times must route to SP-02 logic, never poison gradients.
- **Gradient-check harness** must exist from SP-01 onward and stay green.
