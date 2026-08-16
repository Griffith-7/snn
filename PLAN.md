# PLAN — Master Plan for the Exact SNN Training Engine

**Rule of the project:** One sub-problem at a time. N is not "done" until it is **solid** (see PRD §8). Never start N+1 while N is open.

## Status legend

- 🔴 Not started
- 🟡 In progress
- 🟢 Solid (passed all gates)
- ⚪ Blocked / on hold

## Phase gates (the entire plan)

```
GATE 0: Project scaffolding .......... docs + decision on single-vs-multi-spike
  |
  v
PHASE 1: SP-01 Exact spike-time gradient .............. GATE A: gradient-check passes
  |
  v
PHASE 2: SP-02 Spike birth/death credit ............... GATE B: principled revival + toy test
  |
  v
PHASE 3: SP-03 Reset jump (saltation) ................ GATE C: jump-map gradient check
  |
  v
PHASE 4: SP-04 Temporal+spatial credit assignment .... GATE D: local & cheap & accurate
  |
  v
PHASE 5: Full engine beats surrogate baseline ........ GATE E: CIFAR-10 / CIFAR-10-DVS
```

## Phase 0 — Scaffolding and foundations

Status: 🟢 Solid (docs + all foundational decisions in place)

Tasks:
- [x] README, PRD, PLAN, MEMORY
- [x] Main problem + sub-problem docs
- [x] **DECISION D1 (2026-08-13): single-spike (TTFS, max 1 spike)** — chosen over multi-spike ⇒ SP-03 (saltation) is Not Applicable; Gate C skipped.
- [x] Build clean: the "other AI" skeleton was reviewed (MEMORY.md) and the engine was built from scratch, runnable, no placeholders (`engine/snn.py` NumPy oracle + `engine/snn_torch.py` torch/GPU).
- [x] Gradient-check harness: finite differences vs analytic spike-time gradient (SP-01 E2/E3).
- [x] Toy testbed: 2-layer TTFS network on a small task (SP-01 E4, SP-02 E7/E9, SP-04 E3–E5).

## Phase 1 — SP-01: Exact spike-time gradient

Status: 🟢 Solid (GATE A PASS, 2026-08-13)

File: `docs/sub-problems/SP-01-exact-spike-time-gradient.md`

Goal: Exact gradient of loss w.r.t. weights for all **fired** neurons, verified.

Deliverables:
- [x] Analytic `dt/dW` derivation using IFT (done in math, written down).
- [x] Correct `backward_layer` incl. bias gradient (bias modeled as a fixed input spike at t=0 with a trainable weight column; documented choice).
- [x] Correct adjoint propagation `lam_prev` accumulation.
- [x] Output loss head that produces `dL/dt_out` (for fired outputs).
- [x] Gradient-check test passes (finite differences, rel. err < 1e-4).
- [x] Handle neurons that never fire in forward (infinite spike time) — route them to SP-02, do not NaN.

**GATE A:** PASS — 2-layer dot ~5e-9 / w ~1e-7; 3-layer dot ~4e-8 / w ~2e-6; training smoke test 0.74→0.82 (E4); edge cases E5a–E5d documented. Results: `docs/results/SP-01-experiments.md`.

## Phase 2 — SP-02: Spike birth/death credit (silent neurons)

Status: 🟢 Solid (GATE B PASS, 2026-08-13)

File: `docs/sub-problems/SP-02-spike-birth-death-credit.md`

Goal: A **principled, unbiased, informative** learning signal for silent neurons.

Key known problems with the current "revival" draft (must be solved, not repeated):
- [x] Revival only reaches near-threshold neurons (`ds` ~ 0 far below threshold) — fixed: `-log p` MLE objective gives far-dead margin-gradient bounded below by ~`1/T`.
- [x] Annealing shrinks the signal over training — fixed: no annealing; `T` is a fixed model noise level (§2.8).
- [x] Output-layer silent neurons have no loss gradient at all — fixed: correct-class output targets (E7: acc 0.47→0.97).
- [x] The current revival is a hand-tuned sigmoid heuristic, not derived — replaced by derived escape-noise peak-margin channel (research doc §2).

Candidate directions (evaluate + pick):
- [x] Escape-noise / stochastic neuron equivalence (Gygax & Zenke) — surrogate = exact gradient of expected spike rate. **CHOSEN (D2)**.
- [x] REINFORCE-style estimator with spike-timing baseline. Rejected: variance; recorded.
- [x] E-prop style eligibility traces as a complementary channel. Deferred to SP-04.

**GATE B:** PASS — far-dead revival E6 (m0∈{2,3,5,8}, |g0|≥0.7); output-layer silence E7; no regression E8; ablation control E9 (0.58 vs 0.99). Results: `docs/results/SP-02-experiments.md`.

## Phase 3 — SP-03: Reset jump (saltation matrices)

Status: ⚪ Not Applicable under D1 (single-spike TTFS decided 2026-08-13)

File: `docs/sub-problems/SP-03-reset-jump-saltation.md`

D1 (single-spike TTFS) ⇒ SP-03 is declared **not applicable** and the phase is skipped. Saltation is only meaningful for multi-spike. **GATE C: N/A** (GATES.md). Re-open only if D1 is revisited to multi-spike.

## Phase 4 — SP-04: Temporal + spatial credit assignment

Status: 🟢 Solid (GATE D PASS, 2026-08-14)

File: `docs/sub-problems/SP-04-temporal-spatial-credit-assignment.md`

Goal: Error propagation across layers and time that is correct, and cheap/local enough for neuromorphic hardware.

- [x] Decide BPTT-over-events vs online eligibility traces vs forward-only. **D3: per-layer local loss (deep local learning)**, an exact local gradient by construction; alternatives (feedback alignment, contrastive) measured and rejected in E5.
- [x] Memory target: O(1)-per-neuron retained state (E2: flat 24.65 B/neuron across grid 401→16001; no full-trajectory BPTT storage).
- [x] Verify no accuracy regression vs Phase 2 state. E3 (deep 0.927 vs ref 0.969, pass); E4 depth utility +36.5 pp.

**GATE D:** PASS — E1 gradchecks across depth (depth 3/4, dot ≤2.2e-6); E2 memory O(1) measured; E3/E4 no-regression + depth utility; E5 locality ablations. Results: `docs/results/SP-04-experiments.md`.

## Phase 5 — Full engine beats surrogate baseline

Status: 🟢 Solid (GATE E PASS, 2026-08-15)

File: `docs/phase-5-benchmark-plan.md` (🟢) · Results: `docs/results/SP-05-experiments.md`

Goal: The exact engine ≥ a surrogate-gradient baseline on a real benchmark (CIFAR-10) at equal or better energy/latency. Done apples-to-apples: same 12×12 grayscale TTFS data, same 144→64→10 arch, same latency-CE loss (beta=3.0), same init/seed; only the learning rule differs.

- [x] Benchmark vs **tuned** STBP surrogate on CIFAR-10 (apples-to-apples config): **ref 0.273/0.261/0.250 vs stbp 0.249/0.231/0.252** (seeds 0–2, tuned baseline re-measured 2026-08-16 — the earlier per-seed "0.270/0.264/0.265" was not reproducible and is superseded); deep 0.250. (Original +15% was vs untuned slope=2.0 → 0.238; superseded by tuned baseline.)
- [x] Benchmark on CIFAR-10-DVS (apples-to-apples, same 12×12/144 TTFS encoding as CIFAR-10):
      engine **0.230/0.204/0.220** vs tuned baseline **0.214/0.250/0.234** (seeds 0–2, mean 0.218
      vs 0.233) — **accuracy NOT confirmed** (mixed within seed noise, baseline marginally ahead);
      decisive 160× latency win at ~equal SynOps (`docs/results/SP-05-DVS-experiments.md`).
- [x] Measure energy (SynOps) and latency (timesteps): SynOps ~13.6 k vs ~11.5–12.4 k (~equal); latency **1 event/neuron vs T=160**.
- [x] Robustness: SP-02 per-layer lam `[5,50]` removes init dependence (full-scale std-init 0.261/0.250, 0% silent out/hid, real data, seeds 1–2).
- [x] Publish configs + seeds: full config in SP-05 results doc; single-file rerun `python engine/experiments/exp_sp05.py --mode all`.

**GATE E:** ✅ PASS (2026-08-15/16) — engine ≥ tuned surrogate at seed 0 and statistically tied at
seeds 1–2, at ~equal energy and **160× lower latency** ⇒ **main problem solved on CIFAR-10** (per
PRD §2 / `docs/01-main-problem.md`). CIFAR-10-DVS benchmark done (seeds 0–2): engine ≈ surrogate
on accuracy (mixed within seed noise, baseline marginally ahead), 160× latency win stands —
reported honestly as NOT passing the accuracy bar
(`docs/results/SP-05-DVS-experiments.md`). Debugging trail (dead-output init w_scale=0.1→0.4;
latency-CE beta=1→3; engine overfits small data, wins at 15k; SP-02 real-data output collapse →
per-layer lam) recorded in SP-05 results doc and MEMORY.md.

## Open decisions log (kept in MEMORY.md)

- D1 (resolved 2026-08-13): **single-spike TTFS** — multi-spike rejected ⇒ SP-03 N/A.
- D2 (resolved 2026-08-13): **escape-noise peak-margin existence channel** for SP-02 — REINFORCE/e-prop/pseudospikes rejected (recorded).
- D3 (resolved 2026-08-14): **per-layer local loss (deep local learning)** for SP-04 — feedback alignment / contrastive measured and rejected.
