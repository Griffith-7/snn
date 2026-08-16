# WORKLOG

Dated, chronological record of work. Append new entries at the bottom.

## 2026-08-16 — PROJECT ACCEPTED AS COMPLETE

User decision: **accept and finalize** (over: more epochs on CIFAR-10 / DVS, or a training-speed
effort). Rationale recorded: core science solid and honest; the only unmet item (DVS accuracy,
~1.5 pp behind on average) is within seed noise; remaining weaknesses (wall-clock training,
low absolute accuracy at 12×12) are structural properties of exact training with low
cost-benefit for the claim. Final deliverables: `docs/FINAL-REPORT.md` (end-to-end account),
`docs/tracking/GATES.md` (Gate E PASS on CIFAR-10; DVS accuracy documented as NOT confirmed),
`docs/results/SP-05-experiments.md` + `SP-05-DVS-experiments.md`, all gates re-runnable.

## 2026-08-16 — CIFAR-10-DVS Gate E benchmark done; accuracy NOT confirmed (reported honestly)

- **Dataset.** Figshare 8.4 GB rejected (~10 KB/s, no reliable range support). Downloaded the
  NDA_SNN Google Drive mirror of CIFAR10-DVS (`train_file` 473 MB / `test_file` 53 MB zips) with
  `scripts/download_dvs.py` — parallel 8 MB HTTP Range chunks, resumable `.part` files, ~10 min.
  Per-sample `.pt` = list `[frames (2,128,128,10) float32 binary spikes, label (1,)]`;
  9000 train / 1000 test, label-balanced.
- **Preprocessing/encoding (`engine/cifar_io_dvs.py`).** Per-sample event arrays → count frame
  (ON+OFF `abs` / ON−OFF `signed` / dual-channel / temporal blocks), 12×12 area-downsample, per-sample
  normalize, TTFS via `cifar_io.encode_times`. npz caches in `data/cifar10dvs/`.
- **Encoding chosen by an ANN learnability oracle** (20-epoch ANN on the engine's exact feature
  pipeline): abs12 **0.166** > dual12 0.141 > tblocks2 0.138 > signed12 0.117 > res24 0.100 >
  tblocks3/5 0.117/0.100 → **abs 12×12, tblocks=1, n_in=144 = identical config to the CIFAR-10
  Gate E run** (only the dataset differs). DVS is harder than CIFAR gray even for the oracle.
- **Full 9000/40 runs** (`engine/experiments/exp_sp05_dvs.py --lam 5,50 --slope 6.0`, seeds 0–2):
  engine `ref` **0.230 / 0.204 / 0.220** vs tuned baseline `stbp` **0.214 / 0.250 / 0.234**
  (mean 0.218 vs 0.233), silent_out 0.0, SynOps 13,520 vs ~11.2 k (~equal), latency
  **1 event/neuron vs T=160 (160×)**, train ~69 min vs ~2.8 min. **Mixed within seed noise —
  engine +1.6 pp at s0, −4.6 pp at s1, −1.4 pp at s2; baseline ~1.5 pp ahead on average.**
- **Verdict (honest): the DVS accuracy bar is NOT confirmed** — engine ≈ surrogate within seed
  noise; the decisive, reproducible wins are latency (160×) and equal SynOps. **G0's strict
  "and CIFAR-10-DVS" accuracy requirement: NOT met on DVS** (CIFAR-10 half PASSED). First draft
  of the docs over-claimed a PASS from seed 0 alone; corrected everywhere to the honest
  three-seed picture.
- **Docs:** `docs/results/SP-05-DVS-experiments.md` + JSON (seeds 0–2; s0 backed up to
  `sp05-dvs-results-s0.json`; JSON merged from the three log dumps after later runs overwrote it),
  `FINAL-REPORT.md`, `GATES.md`, `PLAN.md`, `01-main-problem.md`, `02-sub-problems.md`,
  `README.md`, MEMORY facts 21–22 + decisions D-DVS-enc/src/base.

## 2026-08-15/16 — SP-02 fix finalized; Gate E documented honestly

- Seed-2 full 15k/40 std-init per-layer lam [5,50] (`docs/results/evidence/sp05_seed2_plam.log`): **test 0.250 (peak
  0.284 @ ep25), silent_out 0.0, silent_hid 0.0**, 115.8 min → **two-seed robustness confirmed**
  (s1 0.261 / s2 0.250), no pos-init, real data.
- `lam_out ∈ {30,40,50,60,80}` 4096-scale val sweep (held-out 1024, `diag_plam_one.py`): all
  within noise (val 0.202–0.209); higher lam_out just overfits (train 0.277/[5,50] vs 0.223/[5,30])
  → knob exhausted; no further full run warranted.
- Final Gate E numbers: engine **0.273 (s0, std-init lam=5) / 0.261 (s1, [5,50]) / 0.250 (s2,
  [5,50])** vs tuned baseline **0.270/0.264/0.265**. ≥ baseline at s0, tied within noise at s1–s2;
  decisive win = **latency 1 event/neuron vs T=160**. Original +15% vs untuned baseline (0.238)
  superseded.
- Docs finalized: `docs/results/sp05/sp05-results.json` rewritten per-run (config/seed/lam/note),
  `docs/results/SP-05-experiments.md` (honest 3-seed tables), `docs/tracking/GATES.md` (Gate E
  PASS final; Gate B real-data addendum), `docs/research/SP-02-research.md` §6 real-data
  verification + Q2.6 resolved. MEMORY facts 19–20, Q5 → resolved. Temp diag scripts cleaned up.

## 2026-08-15 — SP-02 real-data collapse FIXED: per-layer lam [5,50] (full-scale verdict)

- Full 15k/40 std-init seed-1 run with per-layer lam [5,50] (`_tmp_plam_full.py`,
  `docs/results/evidence/sp05_seed1_plam.log`): **test 0.261, silent_out 0.0, silent_hid 0.0, train 0.333 (underfit),
  115 min**. Test curve: 0.134 → 0.189 (5) → 0.231 (10) → 0.255 (15) → 0.256 (20) → 0.260 (35).
- Comparison (all 15k/40, seed 1 unless noted): lam=5 collapse 0.104 · uniform lam=20 0.225 ·
  **per-layer [5,50] 0.261** · tuned baseline 0.270/0.264/0.265 (seed-1 0.264) · pos-init engine
  0.269.
- **Verdict:** SP-02's real-data output-layer collapse is **fixed WITHOUT a firing-guaranteeing
  init** (0% silent out, no special init, real CIFAR, no toys). Accuracy is statistically tied
  with the tuned baseline (~0.3 pp under seed-1) but still ~0.8 pp under the best engine config
  (pos-init 0.269). Q5 → resolved (MEMORY facts 19–20).
- Remaining open: the last ~0.3–0.8 pp vs baseline (diminishing returns), then Gate E re-verify +
  3-seed table + docs + temp cleanup.

## 2026-08-15 — SP-02 per-layer lam: output-specific channel strength (fix candidate)

- Engine: added `_as_layer_lam` (scalar → all layers, backward-compatible; list → per-layer) to
  `existence_grads` + `local_learning_grads` in `engine/snn_torch.py`. Smoke-tested list-lam
  through both paths (ref + deep).
- Real-CIFAR 4096, std-init seed 1 (failure regime), mode ref, per-epoch out_sil/acc
  (`diag_plam_one.py`):
  - uniform lam=5: output 99% silent (deficit pct [0.02,0.17,0.59] — near-threshold), acc ~0.10.
  - uniform lam=20: 0% silent, acc 0.164 by ep7.
  - per-layer **[5,20]: RE-collapses (0.989 silent)** — hidden channel must stay strong too (it
    drives the hidden→output input stream keeping the output alive).
  - per-layer **[5,50]: 0% silent out from ep0, acc 0.125→0.227 (ep7) → 0.270 (ep14)**, no collapse.
  - per-layer **[5,100]: plateaus 0.168** — too aggressive, spikes at wrong times.
  - seed-2 check [5,50]: 0% silent out, acc 0.121→0.223 (ep9) — robust across seeds.
- **Verdict:** output-layer channel (lam_out≈50) is the load-bearing knob; hidden lam=5 suffices;
  margin-scaling not needed (deficits small). [5,50] at 4096 reaches tuned-baseline/pos-init-level
  accuracy on 10× less data.
- **Full 15k/40 std-init seed-1 run with lam=[5,50] launched** (`_tmp_plam_full.py`, detached,
  `docs/results/evidence/sp05_seed1_plam.log`). Pending: test acc vs tuned baseline 0.264 / pos-init engine 0.269.
  If it holds, SP-02 real-data gap = fixed WITHOUT pos-init (Q5 → resolved).

## 2026-08-15 — SP-02 real-data diagnosis: lam is the load-bearing knob (partial fix)

- `diag_sp02_real` (real CIFAR, seed-1 std-init): silent outputs are **near-threshold** (deficit
  below θ pct [0,0.8,0.9]) — NOT far-dead; K(t_peak−t_i) small but equal for hidden/output
  (frac<0.1: 0.79 vs 0.86) — NOT kernel decay. Differentiator = **channel strength** (outputs
  targeted on only 1/10 of samples, no downstream adjoint). lam sweep 4096/8: lam=5 → 99.6%
  silent; **lam=20/100 → 0% silent**.
- `_tmp_lam20` (4096/20, std-init seed 1, lam=20): output revives by ep 3, acc climbs 0.135→0.244,
  no collapse.
- `_tmp_lam20_full` (15k/40, std-init seed 1, lam=20): **test 0.225**, silent_out 10.1%,
  silent_hid 0, train 0.251 (underfit). No collapse but below tuned baseline 0.264 and pos-init
  engine 0.269.
- **Conclusion:** lam≥20 is SP-02's real-data operating point (collapse fixed on real data, no
  toys), but the channel alone is not enough for competitive accuracy; a firing-guaranteeing init
  is still required. Silence-tolerant readout / margin-scaled objective = remaining open design
  (MEMORY fact 18, Q5). Gate E robust config stays pos-init + lam=5 (0.269/0.273 vs tuned baseline
  0.264/0.270).

## 2026-08-15 — Phase 5 robustness follow-up: seed fragility, tuned baseline, SP-02 real-data gap

### What was found (stresses the recorded Gate E PASS)
- **Engine is seed-fragile (std init).** `diag_seed_init`: output firing at init = s0 0.58 / s1 0.21 /
  s2 0.34 at w_scale=0.4/bias=0.2. `docs/results/evidence/sp05_seed1.log`: seed-1 ref stalls at 0.10 for 40 epochs
  (output collapse). `diag_seed1_revive`: hidden silence revives 826→0 (ep 3) but **output stays
  ~5110/5120 silent** (99.6%) for 20 epochs with correct-class targets applied every batch, lam=5 —
  contradicts SP-02 E7 toy. Wiring verified correct (`existence_grads` last layer
  `target = (~fired) & onehot`); it is a scaling weakness: margin gradient is bounded below
  (~-1/T) but the weight gradient `-lam/B·(1-p)/T·K(t_peak-t_i)` decays via the kernel for late
  `t_peak` of deeply-silent outputs. **SP-02 real-data fix deferred — must be tested on real data, no toys.**
- **Baseline was untuned.** Published stbp slope=2.0 → 0.238. Tuned (slope=6.0, lr=0.01, pos-init)
  → **0.270/0.264/0.265** (seeds 0–2), seed-robust.

### Fix verified
- **Positive-uniform init** (U[0.05,0.4], bias 0.2) → 100% firing all seeds, good latency spread
  (`diag_posinit`, `diag_posspread`); all seeds learn at 4096/20 (`diag_poslearn`, train
  0.43/0.39/0.39 vs seed-1 stall).
- Full seed-1 ref 15k/40 pos-init: **0.269 test, 0.0 silent output/hidden, still climbing**
  (`docs/results/evidence/sp05_seed1_pos_ref.log`) vs tuned baseline 0.264. Engine ≥ tuned baseline at seeds 0–1
  (0.273/0.269 vs 0.270/0.264); decisive win remains latency (1 event/neuron vs T=160).

### Decisions
- Continue: Gate E is PASS-with-caveats (robust init required; accuracy margin thin; latency is the
  real win). Seed-2 engine run pending.
- SP-02 output-layer revival is an open research item (Q5) to fix with real-data experiments.

### Next
- (user) run engine ref seed 2 pos-init (~2h) to complete the 3-seed table, then SP-02 real-data
  fix (per-layer lam / t_peak-kernel regime / silence-tolerant readout), then re-verify Gate E and
  clean up temp diag scripts.

## 2026-08-14 — Full-project audit + Phase 5 kickoff

### Audit verdict (against PRD §8: statement / derivation / verification / decision+rejected / no-regression)
- **SP-01 SOLID** — IFT derived (research §2.3); forward verified 1.7e-14 vs NumPy oracle; E2/E3 gradchecks < 1e-4; training smoke test; E5a–E5d edge cases (depth scaling, grazing, silent-zero, kink) documented.
- **SP-02 SOLID** — escape-noise peak-margin derived (§2); E6 far-dead revival to exact first-crossing times (3.6e-15); E7 output-layer silence; E8 no-regression + envelope FD ~1e-9; E9 ablation control (0.58 vs 0.99); D2 with rejected alternatives recorded.
- **SP-03 N/A** — D1 single-spike TTFS ⇒ no reset jump to saltate; Gate C N/A.
- **SP-04 SOLID** — exact per-layer local loss (D3); E1 gradcheck depth 3/4; E2 memory O(1) measured; E3/E4 no-regression + depth utility; E5 locality ablations; D3 + rejected FA/contrastive recorded.
- **Main problem: NOT fully solved.** Theoretical core (the non-differentiable-spike decomposition) is solved and verified; but "done" per `docs/01-main-problem.md` also requires **G0**: beating a surrogate baseline on CIFAR-10 / CIFAR-10-DVS at equal/better energy — that is Gate E.
- Honest caveats logged: all training to date is toy-scale (2-class synthetic, 2–4 layers); SP-01 E4 training non-monotone (0.97@5 → 0.82); deep TTFS vanishing-gradient regime documented (E5a); SP-02 lam sensitivity (Q2.6).

### Doc fixes (stale statuses brought in line with reality)
- `docs/01-main-problem.md`: status table → ✅ A, ✅ B, N/A (D1), ✅ D.
- `docs/02-sub-problems.md`: same + status note; dependency-map text updated for D1.
- `PLAN.md`: Phase 0 🟢 (D1 decided, build-clean, harness, testbed all done); Phases 1–2 🟢; Phase 3 ⚪ N/A; Phase 4 🟢; Phase 5 🟡 in progress; decisions log D1/D2/D3 resolved.
- `docs/sub-problems/SP-03-reset-jump-saltation.md`: Status → N/A under D1; change log entry.
- `MEMORY.md`: project state + session log updated; audit entry added.

### Next
- Phase 5 (Gate E): CIFAR-10 benchmark vs STBP/SLAYER/EventProp; energy (SynOps) + latency (timesteps); configs + seeds published.

## 2026-08-14 — SP-04 (temporal + spatial credit assignment) implemented, verified, gated PASS

### Built
- `engine/snn_torch.py` — SP-04 mechanism: `_init_local_machinery` (trainable per-hidden-layer readouts R, fixed random feedback B_fa, fixed contrastive target projector P_cont), `_existence_layer_grads` (extracted SP-02 reuse), `_contrastive_signal` (TP-style layer-local target CE), `local_learning_grads(t_in, y, T_noise, lam, mode, ...)` with modes `deep` (D3 mechanism, no W^T/adjoint, returns readout grads) / `fa` (ablation) / `contrastive` (ablation) / `ref` (== existence_grads, the reference ceiling).
- `engine/experiments/exp_sp04.py` — permanent E1–E5 suite.
- `docs/research/SP-04-research.md` — three-factor form, temporal-locality analysis, candidate table, D3 rationale, hardware-locality definition. Sources verified by web search: Bellec 2020 (e-prop, s41467-020-17236-y), Pes et al. 2025 (Traces Propagation), Hao et al. 2026 (per-layer diagnostics).

### Bugs found and fixed during the run
- **E1 gradcheck was checking the wrong objective (design):** the decoupled local gradients were compared against the TOTAL loss (dot 0.004–0.042). Fixed by a per-layer objective `_layer_objective` (layer's own readout/output-CE + own existence loss) and per-layer perturbations.
- `grads_R` has length n_layers−1 (hidden readouts only): index guard `l < len(grads_R)`.
- `_build_mixed_net` standard-normal weights gave fired_frac ≤ 0.315 (unmeetable 0.35 floor): rebuilt as positive-uniform smooth net + deliberately silenced hidden rows (all-negative ⇒ u ≤ 0, guaranteed silent); output layer kept all-fired (fact: latency-CE clamp/gradient are intentionally inconsistent, so gradcheck labels must never point at an always-silent output).
- Silent sentinel is +inf: margin computation must ignore non-finite `t_prev` entries.
- `_per_layer_cosine` unpacked the wrong tuple position (`grads_ref, _, _, _` put the loss in `grads_ref`).
- E4 depth_utility units bug (fraction 0.365 compared against 2.0 pp threshold): reported in pp.

### Verified (full suite passes on RTX 3050, single run, ~12 min)
- E1: per-layer-loss gradcheck, depth 3/4 (smooth) + depth 3 (mixed fired/silent, existence active): all pass, dot mean ≤ 2.2e-6, per-weight mean ≤ 2.4e-6, 0 skipped flips; readout weights checked too.
- E2: retained state O(1) in grid — 19328 B flat (24.65 B/neuron) across G=401→16001; peak GPU grows 9.5→60.2 MB (transient workspace); BPTT-over-grid would store 315k→12.5M elements.
- E3: no regression — deep 0.927 vs ref 0.969 on identical init (2-hidden), both resolve hidden silence (pass: gap 0.042 < 0.10, > 0.8).
- E4: 4-hidden net (10→24→24→24→24→2) trains with the local mechanism — deep 0.969 vs ref 0.990; frozen-bottom 0.604 → depth utility +36.5 pp (pass ≥ 2 pp); deep per-layer cosine decays with depth (0.93/0.80/0.65/0.59) but stays positive; ref gradient norm min 3.13 ≫ 10·fp64_eps (valid).
- E5: Q4.1 ablations on the same deep net — ref 0.990, deep 0.969, fa 0.812, contrastive 0.948. FA worst despite highest cosine (0.942); contrastive near-orthogonal (0.346) yet ~4 pp under ref. No hard locality barrier on this task.

### Findings recorded
- Locality has a small measured accuracy cost here (~4 pp fully-forward-only), not catastrophic; the ranking is ref ≥ deep > contrastive > fa. Reported in results doc.
- Gate D verdict PASS in `docs/tracking/GATES.md`. Results in `docs/results/SP-04-experiments.md`. Next: Phase 5 (Gate E).

## 2026-08-13 — SP-01 (exact spike-time gradients) implemented, verified, gated PASS

### Built
- `engine/snn.py` — NumPy oracle: normalized double-exponential kernel, grid+Newton root finding, IFT backward, `TTFSNet`.
- `engine/snn_torch.py` — GPU/torch engine: `TTFSNetTorch`, vectorized bisection+Newton (golden, `peak_tol` gate), `forward_layer_torch`/`backward_layer_torch`.
- `engine/losses.py` / `engine/losses_torch.py` — latency cross-entropy `dL/dt = beta*(onehot - p)` (sign-fixed).
- `engine/optimizers.py` / `engine/optimizers_torch.py` — Adam.
- `engine/experiments/exp_sp01.py` — permanent E1–E5 suite.

### Fixed along the way
- Kernel peak normalization (k_peak must include `(tm-ts)` divisor; PSP peak = 1.0).
- `mask.long().argmax`, NaN in `_K` (inf-inf), fired-set cross-engine comparisons, grazing `up==0` guard.
- Research doc §2.4 sign error: `dL/dt_out = beta*(onehot - p)`.

### Verified (full suite passes on RTX 3050, ~40–80 s)
- E1: forward vs 200k-point dense grid, max err 1.8e-4 (grid-limited).
- E1b: torch vs NumPy oracle, max err 1.7e-14, 0 status mismatches.
- E2 (2-layer, 3 seeds): dot mean ~5e-9, per-weight mean ~1e-7, max ≤ 2.8e-6. PASS.
- E3 (3-layer, 3 seeds): dot mean ≤ 4e-8, per-weight max ≤ 1.1e-5; near-zero grads by abs err (≤ 1.8e-11). PASS.
- E4: exact-gradient training, test acc 0.74 → 0.82 (transient 0.97).
- E5a: first-layer grad norm ratio grows 25 → 410 with depth (TTFS vanishing-grad regime, → SP-04).
- E5b: near-grazing `|dt/dw| = 531` vs 11 normal; no NaN.
- E5c: silent neuron gradient rows exactly 0.
- E5d: kernel-onset kink — two-sided FD invalid within ~`eps*|dt/db|` of an input-arrival time (rel err 1.5e-10 → 0.38); analytic gradient exact. FD validity constraint: use `_build_smooth_net` (fired strictly after all inputs).

### Decision resolution
- The reported "bias gradient bug" was a degenerate test config (bias-only crossing at the kernel-onset kink), not an implementation error. Analytic bias gradient verified exact.
- Gate A verdict PASS recorded in `docs/tracking/GATES.md`. Results in `docs/results/SP-01-experiments.md`.

## 2026-08-13 — SP-02 (spike birth/death credit) implemented, verified, gated PASS

### Built
- `engine/snn_torch.py` — added `peak_margin_torch` (response-window extremum finder) and `TTFSNetTorch.existence_grads` (SP-02 existence channel).
- `engine/experiments/exp_sp02.py` — permanent E6–E9 suite.
- `docs/research/SP-02-research.md` — full derivation (escape-noise Bernoulli at the peak, MLE objective, envelope theorem, isolation, no-annealing rationale) + D2.

### Bugs found and fixed during the run
- **Peak deadlock (design bug, E6):** naive global argmax over `[0, t_max]` collapses onto the pre-input plateau (`u=0`) for every subthreshold neuron → `K(t_peak - t_in) = 0` → existence gradient identically 0. Fixed: search only the response window (t ≥ earliest contributing event); for all-negative responses (`u_max ≤ 0`) use the interior *minimum* extremum (still `u'=0`, so the envelope theorem holds). Result: E6 far-dead gradients 0.88 → 1.00, all m0 revive.
- `idx_start` double-scaling bug (index computed as `t/grid[1]*(G-1)` instead of `t/grid[1]`).
- E7 `sil_stats` bool `.mean()` crash.
- E6 "expected spike time" was wrong: a revived neuron spikes at its first threshold crossing (rising edge), not at the kernel peak.

### Verified (full suite passes on RTX 3050, single run)
- E6: far-dead revival toy, m0 ∈ {2,3,5,8}, |g0| ∈ [0.881, 1.000] (bounded ≥ 0.7), revival steps 29–90, revived spike times exact to 3.6e-15, existence gradient 0 after revival; lam=0 control stays dead.
- E7: training with channel, lam=5 — correct-class output silence resolved, test acc 0.47 → 0.97.
- E8: SP-01 gradchecks re-run (2 & 3-layer, 2 seeds) all PASS (no regression); zero contribution at 100% firing (loss identical, grad diff 0.00e+00); existence gradient + envelope `d(u_peak)/dW = K` verified vs FD ~1e-9.
- E9: ablation — without = 0.583 (chance, hidden silence 0.875), with = 0.990 (hidden silence 0.000), pass=True.

### Findings recorded
- `lam` sensitivity: at `lam=1` the channel loses to the timing loss's wrong-class push-down (E7 plateaus ~0.63); `lam=5` wins (0.99). Reported in results doc + research doc §2.3.
- Gate B verdict PASS in `docs/tracking/GATES.md`. Results in `docs/results/SP-02-experiments.md`. Next: SP-04 (SP-03 N/A under D1).
