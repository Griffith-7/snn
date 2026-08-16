# SP-05-DVS Experiment Results — Full Engine vs Surrogate Baseline on CIFAR10-DVS

**Date:** 2026-08-16
**Device:** NVIDIA GeForce RTX 3050 Laptop GPU
**Raw JSON:** `docs/results/sp05/sp05-dvs-results.json` (per-run config + history, seeds 0–2)
**Reproduce:** `python engine/experiments/exp_sp05_dvs.py --seed 0 --lam 5,50 --slope 6.0`
**Smoke:** `python engine/experiments/exp_sp05_dvs.py --n-train 256 --epochs 2 --mode ref`

## Purpose

Close the strictly-worded second half of the main-problem definition of done
(`docs/01-main-problem.md`): the exact engine must beat a surrogate-gradient baseline
honestly on **CIFAR-10-DVS** as well as CIFAR-10 (PRD G0). This is the second
Gate-E benchmark.

## Data and encoding (apples-to-apples)

- **Source:** CIFAR10-DVS (Li et al. 2017), preprocessed per-sample binary spike
  frames `(2, 128, 128, 10)` = (ON/OFF polarity, H, W, 10 temporal frames), via the
  NDA_SNN Google Drive mirror. 9,000 train / 1,000 test, class-balanced.
- **Encoding choice (measured, not assumed):** the engine is single-spike TTFS
  (1 input spike per neuron), so each event stream is reduced to a grayscale-like
  intensity frame, then the exact CIFAR-10 pipeline applies (12x12 mean-downsample,
  per-sample normalize, TTFS `t = 0.5 + 7.5·(1−x)`). An ANN learnability oracle
  picked the encoding:

  | encoding | n_in | oracle test acc |
  |---|---|---|
  | ON+OFF counts, 12x12 (`abs`) | 144 | **0.166** |
  | ON−OFF counts, 12x12 (`signed`) | 144 | 0.117 |
  | 2 temporal blocks, 12x12 | 288 | 0.138 |
  | 3 temporal blocks, 12x12 | 432 | 0.117 |
  | 5 temporal blocks, 12x12 | 720 | 0.100 |
  | `abs` 16x16 / 24x24 | 256 / 576 | 0.114–0.100 |

  **Single integrated ON+OFF frame at 12x12 is best** — same `n_in = 144` as the
  CIFAR-10 Gate-E run, so the engine/baseline config is *identical* to
  `docs/results/SP-05-experiments.md`; only the source dataset differs.
- **Architecture / loss / optimizer:** 144 → 64 → 10, double-exponential PSP,
  latency-CE `beta=3.0`, engine `AdamTorch(lr=0.02, clip=5.0)`, baseline
  `torch.optim.Adam(lr=0.01)`, T=160. Engine init `positive_init`, SP-02 existence
  channel `lam=[5,50]` (the robust real-data config). Baseline tuned `slope=6.0`.
- **Budget:** 9,000 train / 1,000 test, 40 epochs, B=128.

## Main result — full 9000/40, seeds 0–2

| seed | mode | test acc (final / best) | SynOps/test | latency | wall-clock train |
|---|---|---|---|---|---|
| 0 | **ref (exact engine)** | **0.230 / 0.239** | 13,520 | **1 event/neuron** | 68.9 min |
| 0 | stbp (surrogate, tuned slope=6.0) | 0.214 / 0.230 | 11,395 | T=160 timesteps | 2.8 min |
| 1 | **ref (exact engine)** | 0.204 / 0.221 | 13,520 | **1 event/neuron** | 68.8 min |
| 1 | stbp (surrogate, tuned slope=6.0) | **0.250 / 0.250** | 11,171 | T=160 timesteps | 2.8 min |
| 2 | **ref (exact engine)** | 0.220 / 0.236 | 13,520 | **1 event/neuron** | 68.8 min |
| 2 | stbp (surrogate, tuned slope=6.0) | 0.234 / 0.235 | 11,159 | T=160 timesteps | 2.9 min |
| mean | ref | **0.218 / 0.232** | 13,520 | **1 event/neuron** | ~69 min |
| mean | stbp | 0.233 / 0.238 | ~11.2 k | T=160 | ~2.8 min |

### Verdict (honest)

- **The engine does NOT beat the tuned surrogate on CIFAR-10-DVS accuracy.** The
  comparison is **mixed across seeds**: engine +1.6 pp at seed 0 (0.230 vs 0.214)
  but −4.6 pp at seed 1 (0.204 vs 0.250) and −1.4 pp at seed 2 (0.220 vs 0.234);
  the baseline is **~1.5 pp ahead on average** (final) and ~0.6 pp ahead on
  best-of-run. Within DVS-scale seed noise, the fair statement is **engine ≈
  surrogate on accuracy, baseline marginally ahead**.
- **What the engine does win decisively, on every seed and at equal compute:**
  **latency — 1 event/neuron (TTFS) vs T=160 timesteps (160×)** at ~equal SynOps
  (13.5 k vs ~11.2 k). No approximation error in the exact gradient.
- Engine `ref` is 0% silent output at all seeds (SP-02 `lam=[5,50]` carried over
  unchanged, no tuning). Engine train accuracy keeps climbing at all seeds
  (0.465 / 0.470 / 0.439 @ ep 39) — the exact timing gradient has not saturated,
  but the DVS test signal is noisy and the engine's learning transfers less
  cleanly to test than the surrogate's.

**This DVS result does NOT pass Gate E's accuracy bar.** The CIFAR-10 Gate E PASS
(engine ≥ tuned baseline at seed 0, tied at seeds 1–2) stands; the strictly-worded
PRD G0 "beats a surrogate on CIFAR-10 **and** CIFAR-10-DVS" is therefore met on
CIFAR-10 only. DVS accuracy is honest **inconclusive-to-slightly-negative** for
the engine, with the latency win intact. (Reported rather than smoothed over —
the project's honesty rule.)

## Honest caveats

- **Single-frame reduction discards the temporal structure** of the event stream
  (the engine's TTFS readout spikes once per neuron). The absolute accuracy
  (0.20–0.25) reflects that and the hard DVS signal; it is not competitive with
  frame-sequence SOTA (~80%+). The claim is only the apples-to-apples one.
- Absolute accuracy is low because DVS 12x12 frames are noisy and the source
  classes are hard even for the ANN oracle (0.166).
- Three seeds (0–2). The engine's DVS test curves are noisy (range 0.20–0.24); the
  robust claims — engine ≈ surrogate on DVS accuracy, decisive 160× latency win,
  0% silent, equal SynOps — hold across all three seeds.
- The engine is ~25× slower per batch in wall-clock on DVS than the surrogate
  (exact IFT scan + adjoint), reported for transparency (same story as CIFAR-10).

## Summary vs Gate E (DVS half)

| Requirement | Result |
|---|---|
| Benchmark vs STBP on CIFAR-10-DVS (apples-to-apples) | done — 9000/40, seeds 0–2, same encoding/arch/loss/init; only the rule differs |
| Engine accuracy ≥ surrogate | **NOT confirmed** — engine 0.230/0.204/0.220 vs baseline 0.214/0.250/0.234 (mean 0.218 vs 0.233); mixed, baseline marginally ahead within seed noise |
| Energy (SynOps) + latency | SynOps ~13.5 k vs ~11.2 k (~equal); latency **1 event/neuron** vs **T=160** |
| Robustness config carried over | done — `lam=[5,50]`, 0% silent out, no tuning |
| Verdict: engine ≥ surrogate at equal/better energy | **NOT a PASS on accuracy** — engine ≈ surrogate (baseline marginally ahead), decisive latency win stands |

**Gate E (DVS half): accuracy NOT PASSED; latency PASSED.** Net effect on the main
problem: solved at the core and on CIFAR-10 (Gate E PASS); the strictly-worded
"and CIFAR-10-DVS" accuracy requirement is not met — the honest finding is that
the exact engine matches a tuned surrogate on DVS within seed noise, at 160×
lower latency.
