# Exact Event-Based SNN Training Engine

Training Spiking Neural Networks (SNNs) with **exact gradients instead of surrogate gradients** —
solving the non-differentiable spike problem so spike networks train as well as dense float networks,
at a fraction of the energy cost.

**Working principle.** We do NOT claim to "differentiate the vertical cliff." Instead the problem is
decomposed into four sub-problems, each solved with exact math where it exists (spike-time gradients,
saltation matrices) and a *principled, quarantined* mechanism where it doesn't (spike birth/death).
One sub-problem at a time — each made solid and verified before moving on.

| # | Sub-problem | Kind of solution | Status |
|---|-------------|------------------|--------|
| 1 | Exact gradient w.r.t. spike time (active neurons) | Math (implicit differentiation / IFT) | ✅ Gate A PASS |
| 2 | Spike birth/death — silent neuron credit | Statistics (escape-noise / expectation) | ✅ Gate B PASS |
| 3 | The reset jump (discontinuity) | Math (saltation matrices / hybrid systems) | N/A under D1 (single-spike TTFS) |
| 4 | Temporal + spatial credit assignment (across layers & time) | Algorithm architecture + systems | ✅ Gate D PASS |

Main problem solved on CIFAR-10 (Gate E PASS) → **`docs/FINAL-REPORT.md`**.

## Headline results

Reproduction: `python engine/experiments/exp_sp05.py --mode all` (15k/40, seeds 0–2).

| Benchmark | Exact engine | Tuned STBP surrogate | Note |
|---|---|---|---|
| CIFAR-10 | **0.273 / 0.261 / 0.250** | 0.270 / 0.264 / 0.265 | ≈equal accuracy, **160× lower latency** (1 event/neuron vs T=160) |
| CIFAR-10-DVS | 0.230 / 0.204 / 0.220 | 0.214 / 0.250 / 0.234 | **Accuracy NOT confirmed** (mixed within seed noise); latency win decisive |

Honesty rule: failures and non-confirmations are reported as prominently as wins. The DVS accuracy
result is explicitly documented as **NOT confirmed** — same encoding/architecture/loss/init, but the
engine lands within seed noise of (and slightly behind) the surrogate, so the DVS verdict rests on the
latency advantage (`docs/results/SP-05-DVS-experiments.md`).

## Getting started

```bash
pip install -r requirements.txt
python scripts/download_dvs.py   # fetch CIFAR-10-DVS (optional; CIFAR-10 auto-downloads)
```

Quick sanity run (4096 train / 8 epochs):
```bash
python engine/experiments/exp_sp05.py --mode ref --n-train 4096 --epochs 8
```

Full reproductions (each takes ~2 h on one GPU):
```bash
python engine/experiments/exp_sp05.py --mode ref --seed 1 --init norm --lam 5,50
python engine/experiments/exp_sp05_dvs.py --seed 0 --lam 5,50 --slope 6.0
```

Every run is seeded and writes a machine-readable JSON under `docs/results/sp05/`.

## Project structure

```
├── engine/                      <- the exact-gradient engine + experiments
│   ├── snn.py / snn_torch.py    <- spiking models (numpy reference + torch)
│   ├── losses.py / optimizers.py
│   ├── cifar_io.py / cifar_io_dvs.py
│   └── experiments/             <- runnable per-phase experiments
├── scripts/                     <- data download / utilities
├── docs/
│   ├── 01-main-problem.md       <- the big problem, precisely stated
│   ├── 02-sub-problems.md       <- the 4 sub-problems + dependency map
│   ├── FINAL-REPORT.md          <- end-to-end account: results + verdict + reproduce
│   ├── tracking/                <- GATES.md checklist, WORKLOG.md
│   ├── research/                <- SP-02..SP-04 deep dives
│   └── results/                 <- per-phase experiment logs + JSON + evidence
│   └── sub-problems/            <- SP-01..SP-04 definitions of done
├── PRD.md                       <- product requirements (what "done" means)
├── PLAN.md                      <- master plan with phase gates
└── MEMORY.md                    <- running log of decisions, findings, state
```

Datasets are downloaded/generated into `cifar-10-python/` and `data/` (gitignored).

## Documentation

- **What "done" means:** `PRD.md` · **Master plan with gates:** `PLAN.md`
- **Gate checklist:** `docs/tracking/GATES.md`
- **End-to-end account:** `docs/FINAL-REPORT.md`
- **Main problem & sub-problem statements:** `docs/01-main-problem.md`, `docs/02-sub-problems.md`
- **Per-phase experiment results:** `docs/results/`

## License

MIT — see `LICENSE`.
