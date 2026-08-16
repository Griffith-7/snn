# Evidence Logs

Raw training/eval console output backing the SP-05 / Phase-5 (Gate E) results. Kept for
authenticity — anyone can re-run the published commands and diff against these logs.

| File | What it records |
|---|---|
| `sp05_seed1.log` | Seed-1 engine stall under standard-normal init (the bug that Gate E robustness fixed) |
| `sp05_seed1_pos_ref.log` | Seed-1 full 15k/40 engine run with positive-uniform init → **0.269** |
| `sp05_seed1_plam.log` | Seed-1 full 15k/40 engine, std-init, per-layer lam `[5,50]` → **0.261** |
| `sp05_seed2_plam.log` | Seed-2 full 15k/40 engine, std-init, per-layer lam `[5,50]` → **0.250** |
| `sp05_seed1_lam20.log` | Seed-1 full run, uniform lam=20 (shows lam=20 alone is not enough) |
| `sp05_dvs_s0.log` | CIFAR-10-DVS seed-0 run (engine 0.230 / baseline 0.214) |
| `sp05_dvs_s1.log` | CIFAR-10-DVS seed-1 run (engine 0.204 / baseline 0.250) |
| `sp05_dvs_s2.log` | CIFAR-10-DVS seed-2 run (engine 0.220 / baseline 0.234) |

Machine-readable per-run results (config + full history): `../sp05/sp05-results.json`,
`../sp05/sp05-dvs-results.json`.
