# Results — Index

All experiment results, per sub-problem. Every experiment has: hypothesis, setup (seeds/config), result (numbers), and verdict. No results written before they are run.

| Sub-problem | Results file | Experiments count | Gate status |
|---|---|---|---|
| SP-01 Exact spike-time gradient | `SP-01-experiments.md` | (see file) | ✅ Gate A PASS |
| SP-02 Spike birth/death credit | `SP-02-experiments.md` | (see file) | ✅ Gate B PASS |
| SP-03 Reset jump / saltation | (N/A under D1) | 0 | N/A |
| SP-04 Temporal+spatial credit assignment | `SP-04-experiments.md` | (see file) | ✅ Gate D PASS |
| SP-05 Phase 5 — main problem on real benchmarks | `SP-05-experiments.md` | 5 | ✅ Gate E PASS (CIFAR-10) |
| SP-05 DVS benchmark | `SP-05-DVS-experiments.md` | 6 | ⚠️ accuracy NOT confirmed; latency decisive |
| Raw evidence logs | `evidence/` | — | tracked console output for the above |

Machine-readable per-run results: `sp01/`, `sp02/`, `sp04/`, `sp05/` (incl. `sp05-results.json`,
`sp05-dvs-results.json`).

## Rules

- Each experiment MUST be reproducible: fixed seed, exact parameters recorded.
- A result that fails is still recorded (failure data is data).
- Verdicts: PASS / FAIL / INCONCLUSIVE — with the criterion it was judged against.
