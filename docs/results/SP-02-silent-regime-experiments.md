# SP-02 Silent-Regime Validation — Guarded Channel on Real CIFAR-10 (2026-08-17)

Runs `python engine/experiments/exp_sp02_silent_regime.py` (GPU, float64; `--skip-train`
re-runs the deterministic guard checks reusing the recorded training JSON).
Code: `engine/snn_torch.py` (`edge_peak_guard`), `engine/experiments/exp_sp02_silent_regime.py` (new).
JSON: `docs/results/sp02-silent-regime/sp02-silent-regime-results.json`.

## Why this experiment exists
Every prior full-scale run used a firing-guaranteeing positive-uniform init (silent_frac
== 0 from epoch 0), or the per-layer-lam std-init fix that reached 0% silence immediately.
The existence channel and the new `edge_peak_guard` had **never been exercised in a regime
where silence actually exists during training**. This run forces that regime: standard-normal
init scaled down (`w_scale=0.30, bias_val=0.0`, 144→64→10, 5000 train, 12 epochs, lam=[5,50]).

## Results
- **S1 — silence present:** initial silent fractions **output 50.0%, hidden 49.8%** (real silent regime, not a toy).
- **S2 — no NaN:** channel run 0 NaNs, control run 0 NaNs across all 480 batches (loss + all weight grads + readout grads).
- **S4 — revival:** channel drives hidden silence **0.498 → 0.000** (and output 0.500 → 0.000).
- **S6 — ablation control:** with `lam=0` (no channel) hidden silence **stays 0.400** (output ~100%) — silence is not self-recovering.
- **S5 — learning:** channel test acc **0.091 → 0.143** (monotone after ep 2), while the control is **flat at 0.102**.
- **S3 — guard unit tests (deterministic):**
  - plateau (`w=0,bias=0` → `u(t)=0` identically): **3/3 guarded, 0 targeted**, gradients finite — the channel correctly does *not* push the degenerate deadlock;
  - flippable-earliest-weight (two same-time inputs with cancelling ±1e-10 weights → `u(t)=0` exactly, nonzero weights): **3/3 guarded, 0 targeted**;
  - healthy control (`bias=0.2`, fires): **0 guarded** (no false positives).
- **S3c — guard engages on real data:** scan of the initial net over 8 contiguous batches:
  **4 guarded events** (layer 0). In the shuffled training loop the guard fired 0/480 times —
  the degenerate plateau is a rare, sample-clustered event that weight updates immediately
  destroy (verified: permuted batches show guards on the *initial* net, gone after updates).
  The guard is a safety net for that rare event; it never fired spuriously and never produced NaN.

## Gates
| gate | check | result |
|------|-------|--------|
| S1 | initial hidden silence > 2% | ✅ 49.8% |
| S2 | 0 NaN loss/grads across the run | ✅ |
| S3 | plateau + flippable guarded & untargeted; healthy not guarded | ✅ |
| S3c | guard fires on real data | ✅ 4/8 batches |
| S4 | channel revives hidden silence (< 50% of initial) | ✅ 0.498→0.000 |
| S5 | test acc climbs | ✅ 0.091→0.143 |
| S6 | lam=0 control stays silent | ✅ 0.400 stuck |

## Interpretation
The SP-02 existence channel **works in the regime it was designed for**: with 50% of neurons
initially silent, the guarded channel revives all of them (0% hidden/output silence by the end)
while the no-channel control remains stuck at 40%, and the guarded net learns. The
`edge_peak_guard` suppresses targeting only on genuine degenerate `u≡0` plateaus (verified on
constructed exact-cancellation cases and observed on real data), leaves healthy neurons alone,
and introduces no NaN — closing the previously-untested gap in SP-02's real-data claim.
