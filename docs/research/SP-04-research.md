# SP-04 Research — Temporal + Spatial Credit Assignment

Status: ✅ Written · Date: 2026-08-14 · Sources: verified papers (below)

## 1. The precise target

SP-01 + SP-02 give an exact per-neuron, per-spike learning signal. SP-04 asks whether that
signal can be delivered to every synapse with **exact AND cheap AND local** credit — local in
time (no O(T) storage, no backprop-in-time) and local in space (no weight transport `W^T`,
no global error bus) — without giving up accuracy.

The literature (Section 2) has no single rule that delivers all four. The reason becomes
crisp once we write our own engine's gradient in three-factor form (Section 3): **temporal
locality is already exact and O(1) in this engine; the ONLY non-local term is the
hidden-layer learning signal, which is the `W^T` transport of the output error.** SP-04
therefore reduces to: what replaces that one term?

## 2. Literature findings (verified)

### 2.1 e-prop (Bellec et al. 2020, Nature Comm s41467-020-17236-y)

- Gradient factorizes as `dE/dW_ji = Σ_t L_j^t · e_ji^t` — a three-factor rule: per-neuron
  learning signal × per-synapse eligibility trace.
- Key theorem: `dW_ji|_local` "is not an approximation... collects the maximal amount of
  information about the network gradient dE/dW_ji that can be computed locally in a forward
  manner."
- Eligibility trace `e_ji^t` is a forward recursion over time (O(1) memory); the learning
  signal `L_j^t` is NOT local — symmetric e-prop uses `W^T`, random e-prop uses fixed
  random `B` (broadcast alignment).
- Learns slower than BPTT but tends to approximate its performance. Complexity O(S)
  (O(1) per synapse).
- Implication: e-prop solves TEMPORAL locality via traces; its spatial half is either
  weight transport or random feedback. Our TTFS engine has no temporal problem at all
  (Section 3); what remains is e-prop's spatial half.

### 2.2 Feedback alignment and its evaluation failure modes (Lillicrap 2016; Hao et al. 2026, arXiv:2606.21126)

- FA replaces `W^T` with a fixed random `B`; DFA projects the OUTPUT error directly to
  each layer (`a_l = B_l^T e`) — no forward state at layer l enters the signal.
- Diagnostic paper (2606.21126): the standard reporting pair (accuracy + aggregate cosine)
  is insufficient. Two silent failure modes: (Mode 1) measurement degeneracy — the BP
  reference gradient collapses to the numerical floor in terminal-LayerNorm residual nets,
  making cosine uninterpretable; (Mode 2) aggregation collapse — aggregate cosine masks
  per-layer heterogeneity that concentrates credit at one end of the network.
- DFA's deep-layer credit is a fixed random projection of the output error, independent of
  the layer's forward state → deep layers' cosine → 0 in expectation. FA degrades by one
  random-matrix product per layer.
- Recommended protocol: per-layer cosine + scale stability + reference validity + depth
  utility (accuracy vs a frozen-deep-layers baseline).
- Implication: FA/DFA is exactly the candidate that removes `W^T` but keeps a global error
  bus. Its diagnostics become our SP-04 measurement protocol — the gate requires depth 3+
  and ≥4 hidden layers, precisely where the paper shows naive aggregate reporting fails.

### 2.3 Traces Propagation (Pes et al. 2025, arXiv:2509.13053)

- Forward-only, fully local in space AND time: eligibility traces (temporal) + layer-wise
  contrastive loss (spatial); no auxiliary layer-wise matrices.
- Each layer clusters each sample's trace toward same-class target traces and away from
  other classes — class structure enters every layer forward-only, WITHOUT a backward error.
- Beats all fully-local rules on NMNIST/SHD (1.4 pp vs BPTT); competitive on DVS-GESTURE
  (0.33 pp gap); 3.38 pp gap on DVS-CIFAR10; scales to VGG-9.
- Costs: not online per-sample (needs batch ≥ 2); O(B²LH) pairwise-similarity compute.
- Implication: the strictest locality (no label-broadcast backward, no weight transport) is
  achievable and near-BPTT on simple datasets, but the gap widens on the harder benchmark
  (CIFAR-class — our final target) and it is NOT exact (contrastive surrogate).

### 2.4 Deep supervision (Lee et al. 2015, "Deeply-Supervised Nets")

- Auxiliary classifiers at hidden layers; each layer's weights trained against its own local
  classification loss. Improves training speed and generalization; classic, cheap result.
- Implication: per-layer local loss is not only a locality trick — it is a known
  training/regularization aid. It is also the only candidate that preserves EXACT gradients
  (each local loss is a real differentiable objective).

## 3. The engine already has the three-factor form (the analysis that shapes D3)

Write the exact SP-01 backward in e-prop's notation. Layer `l`: pre-spike times `t_l`,
post-spike times `t_{l+1}` (`inf` = silent), weights `W_l` (`n_{l+1} × (n_l + 1)`),
membrane `u_j(t) = Σ_i w_ji K(t − t_i) + w_j,bias K(t − t_bias)`. For fired `j`, the IFT
(spike time `t_{l+1,j}` is the first root of `u_j = theta`):

```
d(t_{l+1,j})/dw_ji = −K(t_{l+1,j} − t_i) / u'_j(t_{l+1,j})        (u'_j ≠ 0 at crossing)
```

Given `lam_j = dL/dt_{l+1,j}` (the exact adjoint from `backward_layer_torch`):

```
dW_ji = Σ_batch L_j · e_ji
  eligibility  e_ji = K(t_{l+1,j} − t_i)          (LOCAL: the two spike times at the synapse)
  learning signal L_j = −lam_j / u'_j(t_{l+1,j})
```

This is e-prop's exact three-factor form (per-synapse eligibility × per-neuron learning
signal). The eligibility is computed from pre- and post-spike times only — no global
communication, O(1) memory. Everything non-local lives in `L_j`:

- **Output layer:** `lam_j = β(onehot_j − p_j)/B` — computed at the readout, where the
  label is present. Local.
- **Hidden layers:** `lam_i = Σ_j w_ji · (lam_j/u'_j) · K'(t_{l+1,j} − t_i)` — the `W^T`
  transport of the downstream error. **The ONLY non-local term in the entire engine.**

So the engine is already a three-factor rule whose only non-locality is the hidden-layer
learning signal.

## 4. Temporal locality: already exact, already O(1) (measure, don't implement)

- Retained forward state per layer = `(t_prev, t_post, u')` = `O(n_l · B)`, independent of
  the time grid `G`. Grid scans and root-finding are transient recomputations, not stored
  trajectories. The backward runs over the spike DAG (depth = number of layers `L`), not
  over a time grid.
- Compare BPTT-over-grid: `O(Σ_l n_l · B · G)` stored activations. Our engine:
  `O(Σ_l n_l · B)` retained + `O(max n_l · B · G)` transient per layer.
- The eligibility `e_ji` is a closed-form algebraic quantity (the IFT derivative), NOT a
  recursively maintained trace. In single-spike TTFS, e-prop's temporal eligibility trace
  collapses to this single event-time object — so the "maximal locally-computable gradient"
  e-prop reconstructs through time is obtained here exactly and for free, because the
  spike-time DAG already encodes causality.
- SP-02's existence channel is layer-local too: per layer it needs only `t_prev`, `t_peak`,
  `u_peak` (the layer's own potential) and the target (silent neurons; correct class at the
  output, where the label lives). No cross-layer error, no O(T) state. The sub-problem
  doc's hope — "the informative dead-neuron signal and the hardware-friendly mechanism can
  be the same object" — holds.
- Gate D item "Memory target met: per-neuron eligibility trace O(1)... measured, not
  assumed" → E2 (Section 7) measures retained state vs a grid-BPTT baseline.

## 5. Spatial locality: the actual decision (D3)

The one non-local term = hidden-layer `lam` (`W^T` transport + global error). Replacement
options:

| Option | Removes W^T | Removes global error bus | Exact? | Memory | Accuracy risk | Verdict |
|---|---|---|---|---|---|---|
| **A. Per-layer local loss** (deep supervision; each layer trained by its own readout vs the label) | Yes | Label broadcast to each layer (target, not transported error) | Yes — each local loss is a real objective; the SP-01 adjoint applies per layer | O(1)/neuron | Low-moderate (proxy objectives; classic deep supervision) | **CHOSEN (primary)** |
| B. Feedback alignment / DFA / aDFA | Yes | No (output error still broadcast) | No — not the gradient of any conservative objective (gradcheck fails by construction) | O(1)/neuron + fixed random B | Moderate; deep layers ≈ 0 alignment (2606.21126) | Ablation: measures the cost of removing W^T while keeping global error (Q4.1) |
| C. e-prop trace | No (symmetric) / Yes (random e-prop) | No | No (learning signal approximate) | O(1)/neuron | Moderate | Already-have: our eligibility IS e-prop's trace, collapsed + exact (Sections 3–4); its spatial half = B. Not separately implemented |
| D. Forward-only contrastive (Traces Propagation style) | Yes | Yes — forward-only; no backward error, no per-layer classifier | No — contrastive surrogate | O(1)/neuron + fixed target projector | High on hard datasets (3.38 pp DVS-CIFAR10); not online (batch ≥ 2) | Ablation: measures the cost of removing the backward pass and trained readouts (Q4.1) |

**Why "exact" forces the choice:** an update is exact only if it is the derivative of a
real scalar objective. A and the reference engine satisfy this; B/C/D do not (by
construction). Since the project's standard is derived, exact, verified gradients
(SP-01/SP-02), the mechanism must be A. B and D are not "the same thing but slightly worse"
— they are a different mathematical object, so they belong in the measurement/ablation role,
quantifying the price of stricter locality for Q4.1, not as the headline.

**D3 (recommended):** primary mechanism = **A (per-layer local loss)**, running together
with the SP-02 existence channel at every layer. Reference ceiling = the existing exact
`W^T` backward (the no-regression target). Ablations = B (random-feedback) and D
(contrastive) on the same deep net, measured with the Section 6 diagnostics, to produce the
"cost of locality" numbers Q4.1 demands.

## 6. What "local" means for the target hardware (Gate D documentation)

For an event-driven neuromorphic accelerator (per-neuron state, no global shared bus beyond
the readout/label):

- **Local in time:** the update uses per-neuron/per-synapse state available at the events
  themselves — spike times `t_i`, `t_{l+1,j}`, potential slope `u'` — nothing stored across
  the trial beyond O(1)/neuron. No backward pass in time.
- **Local in space (mechanism A):** each layer's gradient uses only (i) that layer's spike
  times, (ii) that layer's readout, (iii) the label. No transposed weights, no error
  propagated between layers.
- **The label is broadcast to every layer** in all of A/D (A: as the supervised target of
  each layer's readout; D: as label-derived target traces via a fixed projector). None of the
  local mechanisms use an inter-layer error bus; the honest distinction is A removes error
  transport + weight transport (but trains per-layer classifiers), B removes weight transport
  but keeps a global error signal, D removes error transport AND trained classifiers
  (forward-only contrastive). This is the definition of "local" used for the Gate D
  documentation.
- **Online updates allowed:** yes for A — each layer can accumulate its local gradient as
  soon as its readout loss is available.

## 7. Gate D mapping (experiment plan)

| Gate D item | How met |
|---|---|
| Written comparison + D3 + rejected rationale | This doc + MEMORY.md |
| Gradcheck across depth (3+ layers) | E1: gradcheck the combined per-layer-loss objective on depth-3 and depth-4 nets (each local loss exact → total exact; same tolerance as SP-01) |
| Memory measured O(1) vs O(T) | E2: retained-state memory vs grid-BPTT as `G` grows (measured, not assumed) |
| No accuracy regression vs SP-02 solid state | E3: E9-equivalent task, local rule vs reference `W^T` backward, both with the existence channel |
| Fully-local rule on deeper arch (≥4 hidden layers) | E4: ≥4-hidden-layer net, deep supervision + existence channel; report per-layer cosine of the local learning signal vs the exact reference `lam`, scale stability, reference validity, depth utility (2606.21126 protocol) |
| Ablation: cost of stricter locality (Q4.1) | E5: same deep net with (B) random feedback and (D) contrastive; report accuracy gap + per-layer cosine; documents the trade-off if A is not "enough locality" |

## 8. Open questions

- Q4.1 (answered by E5): the measured accuracy cost of removing weight transport (B) and of
  removing the label broadcast (D) on a deep TTFS net.
- Q4.2: does the existence channel keep exact gradients at depth, or does variance dominate?
  (E4 + reference-validity diagnostic)
- Q4.3: does per-layer local loss change the SP-02 uniform-existence prior's role (layers
  trained to be decodable are alive)? (E4 observation)
- Q4.4: label fan-out (A) vs no-label contrastive (D) — is the accuracy gap small enough
  that A's label broadcast is the right engineering point? (E5)

## Sources

1. Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein, Maass — "A solution to the learning
   dilemma for recurrent networks of spiking neurons," Nature Communications 11:3625 (2020),
   s41467-020-17236-y.
2. Hao, Wan, Zhai — "What Accuracy and Gradient Cosine Miss: Evaluating Feedback Alignment
   via Scale Stability, Reference Validity, and Depth Utility," arXiv:2606.21126 (2026).
3. Pes, Yin, Stuijk, Corradi — "Traces Propagation: Memory-Efficient and Scalable
   Forward-Only Learning in Spiking Neural Networks," arXiv:2509.13053 (2025).
4. Lee, Xie, Gallagher, Zhang, Tu — "Deeply-Supervised Nets," AISTATS 2015.
5. Lillicrap et al. — "Random synaptic feedback weights support error backpropagation for
   deep learning," Nature Communications 7:13276 (2016).
6. Bacho & Chu — "Forward Direct Feedback Alignment for Online Gradient Descent Learning in
   Spiking Neural Networks" (SFDFA), arXiv:2403.08804 (2024) [DFA in SNNs; spike-grade
   hardware note].
