# SP-01 Research — Exact Gradient w.r.t. Spike Time

Status: ✅ Written · Date: 2026-08-13 · Sources: verified papers (below)

## 1. Literature findings (what is proven)

### 1.1 Exact gradients DO exist (the math is settled)

- **Lee et al. 2023 (PMLR, "Exact Gradient Computation for SNNs via Forward Propagation"):** applies the Implicit Function Theorem at the discrete spike times and *proves* SNNs have well-defined gradients w.r.t. weights, despite being non-differentiable in time. The firing-time equations, sorted in ascending spike-time order, form a lower-triangular Jacobian → IFT conditions always hold. This is the theoretical foundation for SP-01. Their "FP" algorithm propagates exact gradients *forward* using causality. ✅ verified
- **Göltz et al. 2021 (arXiv 1912.11443):** rigorous TTFS learning in LIF networks with finite time constants; closed-form first-spike times (Lambert-W) for the double-exponential PSP; exact error backpropagation. Reports 97.5% on MNIST-class tasks with exact gradients only.
- **EXODUS (Frontiers 2023):** uses IFT to compute gradients equivalent to BPTT while correctly accounting for the reset kernel — shows the "loopy" computational graph can be handled by IFT.
- **Nature Comms 2024 (0.3% energy TTFS):** confirms the exact gradient equals BPTT when spike presence/absence is FIXED. ⚠️ ALSO confirms the key negative result:

### 1.2 The known failure: vanishing/exploding gradients in deep TTFS

> "None of these theoretically sound studies could train a spiking network with more than six layers to high performance... deep TTFS networks generically cause vanishing-or-exploding gradients."

The `dt^(n+1)/dt^(n)` Jacobians multiply through layers; a naive product vanishes or explodes exponentially. Mitigations: special initialization, weight normalization, normalization layers. This is a **SP-01-scope constraint**: Gate A's gradient check is on 2-3 layers; deep stability is explicitly handed to SP-04 + a documented init strategy (ETTFS-init).

### 1.3 The reverse-gradient problem (subtle, must handle)

- **NeurIPS 2022 (event-driven backprop):** with the double-exponential kernel, when a postsynaptic spike fires on the *decreasing* part of the PSP, the timing gradient's sign flips relative to the correct direction. Fix in the literature: use a *monotone increasing* kernel in the backward pass. Verdict for SP-01: **document and measure**; do not silently ignore. It does NOT break the gradient check (which verifies the analytic gradient matches the finite-difference of the SAME forward model), but it can hurt training — that's a training-quality issue, tracked as SP-01 Q1.3.

### 1.4 Initialization is critical

- **ETTFS-init (2024/25):** Kaiming init on TTFS nets yields gradients at scale 1e-7 (dead training); their init + weight-norm yields 1e-3. TTFS needs its own init. We adopt a documented init (see engine) and verify gradient scale in experiments.

### 1.5 Exact spike-time simulation

- **Brette 2006 / Morrison 2007:** exact event-driven simulation via polynomial root-finding (commensurable time constants → polynomial in `x = e^{-t/tau_lcm}`; Sturm/Descartes for guaranteed spike tests). For SP-01 we implement a simpler robust approach: grid-bracket + Newton refinement, validated against a fine dense-grid simulation (Experiment E1). A Brette-style guaranteed method is a later optimization, not required for correctness.

## 2. The model (what we implement)

### 2.1 Kernel (normalized double-exponential PSP)

```
K(d) = ( exp(-d/tau_m) - exp(-d/tau_s) ) / (tau_m - tau_s) / k_peak ,   d > 0
K'(d) = ( -(1/tau_m) exp(-d/tau_m) + (1/tau_s) exp(-d/tau_s) ) / (tau_m - tau_s) / k_peak
k_peak = max_d K_unscaled(d)   (so peak value of K is exactly 1)
peak time:  d_peak = (tau_m*tau_s/(tau_m - tau_s)) * ln(tau_m/tau_s)
```

### 2.2 Membrane potential (feedforward, first-spike-per-neuron)

```
u_j(t) = sum_i w_ji * K(t - t_i) + w_jb * K(t - t_b)        (t_b = bias input time, fixed)
firing time t_fj:  smallest t > 0 with u_j(t) >= theta and u_j'(t) > 0
```

### 2.3 Exact gradient (IFT) — the SP-01 core

At `t_fj`: `u_j(t_fj) = theta`, differentiate w.r.t. `w_ji`:

```
0 = u_j'(t_fj) * dt_fj/dw_ji + K(t_fj - t_i)
=>  dt_fj/dw_ji = - K(t_fj - t_i) / u_j'(t_fj)
```

Chain rule into loss `L(t_out)`:

```
dL/dw_ji = (dL/dt_fj) * dt_fj/dw_ji
```

Adjoint propagation (dL/dt of pre-synaptic spike times), for fired postsynaptic neuron j at `t_fj` and presynaptic spike `i` at `t_i`:

```
dL/dt_i += (dL/dt_fj) * dt_fj/dt_i
dt_fj/dt_i = w_ji * K'(t_fj - t_i) / u_j'(t_fj)      (note: sign from dt_j/dK * dK/dt_i)
```

### 2.4 Output loss (latency cross-entropy, ANTLR-style)

```
z_k = -beta * t_out_k                       # earlier spike -> larger logit
p_k = exp(z_k) / sum_c exp(z_c)
L = -ln p_y
dL/dz_k = p_k - [k == y]
dL/dt_out_k = beta * ([k == y] - p_k)       # chain rule dz_k/dt_k = -beta
```

Corrected 2026-08-13: the `beta` factor and the sign were wrong in the original note (the `-beta` in `z_k` flips the sign). Both loss files implement `dL/dt = beta*(onehot - p)` and it is verified against spike-time finite differences in the gradient checks.

### 2.5 Bias

Modeled as a fixed input spike at time `t_b=0` with weight `w_jb` (bias column). This is a modeling choice (bias as an impulse input, not a constant current); consistent in forward and backward, so gradient check remains valid. Documented, not silently wrong.

## 3. What SP-01 explicitly does NOT solve (hand-offs)

| Issue | Handed to |
|---|---|
| Silent neurons (no spike → no gradient) | SP-02 |
| Deep vanishing/exploding gradients | SP-04 + init strategy (tracked Q1.3) |
| Multi-spike / reset coupling | SP-03 (needs D1 decision) |
| Local/online credit assignment | SP-04 |

## 4. Open questions that experiments must answer

- Q1.1: Does the analytic gradient match finite differences to < 1e-4? (Experiments E2/E3)
- Q1.2: Does exact-gradient training actually decrease loss on a toy TTFS task? (E4)
- Q1.3: How fast does the gradient magnitude decay with depth, and does the decreasing-kernel sign flip appear in practice? (E5) → informs SP-04.
- Q1.4: What is the right init scale so that a healthy fraction of hidden neurons fire initially (without SP-02)? (E5)

## Sources

1. Lee, et al. — "Exact Gradient Computation for SNNs via Forward Propagation," PMLR v206, 2023.
2. Göltz, et al. — "Fast and deep neuromorphic learning with time-to-first-spike coding," arXiv:1912.11443 (2021).
3. Wunderlich & Pehle — "Event-based backpropagation can compute exact gradients for SNNs," Scientific Reports, 2021 (EventProp).
4. Nature Communications — "High-performance deep SNNs with 0.3% the energy of ANNs using efficient spike-based learning," 2024.
5. NeurIPS 2022 — "Training SNNs with Event-driven Backpropagation" (reverse-gradient problem).
6. Brette 2006, Morrison et al. 2007 — exact event-driven simulation / spike-time root finding.
7. ANTLR (NeurIPS 2020) — latency loss and timing/activation gradient taxonomy.
8. ETTFS-init — "Efficiently Training TTFS SNNs from Scratch," arXiv:2410.23619.
