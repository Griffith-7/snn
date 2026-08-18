# Exact-SNN

**Exact-gradient training for Spiking Neural Networks — no surrogate approximations.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

SNNs are the future of energy-efficient AI, but training them has always required
compromising on gradient accuracy. Exact-SNN solves the non-differentiable spike
problem using mathematically exact Implicit Function Theorem (IFT) gradients,
saltation matrices for reset handling, and a principled escape-noise mechanism
for silent neurons.

## Install

```bash
pip install exact-snn
```

Or from source:

```bash
git clone https://github.com/Griffith-7/snn.git
cd snn
pip install -e .
```

## Quick Start

```python
from exact_snn import TTFSNet, EventTTFSNet, AdamTorch, latency_cross_entropy

# Build a 2-layer SNN
net = TTFSNet([784, 128, 10])

# Or use the event-driven engine (2-3x faster, same math)
net = EventTTFSNet([784, 128, 10])

# Train with exact IFT gradients
params = net.W + net.R
opt = AdamTorch(params, lr=2e-2, clip=5.0)

import torch, numpy as np
t_in = torch.tensor(np.random.uniform(0.5, 8.0, (784, 32)),
                    dtype=net.dtype, device=net.dev)
y = torch.tensor(np.random.randint(0, 10, 32), device=net.dev)

loss, grads, grads_R, _ = net.local_learning_grads(
    t_in, y, T_noise=1.0, lam=5.0, mode="deep")
opt.step(params, list(grads) + grads_R)
```

## Extended Architectures

```python
from exact_snn.extended import ConvTTFSLayer, SNNConvNet, RecurrentTTFSLayer, MultiSpikeNet

# Convolutional SNN
conv_layer = ConvTTFSLayer(in_channels=3, out_channels=16, kernel_size=3)
net = SNNConvNet([784, 128, 10])

# Multi-spike (rate-coded) network
net = MultiSpikeNet([784, 128, 10], max_spikes=5)
```

## What's Inside

| Module | Description |
|---|---|
| `exact_snn.core` | TTFSNet — exact IFT gradients via grid scan |
| `exact_snn.event` | EventTTFSNet — closed-form inter-event (2-3x faster) |
| `exact_snn.extended` | Conv, BatchNorm, Recurrent, Multi-Spike layers |
| `exact_snn.reset` | ResetLIF — saltation matrices for LIF with reset |
| `exact_snn.losses` | Latency cross-entropy, spike-count CE, rate-latency |
| `exact_snn.optim` | Adam optimizer (no autograd, raw tensors) |

## The Math

The non-differentiable spike problem has 3 layers:

1. **Spike generation** (Heaviside step) — Implicit Function Theorem:
   `dt_f/dw = -K(t_f - t_in) / u'(t_f)`

2. **Threshold crossing** — inter-event closed-form analysis:
   membrane `u(t) = A·e^(-t/tm) + B·e^(-t/ts)` has at most one critical point

3. **Instantaneous reset** — saltation matrices:
   `Xi_uu = (i_f - u_reset) / (i_f - theta)`

4. **Silent neurons** — escape-noise expectation:
   Gaussian-perturbed spike times give smooth existence gradients

## Design Criteria

| # | Criterion | Status |
|---|-----------|--------|
| C1 | Exact gradients (cosine > 0.90 vs finite-difference) | PASS |
| C2 | Scalability (time ratio d10/d2 < 10x) | PASS |
| C3 | Transfer gap < 15% | PASS (6.5%) |
| C4 | ANN-competitive (> 40% of baseline) | PASS (50%) |
| C5 | HW-compatible (O(1) memory, local learning) | PASS |
| C6 | General-purpose (3 models, 3 codings, 4+ outputs) | PASS |

## License

MIT
