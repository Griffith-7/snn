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
pip install -r requirements.txt   # for training scripts
```

## Quick Start

### Basic TTFS Network

```python
from exact_snn import TTFSNet, AdamTorch
import torch, numpy as np

# Build a 2-layer TTFS SNN
net = TTFSNet([784, 128, 10])

# Train with exact IFT gradients
params = net.W + net.R
opt = AdamTorch(params, lr=2e-2, clip=5.0)

t_in = torch.tensor(np.random.uniform(0.5, 8.0, (784, 32)),
                     dtype=net.dtype, device=net.dev)
y = torch.tensor(np.random.randint(0, 10, 32), device=net.dev)

loss, grads, grads_R, _ = net.local_learning_grads(
    t_in, y, T_noise=1.0, lam=5.0, mode="deep")
opt.step(params, list(grads) + grads_R)
```

### Event-Driven Engine (2-3x Faster)

```python
from exact_snn import EventTTFSNet

# Same math, closed-form inter-event — 2-3x faster
net = EventTTFSNet([784, 128, 10])
```

### Convolutional SNN for CIFAR-10

```python
from exact_snn.extended import SNNConvNet
from exact_snn.optim import AdamTorch
import torch

# Conv(3->16) -> Pool -> Conv(16->32) -> Pool -> FC(2048->64->10)
net = SNNConvNet(in_channels=3, h_w=32, n_classes=10,
                 tm=15.0, ts=4.0, theta=0.5, t_max=40.0,
                 grid_pts=31, dtype=torch.float32, device=torch.device("cuda"))

# Train with exact gradients through all layers
params = [net.conv1.W, net.conv2.W] + list(net.fc.W)
opt = AdamTorch(params, lr=0.005, clip=5.0)

# Forward + backward in one call
t_images = torch.rand(4, 3, 32, 32, dtype=torch.float32, device="cuda")
y = torch.randint(0, 10, (4,), device="cuda")
loss, grads, grads_R, stats = net.loss_and_grads(t_images, y)
opt.step(params, [g if g is not None else torch.zeros_like(p)
                   for g, p in zip(grads + (grads_R or []), params)])
```

## Training

### MNIST (TTFS, fully-connected)

```bash
python train_mnist.py --epochs 10 --lr 0.005
```

This trains a 784-256-10 TTFS SNN on MNIST with exact IFT gradients. The
input is latency-encoded (brighter pixels → earlier spikes). Output neuron
with earliest spike wins.

### Options

```
--arch 784-256-10        Network sizes (dash-separated)
--epochs 10              Training epochs
--batch-size 128         Mini-batch size
--lr 0.005               Learning rate
--clip 5.0               Gradient clipping
--t-max 40.0             Max simulation time (ms)
--tm 15.0                Membrane time constant
--ts 4.0                 Synaptic time constant
--device cuda            Device (auto/cuda/cpu)
--max-samples 5000       Limit training set for quick tests
```

## Architecture

| Module | Description |
|---|---|
| `exact_snn.core` | TTFSNetTorch — exact IFT gradients via vectorized grid scan |
| `exact_snn.event` | EventTTFSNet — closed-form inter-event (2-3x faster) |
| `exact_snn.extended` | ConvTTFSLayer, SNNConvNet, RecurrentTTFSLayer, MultiSpikeNet |
| `exact_snn.reset` | ResetLIF — saltation matrices for LIF with reset |
| `exact_snn.losses` | Latency cross-entropy, spike-count CE, rate-latency |
| `exact_snn.optim` | AdamTorch — raw-tensor optimizer (no autograd overhead) |

## Key Results

| Result | Details |
|---|---|
| **Gradient accuracy** | cosine = 1.000000 vs finite-difference on all layer types |
| **Event-driven speedup** | 2.89x faster than grid-scan engine |
| **Conv SNN on CIFAR-10** | Loss drops 7.7 -> 2.6 in 1 epoch with exact gradients |

**Honest limitations:**
- FC-only TTFS SNNs achieve ~10% accuracy on CIFAR-10 (random guessing) — spatial features require convolutional architecture
- CIFAR-10 training to convergence (final accuracy) is ongoing work
- The escape-noise temperature `T_noise` is a hyperparameter that may need tuning per task
- No distributed/multi-GPU support yet

## The Math

The non-differentiable spike problem has 3 layers:

1. **Spike generation** (Heaviside step) — Implicit Function Theorem:
   `dt_f/dw = -K(t_f - t_in) / u'(t_f)`

2. **Threshold crossing** — inter-event closed-form analysis:
   membrane `u(t) = A*e^(-t/tm) + B*e^(-t/ts)` has at most one critical point

3. **Instantaneous reset** — saltation matrices:
   `Xi_uu = (i_f - u_reset) / (i_f - theta)`

4. **Silent neurons** — escape-noise expectation:
   Gaussian-perturbed spike times give smooth existence gradients

## API Reference

### Core Classes

- **`TTFSNet(sizes, ...)`** — Standard TTFS SNN with grid-scan forward
- **`EventTTFSNet(sizes, ...)`** — Event-driven TTFS SNN (2-3x faster)
- **`AdamTorch(params, lr, clip)`** — Adam optimizer for raw tensors

### Extended Classes

- **`ConvTTFSLayer(in_ch, out_ch, kernel, ...)`** — Convolutional TTFS layer
- **`SNNConvNet(in_channels, h_w, n_classes, ...)`** — Full conv SNN pipeline
- **`MultiSpikeNet(sizes, max_spikes, ...)`** — Multi-spike (rate-coded) SNN
- **`RecurrentTTFSLayer(n_in, n_out, ...)`** — Recurrent TTFS with eligibility traces
- **`SpikeNorm(n_features)`** — Spike-time batch normalization
- **`ResetLIF(tm, ts, theta, ...)`** — LIF neuron model with saltation matrices

### Loss Functions

- **`latency_cross_entropy(t_out, y, t_max)`** — Latency-based cross-entropy
- **`spike_count_cross_entropy(t_spikes, y)`** — Spike-count cross-entropy
- **`rate_latency_loss(t_spikes, y, t_max)`** — Combined rate-latency loss

### Standalone Functions

- **`forward_layer_torch(W, t_prev, ...)`** — Exact IFT forward for one layer
- **`backward_layer_torch(W, t_prev, ...)`** — Exact IFT backward for one layer
- **`forward_multispike_layer(W, t_prev, ...)`** — Multi-spike forward
- **`peak_margin_torch(W, t_prev, ...)`** — Peak-margin existence regularization

## License

MIT
