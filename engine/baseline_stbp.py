"""Surrogate-gradient (STBP-style) baseline for Phase 5, written from scratch.

Apples-to-apples with the exact TTFS engine:
- Same model family: feedforward MLP, same double-exponential PSP kernel
  (tm, ts, k_peak), same theta, same bias-as-t=0-input-spike model, same
  latency cross-entropy loss on first-spike times, same input encoding.
- Only difference: the discrete-time surrogate-gradient learning rule replaces
  the exact IFT spike-time gradient. The spike decision is a straight-through
  estimator (hard threshold forward, sigmoid surrogate backward).

Single-spike semantics match the engine (a neuron fires at most once; it is
dead afterwards), so the weighted-mean spike time equals the first-spike time
exactly and stays differentiable through the surrogate.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import numpy as np


def _k_peak(tm, ts):
    s = (tm * ts / (tm - ts)) * math.log(tm / ts)
    return (math.exp(-s / tm) - math.exp(-s / ts)) / (tm - ts)


class SpikeSTE(torch.autograd.Function):
    """Straight-through estimator: hard spike forward, atan (Lorentzian)
    surrogate backward. The Lorentzian tails do not vanish far from threshold
    (unlike the sigmoid surrogate), so silent/near-threshold neurons still
    receive a learning signal."""

    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        s = ctx.slope
        g = s / (1.0 + (s * x) * (s * x))
        return grad * g, None


def _atan_q(x, slope):
    """Smooth spike surrogate in [0,1): (1/pi)*atan(slope*x) + 1/2.
    Used as readout timing weights; derivative is the Lorentzian above."""
    return (torch.atan(slope * x) / math.pi) + 0.5


class STBPNet(nn.Module):
    def __init__(self, sizes, tm=15.0, ts=4.0, theta=1.0, t_max=40.0,
                 T=160, w_scale=0.1, bias_val=0.2, seed=0, slope=1.0,
                 dtype=torch.float32, dev=None):
        super().__init__()
        self.sizes = list(sizes)
        self.n_layers = len(sizes) - 1
        self.tm = float(tm)
        self.ts = float(ts)
        self.theta = float(theta)
        self.t_max = float(t_max)
        self.T = int(T)
        self.dt = float(t_max) / (T - 1)
        self.slope = float(slope)
        self.dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.k_peak = _k_peak(self.tm, self.ts)
        delays = torch.arange(T, dtype=dtype) * self.dt
        self.kernel = torch.where(
            delays > 0.0,
            (torch.exp(-delays / self.tm) - torch.exp(-delays / self.ts))
            / (self.tm - self.ts) / self.k_peak,
            torch.zeros_like(delays)).to(self.dev)
        rng = np.random.default_rng(seed)
        self.W = nn.ParameterList()
        for a, b in zip(sizes[:-1], sizes[1:]):
            w = (rng.standard_normal((b, a + 1)) * w_scale).astype(np.float32)
            w[:, -1] = bias_val
            self.W.append(nn.Parameter(torch.tensor(w, dtype=dtype).to(self.dev)))
        self._events = None

    def _spike_train(self, t_in):
        """t_in: (n_in, B) input spike times -> (B, n_in, T) trains. The bias
        column is appended per layer in forward() (bias = a t=0 input spike)."""
        n_in, B = t_in.shape
        k = (t_in / self.dt).round().long().clamp(0, self.T - 1)
        S = torch.zeros(B, n_in, self.T, dtype=self.dtype, device=self.dev)
        S[torch.arange(B).view(1, -1), torch.arange(n_in).view(-1, 1), k] = 1.0
        return S

    def forward(self, t_in):
        """t_in: (n_in, B). Returns t_out (n_out, B) surrogate first-spike times.

        Faithful discrete-time analog of the exact engine: hard single spikes are
        propagated event-wise through the same double-exponential PSP kernel.
        The readout is a soft-argmax spike time t = sum_k t_k q_k / sum_k q_k with
        q = atan(slope*(u-theta)): differentiable everywhere with non-vanishing
        Lorentzian derivative, so silent output neurons still receive a (weak)
        timing signal; hard spikes (u >= theta, single-spike) are counted for
        energy (SynOps).
        """
        S = self._spike_train(t_in)
        events = []
        fracs = []
        B = S.shape[0]
        q = None
        pad = self.T - 1
        for l in range(self.n_layers):
            W = self.W[l]
            n_cur, n_inp = W.shape
            S_bias = torch.zeros(B, 1, self.T, dtype=self.dtype, device=self.dev)
            S_bias[:, 0, 0] = 1.0
            S_in = torch.cat([S, S_bias], dim=1)  # (B, n_in+1, T), bias col
            # membrane = causal temporal convolution of the spike train with the
            # PSP kernel weighted per-synapse: weight[o,i,j] = W[o,i] * K_rev[j],
            # with the kernel reversed so output[t] = sum_m W*K[m]*S_in[t-m].
            krev = torch.flip(self.kernel, dims=(0,)).to(self.dtype)
            filt = W.unsqueeze(-1) * krev.view(1, 1, -1)
            u = F.conv1d(S_in, filt, padding=pad)[:, :, :self.T]
            s_raw = SpikeSTE.apply(u - self.theta, self.slope)
            S = s_raw * (torch.cumsum(s_raw, dim=2) <= 1.0).to(s_raw.dtype)
            events.append(float(S.sum().item()))
            fracs.append(float((S.sum(dim=2) > 0.0).float().mean().item()))
            q = _atan_q(u - self.theta, self.slope)
        t_idx = (torch.arange(self.T, dtype=self.dtype, device=self.dev)
                 * self.dt).view(1, 1, -1)
        denom = q.sum(dim=2)
        t_out = torch.where(
            denom > 1e-9, (t_idx * q).sum(dim=2) / denom.clamp(min=1e-9),
            torch.full_like(denom, 2.0 * self.t_max + 10.0))
        self._events = events
        self._fire_frac = fracs
        return t_out.t().contiguous()  # (n_out, B), engine convention

    def events_per_layer(self):
        return list(self._events)

    def latency_loss(self, t_out, y, beta=1.0):
        """Autograd-compatible latency CE, mirroring engine losses_torch:
        p_k = softmax(-beta * t_k); silent outputs are placed at a large finite
        time for the softmax only (gradient 0 there via torch.where)."""
        B = t_out.shape[1]
        t = torch.where(torch.isfinite(t_out), t_out,
                        torch.full_like(t_out, 2.0 * self.t_max + 10.0))
        logits = -beta * t
        logits = logits - logits.max(dim=0, keepdim=True).values
        p = torch.exp(logits)
        p = p / p.sum(dim=0, keepdim=True)
        loss = -torch.log(p[y, torch.arange(B, device=t_out.device)] + 1e-12).mean()
        return loss
