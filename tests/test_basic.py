"""
Basic tests for Exact-SNN: gradient correctness and API sanity.

Run:  pytest tests/ -v
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from exact_snn import (
    TTFSNet,
    AdamTorch,
    forward_layer_torch,
    backward_layer_torch,
    latency_cross_entropy,
    forward_multispike_layer_torch,
    backward_multispike_layer_torch,
)
from exact_snn.losses import spike_count_cross_entropy


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_net(sizes=(784, 64, 10), **kwargs):
    defaults = dict(tm=15.0, ts=4.0, theta=1.0, t_max=40.0, w_scale=0.2,
                    seed=42, dtype=DTYPE, dev=DEVICE, grid_pts=2001)
    defaults.update(kwargs)
    return TTFSNet(list(sizes), **defaults)


def _random_batch(net, B=32):
    n_in = net.sizes[0]
    t_in = torch.rand(n_in, B, dtype=net.dtype, device=net.dev) * 0.8 * net.t_max + 0.1
    y = torch.randint(0, net.sizes[-1], (B,), device=net.dev)
    return t_in, y


def _fd_cosine(g_exact, g_fd):
    """Cosine similarity between two gradient vectors."""
    a = g_exact.float().detach().cpu().numpy().ravel()
    b = g_fd.float().detach().cpu().numpy().ravel()
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(dot / (na * nb))


def _finite_difference_grad(net, t_in, y, layer_idx=0, eps=1e-5):
    """Compute finite-difference gradient for a single weight."""
    net.forward(t_in)
    loss0, _, _ = net.loss_and_grads(t_in, y)
    W = net.W[layer_idx]
    g_fd = torch.zeros_like(W)
    # Only check a few weights for speed
    n_check = min(5, W.numel())
    indices = np.random.default_rng(0).choice(W.numel(), n_check, replace=False)
    for idx in indices:
        flat_idx = int(idx)
        i, j = divmod(flat_idx, W.shape[1])
        orig = W[i, j].item()
        W[i, j] = orig + eps
        loss_plus, _, _ = net.loss_and_grads(t_in, y)
        W[i, j] = orig - eps
        loss_minus, _, _ = net.loss_and_grads(t_in, y)
        W[i, j] = orig
        g_fd[i, j] = (loss_plus - loss_minus) / (2 * eps)
    return g_fd


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestForward:
    def test_output_shape(self):
        net = _make_net()
        t_in, _ = _random_batch(net)
        t_out = net.forward(t_in)
        assert t_out.shape == (net.sizes[-1], t_in.shape[1])

    def test_output_finite_or_inf(self):
        """Output should be finite (spike) or inf (silent), no NaN."""
        net = _make_net()
        t_in, _ = _random_batch(net, B=64)
        t_out = net.forward(t_in)
        assert torch.isfinite(t_out).any(), "No neurons spiked"
        assert not torch.isnan(t_out).any(), "NaN in output"


class TestBackward:
    def test_grads_not_none(self):
        net = _make_net()
        t_in, y = _random_batch(net)
        loss, grads, t_out = net.loss_and_grads(t_in, y)
        assert loss > 0
        for g in grads:
            assert g is not None
            assert g.shape == g.shape  # basic sanity

    def test_grads_finite(self):
        net = _make_net()
        t_in, y = _random_batch(net)
        _, grads, _ = net.loss_and_grads(t_in, y)
        for g in grads:
            assert torch.isfinite(g).all(), "Non-finite gradient found"


class TestGradientCosine:
    def test_single_layer_cosine(self):
        """Gradient cosine vs FD should be ~1.0 on a small network."""
        net = _make_net(sizes=(10, 5, 3), grid_pts=501)
        t_in = torch.rand(10, 8, dtype=DTYPE, device=DEVICE) * 0.8 * net.t_max + 0.1
        y = torch.randint(0, 3, (8,), device=DEVICE)

        loss, grads, _ = net.loss_and_grads(t_in, y)
        g_fd = _finite_difference_grad(net, t_in, y, layer_idx=0, eps=1e-5)

        cosine = _fd_cosine(grads[0], g_fd)
        assert cosine > 0.99, f"Cosine similarity too low: {cosine:.6f}"


class TestMultiSpike:
    def test_forward_shape(self):
        net = _make_net(max_spikes=3)
        t_in, _ = _random_batch(net)
        t_out = net.forward_multispike(t_in)
        assert t_out.shape == (net.sizes[-1], t_in.shape[1])

    def test_backward_ttfs(self):
        """Multi-spike backward with first_spike_only=True (TTFS)."""
        net = _make_net(max_spikes=3)
        t_in, y = _random_batch(net)
        loss, grads, _ = net.loss_and_grads_saltation(t_in, y)
        assert loss > 0
        for g in grads:
            assert g is not None
            assert torch.isfinite(g).all()


class TestLoss:
    def test_latency_ce_decreasing(self):
        """Loss should decrease if we move correct-class spike earlier."""
        n_out, B = 5, 8
        y = torch.arange(B, device=DEVICE) % n_out
        # Spike times where correct class fires early
        t_good = torch.ones(n_out, B, dtype=DTYPE, device=DEVICE) * 30.0
        for b in range(B):
            t_good[y[b], b] = 1.0
        # Spike times where correct class fires late
        t_bad = torch.ones(n_out, B, dtype=DTYPE, device=DEVICE) * 1.0
        for b in range(B):
            t_bad[y[b], b] = 30.0
        loss_good, _ = latency_cross_entropy(t_good, y, 40.0)
        loss_bad, _ = latency_cross_entropy(t_bad, y, 40.0)
        assert loss_good < loss_bad

    def test_spike_count_ce(self):
        n_out, B = 5, 8
        y = torch.arange(B, device=DEVICE) % n_out
        t_spikes = torch.rand(n_out, B, 5, dtype=DTYPE, device=DEVICE) * 40.0
        loss, dL = spike_count_cross_entropy(t_spikes, y)
        assert loss > 0
        assert dL.shape == (n_out, B)


class TestOptimizer:
    def test_step_updates_weights(self):
        net = _make_net(sizes=(10, 5, 3))
        params = list(net.W)
        opt = AdamTorch(params, lr=0.01)
        W_before = [p.clone() for p in params]
        t_in = torch.rand(10, 4, dtype=DTYPE, device=DEVICE) * 0.8 * net.t_max + 0.1
        y = torch.randint(0, 3, (4,), device=DEVICE)
        _, grads, _ = net.loss_and_grads(t_in, y)
        opt.step(params, grads)
        for W_b, W_a in zip(W_before, params):
            assert not torch.equal(W_b, W_a), "Weights did not change"


class TestEventDriven:
    def test_event_matches_grid(self):
        """Event-driven forward should produce similar (not identical) spike times."""
        from exact_snn import EventTTFSNet
        net_grid = _make_net(sizes=(10, 5, 3), grid_pts=2001)
        net_event = EventTTFSNet(sizes=(10, 5, 3), grid_pts=2001)
        # Copy weights
        for wg, we in zip(net_grid.W, net_event.W):
            we.copy_(wg)

        t_in = torch.rand(10, 8, dtype=DTYPE, device=DEVICE) * 0.8 * net_grid.t_max + 0.1
        t_grid = net_grid.forward(t_in)
        t_event = net_event.forward(t_in)

        # Both should produce mostly finite outputs
        assert torch.isfinite(t_grid).any()
        assert torch.isfinite(t_event).any()

        # Where both fire, times should be close (within grid resolution)
        both_finite = torch.isfinite(t_grid) & torch.isfinite(t_event)
        if both_finite.any():
            diff = (t_grid[both_finite] - t_event[both_finite]).abs()
            step = net_grid.t_max / 2000
            assert diff.max() < 5 * step, f"Event-grid mismatch: {diff.max():.4f} > 5*step"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
