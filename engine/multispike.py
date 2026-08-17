"""SP-03 trainable multi-spike LIF network with exact saltation backward.

Wires the scalar `reset_lif.ResetLIF` engine into a multi-layer feedforward
network trainable on real data.  Each neuron receives Delta-current impulses
from presynaptic spikes, fires multiple times (hard reset), and the LAST
spike time is passed to the next layer (hybrid: multi-spike internal dynamics
+ TTFS-style inter-layer communication).

Gradients:
  - Weight gradients dL/dW: EXACT via `ResetLIF.sensitivity_all()` which
    propagates forward-mode variational states through ALL resets using the
    saltation matrix Xi_uu = (i_f - u_reset)/(i_f - theta).
  - Input-time gradients dL/dt_in: IFT at the last spike time (same formula
    as TTFS, exact for TTFS, good approximation for multi-spike).
  - Loss: latency cross-entropy on the last spike time of output neurons.

Architecture follows `snn_torch.TTFSNetTorch` conventions (W[l] = (n_out, n_in+1),
bias in W[:,-1]) but operates in pure Python/NumPy for correctness proof-of-concept.
"""
import math
import time

import numpy as np

from reset_lif import ResetLIF
from losses_torch import latency_cross_entropy


def _K_np(d, tm, ts):
    """Two-exponential kernel for numpy scalars/vectors.  0 for d <= 0."""
    d = np.maximum(d, 0.0)
    return (np.exp(-d / tm) - np.exp(-d / ts)) / (tm - ts)


def _Kd_np(d, tm, ts):
    """dK/dd, 0 for d <= 0."""
    d = np.maximum(d, 0.0)
    val = (-np.exp(-d / tm) / tm + np.exp(-d / ts) / ts) / (tm - ts)
    return np.where(d > 0, val, 0.0)


class MultiSpikeLIFNet:
    """Multi-layer LIF network with hard reset, trained via exact saltation
    backward (SP-03) + latency cross-entropy.

    Parameters
    ----------
    sizes : list[int]
        Layer sizes, e.g. [784, 128, 10].
    tm, ts, theta : float
        LIF parameters (membrane/synaptic time constants, threshold).
    u_reset : float
        Reset potential after firing (0 = hard reset to 0).
    t_max : float
        Simulation window [0, t_max].
    w_scale : float
        Weight init std.
    seed : int
        RNG seed.
    """

    def __init__(self, sizes, tm=15.0, ts=4.0, theta=1.0, u_reset=0.0,
                 t_max=40.0, w_scale=0.2, seed=0):
        self.sizes = list(sizes)
        self.n_layers = len(sizes) - 1
        self.tm = float(tm)
        self.ts = float(ts)
        self.theta = float(theta)
        self.u_reset = float(u_reset)
        self.t_max = float(t_max)
        self.lif = ResetLIF(tm=tm, ts=ts, theta=theta, u_reset=u_reset)
        rng = np.random.default_rng(seed)
        self.W = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            w = rng.standard_normal((b, a + 1)) * w_scale
            w[:, -1] = 0.0
            self.W.append(w)
        self._cache = None

    # ---- forward (single-neuron multi-spike, per-sample loop) -----------
    def _forward_layer(self, W, t_prev):
        """Run one layer: for each neuron j, sample b, feed input events
        (t_prev[i,b], W[j,i]) to ResetLIF.run(), return last spike time.

        Returns
        -------
        t_post : np.ndarray (n_cur, B)  -- last spike time or inf
        up : np.ndarray (n_cur, B)      -- u'(t_f) at last spike (0 if silent)
        fires_list : list[list]         -- all spike times per (j, b)
        ups_list : list[list]           -- all u'(t_f) per (j, b)
        """
        n_cur, n_inp = W.shape
        n_in = n_inp - 1
        B = t_prev.shape[1]
        t_post = np.full((n_cur, B), np.inf)
        up = np.zeros((n_cur, B))
        fires_list = [[None] * B for _ in range(n_cur)]
        ups_list = [[None] * B for _ in range(n_cur)]

        for j in range(n_cur):
            for b in range(B):
                inputs = []
                for i in range(n_in):
                    t_val = float(t_prev[i, b]) if np.isfinite(t_prev[i, b]) else self.t_max
                    inputs.append((t_val, float(W[j, i])))
                inputs.append((0.0, float(W[j, n_in])))
                fires, ups = self.lif.run_with_state(inputs, t_end=self.t_max)
                fires_list[j][b] = fires
                ups_list[j][b] = ups
                if fires:
                    t_post[j, b] = fires[-1]
                    up[j, b] = ups[-1]
        return t_post, up, fires_list, ups_list

    def forward(self, t_in):
        """t_in: (n_in, B) input spike times.  Returns t_out (n_out, B)."""
        t = t_in
        cache = []
        for l in range(self.n_layers):
            t_post, up, fl, ul = self._forward_layer(self.W[l], t)
            cache.append((t, t_post, up, fl, ul))
            t = t_post
        self._cache = cache
        return t

    # ---- backward (exact weight grads + IFT input-time grads) -----------
    def backward(self, dL_dt_out):
        """Weight gradients via sensitivity_all, input-time via IFT.

        Returns list of grad arrays, one per layer, shape (n_cur, n_in+1).
        """
        grads = [None] * self.n_layers
        lam = dL_dt_out

        for l in reversed(range(self.n_layers)):
            W = self.W[l]
            t_prev, t_post, up, fl, ul = self._cache[l]
            n_cur, n_inp = W.shape
            n_in = n_inp - 1
            B = t_prev.shape[1]
            g = np.zeros_like(W)

            for j in range(n_cur):
                for b in range(B):
                    lam_jb = lam[j, b]
                    if lam_jb == 0.0 or not np.isfinite(t_post[j, b]):
                        continue
                    up_jb = up[j, b]
                    if abs(up_jb) < 1e-12:
                        continue

                    # Build input events for this neuron/sample (all n_in+1)
                    inputs = []
                    for i in range(n_in):
                        t_val = float(t_prev[i, b]) if np.isfinite(t_prev[i, b]) else self.t_max
                        inputs.append((t_val, float(W[j, i])))
                    inputs.append((0.0, float(W[j, n_in])))

                    # Exact weight sensitivities via saltation
                    fires, dtdw_matrix = self.lif.sensitivity_all(
                        inputs, t_end=self.t_max)
                    if not fires:
                        continue

                    # Gradient of LAST spike time w.r.t. all weights
                    k_last = len(fires) - 1
                    dt_last_dW = np.array(dtdw_matrix[k_last])

                    # dL/dW[j,:] = (dL/dt_last) * (dt_last/dW) / u'(t_last)
                    # Wait -- the adjoint already incorporates dL/dt_post.
                    # The IFT: dt/dw = -s_u(t_f)/u'(t_f), and s_u is the
                    # variational state that sensitivity_all already divided by u'.
                    # So dt_last_dW IS dt/dw.  We just multiply by the adjoint.
                    g[j, :] += lam_jb * dt_last_dW

            grads[l] = g

            # Input-time adjoint for previous layer (TTFS IFT at last spike)
            lam_prev = np.zeros((n_in, B))
            for j in range(n_cur):
                for b in range(B):
                    if lam[j, b] == 0.0 or not np.isfinite(t_post[j, b]):
                        continue
                    up_jb = up[j, b]
                    if abs(up_jb) < 1e-12:
                        continue
                    for i in range(n_in):
                        if not np.isfinite(t_prev[i, b]):
                            continue
                        d = float(t_post[j, b] - t_prev[i, b])
                        if d <= 0:
                            continue
                        # dt_last/dt_in_i = -W[j,i] * K'(d) / u'(t_last)
                        kd = _Kd_np(d, self.tm, self.ts)
                        dt_dtin = -W[j, i] * kd / up_jb
                        lam_prev[i, b] += lam[j, b] * dt_dtin
            lam = lam_prev

        return grads

    def loss_and_grads(self, t_in, y, beta=1.0):
        """Forward + latency CE + backward.  Returns (loss, grads, t_out)."""
        t_out = self.forward(t_in)
        loss, dL_dt = latency_cross_entropy(
            torchify(t_out), y, self.t_max, beta)
        grads = self.backward(np.array(dL_dt))
        return loss, grads, t_out

    def sgd_step(self, grads, lr=0.01, wd=0.0, clip=None):
        """Vanilla SGD with optional weight decay + gradient clipping."""
        for l in range(self.n_layers):
            g = grads[l]
            if g is None:
                continue
            if clip is not None:
                g = np.clip(g, -clip, clip)
            if wd > 0.0:
                g = g + wd * self.W[l]
            self.W[l] -= lr * g


def torchify(arr):
    """np.ndarray -> torch.tensor (float64, no grad)."""
    import torch
    return torch.tensor(arr, dtype=torch.float64)


# ---------------------------------------------------------------------------
def train_cifar10(n_train=512, n_test=256, n_epochs=5, B=8,
                   sizes=(144, 32, 10), tm=15.0, ts=4.0, theta=1.0,
                   t_max=40.0, w_scale=2.0, lr=0.01, seed=42,
                   report_every=16):
    """CIFAR-10 training loop.  Loads via cifar_io (flat, pixel-rate encoding).

    This is a PROOF-OF-CONCEPT: pure-Python scalar loops are slow.  The point
    is to demonstrate that the exact saltation backward produces real gradients
    that train a real network on real data.  Full-scale training would use a
    CUDA-vectorized version (future work).
    """
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cifar_io import load_cifar10, to_grayscale_resized, encode_times, subset

    print(f"[SP-03 multispike] sizes={sizes}  tm={tm} ts={ts} theta={theta} "
          f"t_max={t_max}  lr={lr}  B={B}  n_train={n_train}  n_epochs={n_epochs}")
    print("Loading CIFAR-10...")
    x_train_full, y_train_full, x_test_full, y_test_full = load_cifar10()
    # Grayscale 12x12 -> 144 pixels (matches existing architectures)
    x_train = to_grayscale_resized(x_train_full, res=12).reshape(-1, 144)
    x_test = to_grayscale_resized(x_test_full, res=12).reshape(-1, 144)
    x_train = x_train[:n_train]
    y_train = y_train_full[:n_train]
    x_test = x_test[:n_test]
    y_test = y_test_full[:n_test]
    n_in = x_train.shape[1]

    # Convert pixel values to spike times (latency encoding: brighter = earlier)
    def encode(x):
        """(N, n_in) pixel [0,1] -> (n_in, N) spike times."""
        return encode_times(x.reshape(x.shape[0], -1),
                            t_lo=1.0, t_hi=t_max * 0.9).T

    net = MultiSpikeLIFNet([n_in] + list(sizes), tm=tm, ts=ts, theta=theta,
                           t_max=t_max, w_scale=w_scale, seed=seed)

    # Mini-batch SGD
    n_batches = n_train // B
    for epoch in range(n_epochs):
        t0 = time.time()
        perm = np.random.default_rng(seed + epoch).permutation(n_train)
        epoch_loss = 0.0
        epoch_correct = 0
        for mb in range(n_batches):
            idx = perm[mb * B:(mb + 1) * B]
            t_in = encode(x_train[idx])
            y_batch = y_train[idx]

            # Forward + backward
            loss, grads, t_out = net.loss_and_grads(t_in, y_batch)
            epoch_loss += loss

            # Accuracy
            fired = np.isfinite(t_out)
            if fired.any():
                pred = np.full(B, -1, dtype=int)
                for b in range(B):
                    fired_idx = np.where(fired[:, b])[0]
                    if len(fired_idx) > 0:
                        pred[b] = fired_idx[np.argmin(t_out[fired_idx, b])]
                epoch_correct += int((pred == y_batch).sum())

            # Weight update
            net.sgd_step(grads, lr=lr, clip=5.0)

            if (mb + 1) % report_every == 0:
                avg_loss = epoch_loss / (mb + 1)
                acc = epoch_correct / ((mb + 1) * B) * 100
                print(f"  epoch {epoch+1}  batch {mb+1}/{n_batches}  "
                      f"loss={avg_loss:.4f}  acc={acc:.1f}%")

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        acc = epoch_correct / n_train * 100
        print(f"epoch {epoch+1}/{n_epochs}  loss={avg_loss:.4f}  "
              f"train_acc={acc:.1f}%  time={elapsed:.1f}s")

    # Test
    print("\nTest set evaluation...")
    test_correct = 0
    n_test_batches = n_test // B
    for mb in range(n_test_batches):
        idx = mb * B
        t_in = encode(x_test[idx:idx + B])
        y_batch = y_test[idx:idx + B]
        t_out = net.forward(t_in)
        fired = np.isfinite(t_out)
        for b in range(B):
            fired_idx = np.where(fired[:, b])[0]
            if len(fired_idx) > 0:
                pred = fired_idx[np.argmin(t_out[fired_idx, b])]
                if pred == y_batch[b]:
                    test_correct += 1
    test_acc = test_correct / (n_test_batches * B) * 100
    print(f"test_acc={test_acc:.1f}%  ({test_correct}/{n_test_batches * B})")

    return net


# ---------------------------------------------------------------------------
def gradient_check(seed=7, eps=1e-5):
    """FD gradient check on a tiny 4->3->2 network.

    Verifies that the saltation-based backward matches central finite
    differences for EVERY weight in the network.  Uses large weights and
    early input times to ensure neurons fire (non-trivial gradients).
    """
    print("[SP-03 gradient check] tiny net: [4, 3, 2]")
    sizes = [4, 3, 2]
    net = MultiSpikeLIFNet(sizes, tm=10.0, ts=3.0, theta=0.5, t_max=20.0,
                           w_scale=5.0, seed=seed)
    # Set a positive bias so neurons can fire with 4 inputs
    for l in range(net.n_layers):
        net.W[l][:, -1] = 5.0
    B = 2
    rng = np.random.default_rng(seed + 100)
    t_in = rng.uniform(0.1, 3.0, (4, B))
    y = np.array([0, 1])

    # Verify some neurons actually fire
    loss, grads, t_out = net.loss_and_grads(t_in, y)
    n_fired = np.isfinite(t_out).sum()
    print(f"  loss = {loss:.6f}  neurons fired = {n_fired}/{t_out.size}")
    if n_fired == 0:
        print("  WARNING: no neurons fired -- trivial zero-gradient check")

    max_rel = 0.0
    n_checked = 0
    n_ok = 0
    n_nonzero = 0
    for l in range(net.n_layers):
        W_orig = net.W[l].copy()
        g_analytic = grads[l]
        for j in range(W_orig.shape[0]):
            for m in range(W_orig.shape[1]):
                net.W[l] = W_orig.copy()
                net.W[l][j, m] += eps
                fp, _, _ = net.loss_and_grads(t_in, y)
                net.W[l] = W_orig.copy()
                net.W[l][j, m] -= eps
                fn, _, _ = net.loss_and_grads(t_in, y)
                fd = (fp - fn) / (2 * eps)
                a = g_analytic[j, m]
                if abs(a) > 1e-12 or abs(fd) > 1e-12:
                    n_nonzero += 1
                denom = max(abs(a), abs(fd), 1e-12)
                rel = abs(a - fd) / denom
                n_checked += 1
                if rel < 1e-2 or abs(a - fd) < 1e-6:
                    n_ok += 1
                max_rel = max(max_rel, rel)
        net.W[l] = W_orig

    status = "PASS" if n_ok == n_checked else "FAIL"
    print(f"  checked {n_checked} weights: {n_ok}/{n_checked} within tolerance "
          f"(max_rel={max_rel:.2e})  nonzero_grads={n_nonzero} -> {status}")
    return n_ok == n_checked, max_rel


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--gradcheck":
        gradient_check()
    else:
        train_cifar10()
