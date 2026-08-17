"""SP-01 experiments (GPU/torch engine + NumPy oracle).

Run:  python engine/experiments/exp_sp01.py
Writes JSON to docs/results/sp01/ and prints a summary.

E2/E3  exact-gradient check vs central finite differences (2- and 3-layer).
E4  training smoke test on the torch/GPU engine (synthetic TTFS 2-class task).
E5  edge cases: gradient scale vs depth, silent-neuron zero gradient.

Note: E1, E1b, and E5b require the NumPy oracle (engine/snn.py) which has
been removed. They are kept as reference code but will raise ImportError if
called without the oracle.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snn_torch import TTFSNetTorch, forward_layer_torch, device, _K, _Kd
from losses_torch import latency_cross_entropy
from optimizers_torch import AdamTorch


RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "..", "docs", "results", "sp01")


def _torch_fired_sets(net, t_in):
    net.forward(t_in)
    return [torch.isfinite(c[1]).cpu().numpy() for c in net._cache]


def _status_flip(base, plus, minus):
    return any(not np.array_equal(fp, fm) or not np.array_equal(fp, fb)
               for fb, fp, fm in zip(base, plus, minus))


def _build_smooth_net(depth, seed, n_in=8, widths=None, B=8, n_out=3,
                      w_lo=0.8, w_hi=1.2, bias_val=0.3, t_lo=0.3, t_hi=1.0,
                      min_margin=0.05, min_fired=0.9, max_tries=20):
    """Build a forward-pass that is everywhere smooth for central-difference
    gradient checking.

    Central differences are only valid when the spike-time map is smooth in a
    neighbourhood of the operating point on BOTH sides. The spike-time map has
    a non-differentiable kink wherever a firing time t_f coincides with an
    input arrival time (the clamped kernel onset at d = t_f - t_in = 0; also
    whenever a bias-driven layer fires in synchrony with its inputs). We
    therefore construct networks with positive weights and early, spread input
    times, so every layer fires strictly AFTER all of its inputs (margin > 0),
    which makes both FD sides smooth. Returns (net, t_in, y, stats); re-samples
    seeds until the conditioning thresholds are met.
    """
    sizes = [n_in] + (widths or [12] * (depth - 1) + [n_out])
    dev = TTFSNetTorch([n_in, 2, n_out], seed=seed, w_scale=1.0,
                       bias_val=bias_val, peak_tol=0.0).dev
    for attempt in range(max_tries):
        rng = np.random.default_rng(seed + 1000 * attempt)
        net = TTFSNetTorch(sizes, seed=seed, w_scale=1.0, bias_val=bias_val,
                           peak_tol=0.0)
        for W in net.W:
            a, b = W.shape
            W[:, :b - 1] = torch.tensor(rng.uniform(w_lo, w_hi, (a, b - 1)),
                                        dtype=W.dtype, device=W.device)
        t_in = torch.tensor(rng.uniform(t_lo, t_hi, (n_in, B)),
                            dtype=net.dtype, device=dev)
        y = torch.tensor(rng.integers(0, n_out, size=B), device=dev)
        net.forward(t_in)
        fired = np.concatenate([torch.isfinite(c[1]).cpu().numpy().ravel()
                                for c in net._cache])
        fired_frac = float(fired.mean())
        margins = []
        for l in range(1, len(net._cache)):
            tp = net._cache[l][1].cpu().numpy()
            tp_prev = net._cache[l][0].cpu().numpy()
            for j in range(tp.shape[0]):
                for b in range(tp.shape[1]):
                    tf = tp[j, b]
                    if np.isfinite(tf):
                        mx = np.max(tp_prev[:, b])
                        if np.isfinite(mx):
                            margins.append(tf - mx)
        min_margin = float(min(margins)) if margins else -1.0
        stats = {"fired_frac": fired_frac, "min_margin": min_margin,
                 "attempts": attempt + 1}
        if fired_frac >= min_fired and min_margin >= min_margin:
            return net, t_in, y, stats
    raise RuntimeError(f"no well-conditioned config found in {max_tries} tries")


def _gradcheck_torch(depth, seed, n_in=8, widths=None, B=8, eps=1e-5,
                     n_dir=6, n_w=15, target_rel=1e-4, atol=1e-6,
                     progress=None):
    """Exact-gradient check on the torch/GPU engine.

    Two independent checks:
      (a) dot-product test: analytic grad dot a random direction vs its finite
          difference -- exercises ALL weights with just 2 forwards per direction;
      (b) per-weight central differences on a random sample of weights.
    The forward pass is built with positive weights + early inputs so every
    layer fires strictly after its inputs; central differences are otherwise
    invalid at the kernel-onset kinks (t_f == t_in; see E5d). Weights whose
    perturbation flips a spike's fired/silent status are still skipped (the
    spike-time gradient is undefined there by design; see E5b).
    """
    net, t_in, y, stats = _build_smooth_net(depth, seed, n_in=n_in,
                                            widths=widths, B=B)
    dev = net.dev
    rng = np.random.default_rng(seed + 12345)
    loss, grads, _ = net.loss_and_grads(t_in, y)
    base_fired = _torch_fired_sets(net, t_in)

    def fwd_loss_fired():
        t_out = net.forward(t_in)
        l, _ = latency_cross_entropy(t_out, y, net.t_max)
        return l, [torch.isfinite(c[1]).cpu().numpy() for c in net._cache]

    dot_errors = []
    for d in range(n_dir):
        V = [torch.randn_like(net.W[l]) for l in range(depth)]
        for l in range(depth):
            V[l].div_(V[l].norm() + 1e-12)
        for l in range(depth):
            net.W[l].add_(eps * V[l])
        l_plus, f_plus = fwd_loss_fired()
        for l in range(depth):
            net.W[l].sub_(2.0 * eps * V[l])
        l_minus, f_minus = fwd_loss_fired()
        for l in range(depth):
            net.W[l].add_(eps * V[l])
        if _status_flip(base_fired, f_plus, f_minus):
            continue
        fd = (l_plus - l_minus) / (2.0 * eps)
        an = sum((grads[l] * V[l]).sum() for l in range(depth) if grads[l] is not None)
        dot_errors.append(abs(float(an) - float(fd)) / (abs(float(fd)) + 1e-12))

    w_errors = []
    w_abs_errors = []
    n_near_zero = 0
    skipped_flips = 0
    for l in range(depth):
        W = net.W[l]
        flat = rng.choice(int(W.numel()), size=min(n_w, int(W.numel())), replace=False)
        for k, fi in enumerate(flat):
            idx = np.unravel_index(int(fi), W.shape)
            orig = float(W[idx])
            W[idx] = orig + eps
            l_plus, f_plus = fwd_loss_fired()
            W[idx] = orig - eps
            l_minus, f_minus = fwd_loss_fired()
            W[idx] = orig
            if _status_flip(base_fired, f_plus, f_minus):
                skipped_flips += 1
                continue
            fd = (l_plus - l_minus) / (2.0 * eps)
            an = float(grads[l][idx])
            if max(abs(an), abs(fd)) < atol:
                n_near_zero += 1
                w_abs_errors.append(abs(an - fd))
            else:
                w_errors.append(abs(an - fd) / (abs(fd) + 1e-12))
            if progress is not None:
                progress()

    def _stat(a):
        a = np.array(a)
        return {"mean": float(a.mean()) if a.size else None,
                "median": float(np.median(a)) if a.size else None,
                "max": float(a.max()) if a.size else None,
                "p99": float(np.percentile(a, 99)) if a.size else None}

    dot_stat = _stat(dot_errors)
    w_stat = _stat(w_errors)
    w_abs_stat = _stat(w_abs_errors)
    fired_frac = float(np.mean([s.mean() for s in base_fired]))
    return {
        "depth": depth,
        "seed": seed,
        "sizes": net.sizes,
        "loss": float(loss),
        "n_dot_dirs": len(dot_errors),
        "dot_rel_err": dot_stat,
        "n_weights_checked": len(w_errors),
        "n_near_zero_grad": n_near_zero,
        "skipped_flips": skipped_flips,
        "per_weight_rel_err": w_stat,
        "near_zero_abs_err": w_abs_stat,
        "fired_frac": fired_frac,
        "conditioning": stats,
        "pass": bool(dot_stat["mean"] is not None and w_stat["mean"] is not None
                     and dot_stat["mean"] < target_rel and w_stat["mean"] < target_rel
                     and (w_abs_stat["max"] is None or w_abs_stat["max"] < atol)),
    }


def e1_forward_vs_dense(tm=15.0, ts=4.0, theta=1.0, t_max=40.0, seed=0):
    """NumPy oracle spike times vs a 200k-point dense simulation reference."""
    from snn import DoubleExpKernel, forward_layer_batch
    k = DoubleExpKernel(tm, ts)
    rng = np.random.default_rng(seed)
    n_in, n_cur, B = 5, 8, 4
    W = rng.standard_normal((n_cur, n_in + 1)) * 0.4
    W[:, -1] = 0.3
    t_in = rng.uniform(1.0, 20.0, size=(n_in, B))
    grid = np.linspace(0.0, t_max, 4001)
    t_fine = np.linspace(0.0, t_max, 200001)

    max_err = 0.0
    n_fired = 0
    for b in range(B):
        t_post, _ = forward_layer_batch(W, t_in[:, b:b + 1], 0.0, theta, t_max, k, grid)
        for j in range(n_cur):
            tf = t_post[j, 0]
            if not np.isfinite(tf):
                continue
            n_fired += 1
            u = np.zeros_like(t_fine)
            for i in range(n_in):
                u += W[j, i] * k.K(t_fine - t_in[i, b])
            u += W[j, -1] * k.K(t_fine)
            idx = np.argmax(u >= theta)
            max_err = max(max_err, abs(tf - t_fine[idx]))
    return {"max_spike_time_err_vs_dense_grid": float(max_err), "n_fired": n_fired}


def e1b_torch_vs_oracle(seed=0, B=8, n_in=6, sizes=(6, 10, 4)):
    """torch/GPU forward vs NumPy oracle forward on identical inputs.

    bias_val must differ from theta: with bias_val == theta the normalized
    kernel (peak 1.0) makes the bias alone graze the threshold at the kernel
    peak, so fired/silent status flips on tiny root-finder differences between
    the two engines (see E5b/E5d). bias_val=1.5 gives a robust crossing on the
    rising edge.
    """
    rng = np.random.default_rng(seed)
    net_t = TTFSNetTorch(list(sizes), seed=seed, w_scale=1.0, bias_val=1.5)
    t_in_np = rng.uniform(1.0, 25.0, size=(n_in, B))
    t_in = torch.tensor(t_in_np, dtype=net_t.dtype, device=net_t.dev)
    t_out_t = net_t.forward(t_in).cpu().numpy()

    from snn import TTFSNet
    net_np = TTFSNet(list(sizes), seed=seed, w_scale=1.0, bias_val=1.5)
    t_out_np = net_np.forward(t_in_np)

    diffs = np.full_like(t_out_t, np.inf)
    both_fired = np.isfinite(t_out_t) & np.isfinite(t_out_np)
    diffs[both_fired] = np.abs(t_out_t[both_fired] - t_out_np[both_fired])
    mismatch_status = (np.isfinite(t_out_t) != np.isfinite(t_out_np))
    max_err = float(diffs[both_fired].max()) if both_fired.any() else 0.0
    n_status_mismatch = int(mismatch_status.sum())
    return {
        "max_spike_time_err_torch_vs_oracle": max_err,
        "status_mismatches": n_status_mismatch,
        "n_both_fired": int(both_fired.sum()),
    }


def e4_train_smoke(seed=1, epochs=60, B=64, n_train=320, n_test=96):
    """Train a 2-layer TTFS net with EXACT gradients only (torch/GPU).

    The previous config (w_scale=0.3, bias_val=0.25, times up to t_max)
    barely fired (fired_frac ~ 0.04), so gradients were ~0 and nothing moved.
    w_scale=0.5, bias_val=0.5 with input times encoded into the early band
    [0.5, 8.0] fires ~60% of neurons with healthy gradient magnitudes, and the
    net reliably learns the 2-class task.
    """
    n_in, n_hid, n_out = 10, 24, 2
    t_max = 40.0
    rng = np.random.default_rng(seed)
    centers = np.zeros((2, n_in))
    centers[0, :4] = 1.8
    centers[1, :4] = 0.2

    def make_data(n):
        y = rng.integers(0, 2, size=n)
        x = rng.normal(loc=centers[y], scale=0.45).clip(0.0, 1.0)
        return x, y

    x, y = make_data(n_train)
    xt, yt = make_data(n_test)

    def encode(xs):
        return 0.5 + 7.5 * (1.0 - xs)

    net = TTFSNetTorch([n_in, n_hid, n_out], t_max=t_max, seed=seed,
                       w_scale=0.5, bias_val=0.5)
    dev = net.dev
    opt = AdamTorch(net.W, lr=0.02, clip=5.0)

    def predict(xs, ys):
        t_in = torch.tensor(encode(xs).T, dtype=net.dtype, device=dev)
        t_out = net.forward(t_in).cpu().numpy()
        pred = np.argmin(np.where(np.isfinite(t_out), t_out, 1e9), axis=0)
        return float(np.mean(pred == ys))

    history = []
    for ep in range(epochs):
        perm = rng.permutation(n_train)
        for s in range(0, n_train, B):
            idx = perm[s:s + B]
            t_in = torch.tensor(encode(x[idx].T), dtype=net.dtype, device=dev)
            yy = torch.tensor(y[idx], device=dev)
            loss, grads, _ = net.loss_and_grads(t_in, yy)
            opt.step(net.W, grads)
        if ep % 5 == 0 or ep == epochs - 1:
            history.append({"epoch": ep, "train_acc": predict(x, y),
                            "test_acc": predict(xt, yt)})
    return {"n_train": n_train, "n_test": n_test, "history": history}


def e5a_grad_scale_vs_depth(seed=0, B=8, n_in=8, n_out=4, max_depth=5):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_out, size=B)
    rows = []
    for depth in range(2, max_depth + 1):
        net, t_in, y_t, _ = _build_smooth_net(depth, seed, n_in=n_in, B=B,
                                              n_out=n_out)
        _, grads, _ = net.loss_and_grads(t_in, y_t)
        first_norm = float(grads[0].norm().item()) if grads[0] is not None else 0.0
        last_norm = float(grads[-1].norm().item()) if grads[-1] is not None else 0.0
        rows.append({
            "depth": depth,
            "sizes": net.sizes,
            "grad_norm_first_layer": first_norm,
            "grad_norm_last_layer": last_norm,
            "ratio_last_to_first": (last_norm / first_norm) if first_norm > 0 else None,
        })
    return rows


def e5b_edge_near_grazing():
    """dt/dw blow-up when u'(t_f) ~ 0 (NumPy oracle, exact root finding)."""
    from snn import DoubleExpKernel, forward_layer_batch, _refine_peak
    tm, ts, theta, t_max = 15.0, 4.0, 1.0, 40.0
    k = DoubleExpKernel(tm, ts)
    t_in = np.array([5.0, 8.0])
    w = {5.0: 0.5, 8.0: 0.5}
    grid = np.linspace(0, t_max, 4001)

    def peak_of_total(bias):
        def total(t):
            v = 0.0
            for ti in t_in:
                v += w[ti] * float(k.K(t - ti))
            return v + bias * float(k.K(t))

        sig = np.zeros_like(grid)
        for ti in t_in:
            sig += w[ti] * k.K(grid - ti)
        imax = int(np.argmax(sig + bias * k.K(grid)))
        lo = grid[max(imax - 1, 0)]
        hi = grid[min(imax + 1, len(grid) - 1)]
        return total(_refine_peak(lo, hi, total))

    def solve_bias(target):
        lo, hi = 0.0, 10.0
        flo = peak_of_total(lo) - target
        fhi = peak_of_total(hi) - target
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            fmid = peak_of_total(mid) - target
            if abs(fmid) < 1e-10:
                break
            if fmid * flo <= 0.0:
                hi, fhi = mid, fmid
            else:
                lo, flo = mid, fmid
        return 0.5 * (lo + hi)

    W = np.zeros((1, 3))
    W[0, 0] = 0.5
    W[0, 1] = 0.5
    W[0, 2] = solve_bias(1.0 + 1e-4)
    t_post, up = forward_layer_batch(W, t_in[:, None], 0.0, theta, t_max, k, grid)
    row = {
        "margin_above_threshold": 1e-4,
        "fired": bool(np.isfinite(t_post[0, 0])),
        "u_prime_at_fire": float(up[0, 0]) if np.isfinite(t_post[0, 0]) else None,
    }
    if np.isfinite(t_post[0, 0]):
        tf = t_post[0, 0]
        row["dt_dw_magnitude"] = float(abs(-float(k.K(tf - t_in[0])) / up[0, 0]))
    W2 = W.copy()
    W2[0, 2] = solve_bias(1.0 + 0.5)
    t_post2, up2 = forward_layer_batch(W2, t_in[:, None], 0.0, theta, t_max, k, grid)
    tf2 = t_post2[0, 0]
    row["normal_fired"] = bool(np.isfinite(tf2))
    row["normal_u_prime"] = float(up2[0, 0])
    row["normal_dt_dw_magnitude"] = float(abs(-float(k.K(tf2 - t_in[0])) / up2[0, 0]))
    return row


def e5c_edge_silent_zero_grad(seed=0):
    """Silent neurons contribute exactly zero gradient (hand-off to SP-02).

    Uses a config with a MIX of fired and silent hidden neurons (random
    standard-normal weights, bias 0.5: some neurons are net-inhibitory and
    never cross threshold). Every silent hidden neuron must have an exactly
    zero gradient row.
    """
    for s in range(seed, seed + 20):
        rng = np.random.default_rng(s)
        net = TTFSNetTorch([6, 10, 3], seed=s, w_scale=1.0, bias_val=0.5)
        t_in = torch.tensor(rng.uniform(1.0, 20.0, size=(6, 8)),
                            dtype=net.dtype, device=net.dev)
        y = torch.tensor(rng.integers(0, 3, size=8), device=net.dev)
        _, grads, _ = net.loss_and_grads(t_in, y)
        t_hidden = net._cache[0][1].cpu().numpy()
        silent = [j for j in range(10) if not np.any(np.isfinite(t_hidden[j]))]
        fired = [j for j in range(10) if j not in silent]
        if silent and fired:
            rows = [{
                "hidden_neuron": j,
                "grad_row_norm": float(grads[0][j].norm().item()),
                "grad_row_abs_sum": float(grads[0][j].abs().sum().item()),
            } for j in silent]
            return {
                "n_fired_hidden": len(fired),
                "n_silent_hidden": len(silent),
                "silent_grad_rows_exactly_zero": bool(all(r["grad_row_norm"] == 0.0
                                                          for r in rows)),
                "silent_neuron_grad_rows": rows,
            }
    raise RuntimeError("no config with mixed fired/silent hidden neurons found")


def e5d_edge_kernel_onset_kink(eps=1e-5):
    """Two-sided central differences are invalid when a firing time t_f sits at
    (or within ~eps*|dt/db| of) an input-arrival time t_in: the clamped kernel
    onset K(d<=0)=0 makes the spike-time map non-differentiable there. The
    analytic gradient (and the one-sided FD away from the kink) stay exact.

    Single neuron, one weak input at t_in, bias 1.5 drives the spike to a
    fixed time t_f ~ 2.3875 (bias-only crossing). Sweeping t_in toward t_f
    shrinks the gap; the two-sided FD of dt/dbias breaks down as the gap
    approaches eps*|dt/db| while the analytic value is unchanged.
    """
    tm, ts = 15.0, 4.0
    theta = 1.0
    dev = device()
    dtype = torch.float64
    bias = 1.5
    w_in = 2.0  # strong: once t_f reaches t_in the input onset pins it there
    grid = torch.linspace(0.0, 40.0, 4001, dtype=dtype, device=dev)

    s = (tm * ts / (tm - ts)) * math.log(tm / ts)
    k_peak = float((math.exp(-s / tm) - math.exp(-s / ts)) / (tm - ts))

    def tf_of(t_arr, bval):
        W = torch.tensor([[w_in, bval]], dtype=dtype, device=dev)
        tp = torch.tensor([[t_arr]], dtype=dtype, device=dev)
        t_post, _ = forward_layer_torch(W, tp, 0.0, theta, grid, tm, ts,
                                        False, k_peak, n_bisect=15, n_newton=8,
                                        peak_tol=0.0)
        return t_post[0, 0].item()

    t_f = tf_of(12.0, bias)  # input long after the spike: t_f = bias crossing
    dt_an = -float(_K(torch.tensor(t_f, dtype=dtype, device=dev), tm, ts,
                      False, k_peak)) / (bias * float(_Kd(
                          torch.tensor(t_f, dtype=dtype, device=dev),
                          tm, ts, False, k_peak)))
    rows = []
    for gap in [1e-1, 1e-3, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 0.0]:
        t_arr = t_f + gap
        t_plus = tf_of(t_arr, bias + eps)
        t_minus = tf_of(t_arr, bias - eps)
        two_sided = (t_plus - t_minus) / (2.0 * eps)
        rows.append({
            "gap_t_in_minus_t_f": float(gap),
            "analytic_dt_db": dt_an,
            "two_sided_fd_dt_db": two_sided,
            "two_sided_rel_err": (abs(dt_an - two_sided) / (abs(dt_an) + 1e-12)),
        })
    return {"note": "two-sided FD breaks once the gap drops below ~eps*|dt/db| (~%.2g); analytic value is exact" % (eps * abs(dt_an)),
            "rows": rows}


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    t0_all = time.perf_counter()
    report = {"experiments": {}, "meta": {"device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu"}}

    print("=== E1: oracle forward vs 200k-point dense simulation ===")
    try:
        e1 = e1_forward_vs_dense()
        print(json.dumps(e1, indent=2))
        report["experiments"]["E1_oracle_vs_dense"] = e1
    except ImportError:
        print("  SKIPPED (NumPy oracle not available)")
        report["experiments"]["E1_oracle_vs_dense"] = {"skip": "oracle removed"}

    print("=== E1b: torch/GPU forward vs NumPy oracle ===")
    try:
        e1b = e1b_torch_vs_oracle()
        print(json.dumps(e1b, indent=2))
        report["experiments"]["E1b_torch_vs_oracle"] = e1b
    except ImportError:
        print("  SKIPPED (NumPy oracle not available)")
        report["experiments"]["E1b_torch_vs_oracle"] = {"skip": "oracle removed"}

    print("=== E2: gradient check (2-layer, torch/GPU) ===")
    e2 = []
    for seed in range(3):
        t0 = time.perf_counter()
        counter = {"n": 0}

        def prog():
            counter["n"] += 1
            if counter["n"] % 15 == 0:
                print(f"    [seed={seed}] {counter['n']} weights checked...",
                      flush=True)

        s = _gradcheck_torch(depth=2, seed=seed, progress=prog)
        e2.append(s)
        d_, w_ = s["dot_rel_err"], s["per_weight_rel_err"]
        print(f"  seed={seed} dot_mean={d_['mean']:.3e} w_mean={w_['mean']:.3e} "
              f"w_max={w_['max']:.3e} checked={s['n_weights_checked']} "
              f"zero={s['n_near_zero_grad']} flips={s['skipped_flips']} "
              f"fired={s['fired_frac']:.2f} "
              f"pass={s['pass']} ({time.perf_counter()-t0:.1f}s)")
    report["experiments"]["E2_gradient_check_2layer"] = e2

    print("=== E3: gradient check (3-layer, torch/GPU) ===")
    e3 = []
    for seed in range(3):
        t0 = time.perf_counter()
        counter = {"n": 0}

        def prog():
            counter["n"] += 1
            if counter["n"] % 15 == 0:
                print(f"    [seed={seed}] {counter['n']} weights checked...",
                      flush=True)

        s = _gradcheck_torch(depth=3, seed=seed, progress=prog)
        e3.append(s)
        d_, w_ = s["dot_rel_err"], s["per_weight_rel_err"]
        print(f"  seed={seed} dot_mean={d_['mean']:.3e} w_mean={w_['mean']:.3e} "
              f"w_max={w_['max']:.3e} checked={s['n_weights_checked']} "
              f"zero={s['n_near_zero_grad']} flips={s['skipped_flips']} "
              f"fired={s['fired_frac']:.2f} "
              f"pass={s['pass']} ({time.perf_counter()-t0:.1f}s)")
    report["experiments"]["E3_gradient_check_3layer"] = e3

    print("=== E4: exact-gradient training smoke test (torch/GPU) ===")
    e4 = e4_train_smoke()
    for h in e4["history"]:
        print(f"  epoch={h['epoch']:3d} train_acc={h['train_acc']:.3f} test_acc={h['test_acc']:.3f}")
    report["experiments"]["E4_train_smoke"] = e4

    print("=== E5a: exact-gradient scale vs depth ===")
    e5a = e5a_grad_scale_vs_depth()
    for r in e5a:
        print(f"  depth={r['depth']} first={r['grad_norm_first_layer']:.3e} "
              f"last={r['grad_norm_last_layer']:.3e}")
    report["experiments"]["E5a_grad_scale_vs_depth"] = e5a

    print("=== E5b: near-grazing dt/dw ===")
    try:
        e5b = e5b_edge_near_grazing()
        print(json.dumps(e5b, indent=2))
        report["experiments"]["E5b_near_grazing"] = e5b
    except ImportError:
        print("  SKIPPED (NumPy oracle not available)")
        report["experiments"]["E5b_near_grazing"] = {"skip": "oracle removed"}

    print("=== E5c: silent neuron zero gradient ===")
    e5c = e5c_edge_silent_zero_grad()
    print(json.dumps(e5c, indent=2))
    report["experiments"]["E5c_silent_zero_grad"] = e5c

    print("=== E5d: kernel-onset kink vs two-sided FD ===")
    e5d = e5d_edge_kernel_onset_kink()
    print(json.dumps(e5d, indent=2))
    report["experiments"]["E5d_kernel_onset_kink"] = e5d

    report["meta"]["wall_time_s"] = time.perf_counter() - t0_all
    path = os.path.join(RESULT_DIR, "sp01-results.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
