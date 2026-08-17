"""SP-03 multi-spike integration: gradient check + training comparison.

E1: Compare multi-spike backward (sensitivity_all) against finite differences.
E2: Train with multi-spike backward vs single-spike and compare loss trajectories.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snn_torch import TTFSNetTorch, forward_multispike_layer, forward_multispike_layer_torch, backward_multispike_layer
from losses_torch import latency_cross_entropy
from reset_lif import ResetLIF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cifar_io import load_cifar10, to_grayscale_resized, encode_times, subset


def gradient_check_multispike(sizes=(144, 64, 10), B=4, t_max=40.0,
                               tm=15.0, ts=4.0, theta=1.0, seed=42):
    """E1: Compare multi-spike backward against finite differences."""
    print("[E1] Multi-spike gradient check: sensitivity_all vs finite differences")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = TTFSNetTorch(list(sizes), t_max=t_max, grid_pts=401,
                       dtype=torch.float64, dev=dev, seed=seed)
    Xtr, ytr, _, _ = load_cifar10()
    gtr = to_grayscale_resized(Xtr[:200])
    t_enc = encode_times(gtr)
    t_batch = torch.tensor(t_enc[:B].T, dtype=torch.float64, device=dev)

    net.forward(t_batch)
    layer = 0
    W = net.W[layer].detach().clone().requires_grad_(True)
    t_prev, t_post, up = net._cache[layer]

    t_post_ms, up_ms, t_all_ms, up_all_ms = forward_multispike_layer_torch(
        W, t_prev, net.t_bias, tm, ts, theta, net.k_peak, t_max, net.grid, max_spikes=20)

    n_fires = int(fired_mask.sum().item()) if (fired_mask := torch.isfinite(t_post_ms)).any() else 0
    max_sp = int(torch.isfinite(t_all_ms).sum(dim=2).max().item()) if t_all_ms.numel() > 0 else 0
    print(f"  layer {layer}: {n_fires} fired neurons, max spikes per neuron = {max_sp}")

    lam = torch.zeros_like(t_post_ms)
    lam[fired_mask] = 1.0 / max(n_fires, 1)

    grad_exact, lam_prev = backward_multispike_layer(
        W.detach(), t_prev, net.t_bias, t_all_ms, up_all_ms, lam,
        tm, ts, net.k_peak, t_max, theta)

    eps = 1e-5
    grad_fd = torch.zeros_like(W)
    test_indices = [(0, 0), (0, 1), (1, 0), (32, 10), (63, 144)]
    for idx in test_indices:
        W_plus = W.detach().clone()
        W_plus[idx] += eps
        t_post_p, _, _, _ = forward_multispike_layer_torch(
            W_plus, t_prev, net.t_bias, tm, ts, theta, net.k_peak, t_max, net.grid)
        t_p = torch.where(torch.isfinite(t_post_p), t_post_p,
                          torch.tensor(2.0 * t_max, dtype=t_post_p.dtype, device=dev))
        loss_plus = (lam * t_p).sum().item()

        W_minus = W.detach().clone()
        W_minus[idx] -= eps
        t_post_m, _, _, _ = forward_multispike_layer_torch(
            W_minus, t_prev, net.t_bias, tm, ts, theta, net.k_peak, t_max, net.grid)
        t_m = torch.where(torch.isfinite(t_post_m), t_post_m,
                          torch.tensor(2.0 * t_max, dtype=t_post_m.dtype, device=dev))
        loss_minus = (lam * t_m).sum().item()

        grad_fd[idx] = (loss_plus - loss_minus) / (2 * eps)

    mask = (grad_exact.abs() > 1e-12) | (grad_fd.abs() > 1e-12)
    test_mask = torch.zeros_like(mask)
    for idx in test_indices:
        test_mask[idx] = True
    mask = mask & test_mask
    if mask.any():
        rel_err = torch.abs(grad_exact[mask] - grad_fd[mask]) / torch.maximum(
            torch.abs(grad_exact[mask]), torch.abs(grad_fd[mask]))
        max_rel = float(rel_err.max().item())
        mean_rel = float(rel_err.mean().item())
    else:
        max_rel = 0.0
        mean_rel = 0.0

    status = "PASS" if max_rel < 0.1 else "FAIL"
    print(f"  max_rel={max_rel:.2e}  mean_rel={mean_rel:.2e}  "
          f"|exact|={grad_exact.abs().max():.6f}  "
          f"|fd|={grad_fd.abs().max():.6f}  -> {status}")
    return status


def training_comparison(sizes=(144, 64, 10), n_train=2000, n_test=2000,
                        epochs=3, B=128, t_max=40.0, seed=0):
    """E2: Train multi-spike vs single-spike and compare."""
    print("\n[E2] Training comparison: multi-spike vs single-spike backward")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, ytr, Xte, yte = load_cifar10()
    gtr = to_grayscale_resized(Xtr)
    gte = to_grayscale_resized(Xte)
    t_enc = encode_times(gtr)
    tte_enc = encode_times(gte)
    n_in = t_enc.shape[1]

    ttr_sub, ytr_sub = subset(seed, t_enc, ytr, n_train)
    tte_sub, yte_sub = subset(seed, tte_enc, yte, n_test)

    from optimizers_torch import AdamTorch

    for label, use_multispike in [("single-spike", False), ("multi-spike", True)]:
        print(f"\n  --- {label} ---")
        net = TTFSNetTorch([n_in] + list(sizes[1:]), t_max=t_max, grid_pts=401,
                           dtype=torch.float64, dev=dev, seed=seed)
        opt = AdamTorch(net.W, lr=0.02)

        for ep in range(epochs):
            t0 = time.time()
            perm = np.random.default_rng(seed + ep).permutation(n_train)
            losses = []
            for s in range(0, n_train, B):
                idx = perm[s:s + B]
                tb = torch.tensor(ttr_sub[idx].T, dtype=torch.float64, device=dev)
                yb = torch.tensor(ytr_sub[idx], dtype=torch.long, device=dev)
                if use_multispike:
                    loss, grads, _ = net.loss_and_grads_multispike(tb, yb)
                else:
                    loss, grads, _ = net.loss_and_grads(tb, yb)
                opt.step(net.W, grads)
                losses.append(loss)
            dt = time.time() - t0

            tte_t = torch.tensor(tte_sub.T, dtype=torch.float64, device=dev)
            with torch.no_grad():
                t_out = net.forward(tte_t)
            pred = t_out.argmin(dim=0).cpu().numpy()
            acc = float((pred == yte_sub).mean())
            print(f"    ep {ep}: loss={np.mean(losses):.4f} "
                  f"acc={acc:.3f} time={dt:.1f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", action="store_true", help="Quick gradient check only")
    args = ap.parse_args()

    e1 = gradient_check_multispike()
    if not args.short:
        training_comparison()

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "docs", "results", "sp03-multispike")
    os.makedirs(out_dir, exist_ok=True)
    results = {"E1_gradient_check": e1}
    out_path = os.path.join(out_dir, "sp03-multispike-results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")
