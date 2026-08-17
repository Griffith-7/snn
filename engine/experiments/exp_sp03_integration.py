"""SP-03 integration: compare grid backward vs saltation backward.

Experiments:
  E1  Gradient comparison: same net, same forward, compare grid vs saltation
      backward weight gradients on every layer.
  E2  Training comparison: both backward methods on same data, same init,
      compare loss trajectory and accuracy.

Run:
  python engine/experiments/exp_sp03_integration.py

Writes JSON to docs/results/sp03-integration/.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snn_torch import (TTFSNetTorch, backward_layer_torch,
                        backward_layer_saltation)  # noqa: E402
from losses_torch import latency_cross_entropy  # noqa: E402

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "..", "docs", "results", "sp03-integration")


def gradient_comparison(seed=7, sizes=(8, 6, 4), B=4):
    """E1: Compare grid vs saltation backward on the same forward pass."""
    print("[E1] Gradient comparison: grid vs saltation backward")
    net = TTFSNetTorch(sizes, tm=15.0, ts=4.0, theta=1.0, t_max=40.0,
                       w_scale=2.0, seed=seed, grid_pts=1001)
    rng = np.random.default_rng(seed + 100)
    t_in = torch.tensor(rng.uniform(1.0, 15.0, (sizes[0], B)),
                        dtype=net.dtype, device=net.dev)
    y = torch.tensor(rng.integers(0, sizes[-1], B), device=net.dev)

    t_out = net.forward(t_in)
    loss, dL_dt = latency_cross_entropy(t_out, y, net.t_max, net.beta)

    n_fired = torch.isfinite(t_out).sum().item()
    print(f"  loss = {loss:.6f}  fired = {n_fired}/{t_out.numel()}")

    grads_grid = net.backward(dL_dt.clone())
    grads_salt = net.backward_saltation(dL_dt.clone())

    results = {}
    all_ok = True
    for l in range(net.n_layers):
        g_grid = grads_grid[l].detach().cpu().numpy()
        g_salt = grads_salt[l].detach().cpu().numpy()
        denom = np.maximum(np.abs(g_grid), np.abs(g_salt))
        denom = np.where(denom > 1e-12, denom, 1.0)
        rel = np.abs(g_grid - g_salt) / denom
        max_rel = float(rel.max())
        n_nonzero = int((np.abs(g_grid) > 1e-10).sum() + (np.abs(g_salt) > 1e-10).sum())
        ok = max_rel < 1e-2 or np.abs(g_grid - g_salt).max() < 1e-6
        all_ok = all_ok and ok
        results[f"layer_{l}"] = {
            "max_rel_error": max_rel,
            "max_abs_error": float(np.abs(g_grid - g_salt).max()),
            "grid_norm": float(np.sqrt((g_grid**2).sum())),
            "salt_norm": float(np.sqrt((g_salt**2).sum())),
            "nonzero_grads": n_nonzero,
            "pass": ok,
        }
        status = "PASS" if ok else "FAIL"
        print(f"  layer {l}: max_rel={max_rel:.2e}  "
              f"|grid|={results[f'layer_{l}']['grid_norm']:.6f}  "
              f"|salt|={results[f'layer_{l}']['salt_norm']:.6f}  -> {status}")

    status = "PASS" if all_ok else "FAIL"
    print(f"  E1 overall: {status}")
    return all_ok, results


def training_comparison(n_train=256, n_epochs=3, B=16, seed=42):
    """E2: Train same net with grid vs saltation backward, compare loss."""
    print("\n[E2] Training comparison: grid vs saltation backward")

    def make_net():
        return TTFSNetTorch([144, 16, 10], tm=15.0, ts=4.0, theta=1.0,
                            t_max=40.0, w_scale=0.5, seed=seed, grid_pts=1001)

    from cifar_io import load_cifar10, to_grayscale_resized, encode_times
    x_full, y_full, _, _ = load_cifar10()
    x_12 = to_grayscale_resized(x_full, res=12).reshape(-1, 144)

    x_train = x_12[:n_train]
    y_train = y_full[:n_train]

    def encode(x):
        arr = encode_times(x.reshape(x.shape[0], -1),
                           t_lo=0.5, t_hi=36.0).T
        return torch.tensor(arr, dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")

    results = {"grid": [], "saltation": []}
    for method_name, backward_fn in [("grid", "backward"), ("saltation", "backward_saltation")]:
        net = make_net()
        print(f"\n  --- {method_name} ---")
        for epoch in range(n_epochs):
            t0 = time.time()
            perm = np.random.default_rng(seed + epoch).permutation(n_train)
            epoch_loss = 0.0
            epoch_correct = 0
            n_batches = n_train // B
            for mb in range(n_batches):
                idx = perm[mb * B:(mb + 1) * B]
                t_in = encode(x_train[idx])
                y_batch = torch.tensor(y_train[idx], device=net.dev)

                t_out = net.forward(t_in)
                loss, dL_dt = latency_cross_entropy(t_out, y_batch, net.t_max, net.beta)
                grads = net.backward(dL_dt) if backward_fn == "backward" \
                    else net.backward_saltation(dL_dt)
                epoch_loss += loss

                fired = torch.isfinite(t_out)
                for b in range(B):
                    fi = torch.where(fired[:, b])[0]
                    if len(fi) > 0:
                        pred = fi[torch.argmin(t_out[fi, b])]
                        if pred.item() == y_batch[b].item():
                            epoch_correct += 1

                for l in range(net.n_layers):
                    net.W[l] = net.W[l] - 0.01 * grads[l].clamp(-5.0, 5.0)

            elapsed = time.time() - t0
            avg_loss = epoch_loss / max(n_batches, 1)
            acc = epoch_correct / n_train * 100
            results[method_name].append({
                "epoch": epoch + 1, "loss": avg_loss, "acc": acc, "time": elapsed
            })
            print(f"    epoch {epoch+1}: loss={avg_loss:.4f} acc={acc:.1f}% "
                  f"time={elapsed:.1f}s")

    return results


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    out = {}

    ok, e1 = gradient_comparison()
    out["E1_gradient_comparison"] = {"pass": ok, "details": e1}

    e2 = training_comparison()
    out["E2_training_comparison"] = e2

    path = os.path.join(RESULT_DIR, "sp03-integration-results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
