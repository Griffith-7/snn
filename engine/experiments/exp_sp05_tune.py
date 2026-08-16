"""Temp tuning: find the STBP baseline's best config on 15k CIFAR-10 (Gate E strengthening).

Sweeps lr_stbp, slope, T_stbp, hidden for the surrogate baseline, fixed data/split and
engine-identical w_scale/bias/beta/loss. Writes sp05-stbp-tune.json in docs/results/sp05/.
Run:  python engine/experiments/exp_sp05_tune.py   (~1 h)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.exp_sp05 import (cfg, load_data, train_baseline)  # noqa: E402

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "..", "docs", "results", "sp05")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=15000)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    c0 = cfg(epochs=args.epochs, n_train=args.n_train, seed=0)
    ttr, ytr, tte, yte = load_data(c0)
    print(f"data: {c0['n_train']} train / 10000 test, res {c0['res']}x{c0['res']}")

    # staged sweep: first lr x slope at the engine-matching arch/T, then T and hidden.
    grid = []
    for lr_s in (0.005, 0.01, 0.02):
        for sl in (2.0, 6.0):
            grid.append(dict(lr_stbp=lr_s, slope=sl, T_stbp=160, hidden=64))
    results = []
    for i, g in enumerate(grid):
        c = cfg(epochs=args.epochs, n_train=args.n_train, seed=0,
                lr_stbp=g["lr_stbp"], slope=g["slope"], T_stbp=g["T_stbp"])
        c["sizes"][1] = g["hidden"]
        print(f"[tune {i + 1}/{len(grid)}] {g}", flush=True)
        r = train_baseline(c, ttr, ytr, tte, yte)
        r.update(g)
        results.append(r)
        print(f"  -> test {r['test_acc']:.3f} train {r['history'][-1]['train_acc']:.3f} "
              f"({r['train_seconds'] / 60:.1f} min)", flush=True)
        with open(os.path.join(RESULT_DIR, "sp05-stbp-tune.json"), "w") as f:
            json.dump({"config": c, "runs": results}, f, indent=2)

    best = max(results, key=lambda r: r["test_acc"])
    print(f"\nbest lr x slope: lr_stbp={best['lr_stbp']}, slope={best['slope']} "
          f"-> test {best['test_acc']:.3f}")

    # stage 2: T and hidden at the best lr/slope
    g2 = []
    for T_s in (160, 320):
        for h in (64, 128):
            g2.append(dict(lr_stbp=best["lr_stbp"], slope=best["slope"],
                           T_stbp=T_s, hidden=h))
    for i, g in enumerate(g2):
        c = cfg(epochs=args.epochs, n_train=args.n_train, seed=0,
                lr_stbp=g["lr_stbp"], slope=g["slope"], T_stbp=g["T_stbp"])
        c["sizes"][1] = g["hidden"]
        print(f"[tune stage2 {i + 1}/{len(g2)}] {g}", flush=True)
        r = train_baseline(c, ttr, ytr, tte, yte)
        r.update(g)
        results.append(r)
        print(f"  -> test {r['test_acc']:.3f} ({r['train_seconds'] / 60:.1f} min)", flush=True)
        with open(os.path.join(RESULT_DIR, "sp05-stbp-tune.json"), "w") as f:
            json.dump({"config": c, "runs": results}, f, indent=2)

    best = max(results, key=lambda r: r["test_acc"])
    print(f"\n=== best STBP config: lr_stbp={best['lr_stbp']}, slope={best['slope']}, "
          f"T={best['T_stbp']}, hidden={best['hidden']} -> test {best['test_acc']:.3f} ===")


if __name__ == "__main__":
    main()
