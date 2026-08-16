"""Phase 5 / Gate E on CIFAR10-DVS: exact engine vs tuned surrogate baseline.

Apples-to-apples extension of exp_sp05 to the neuromorphic dataset, using the
SAME pipeline/architecture/loss/optimizer/encoding; only the data source changes.

Data: CIFAR10-DVS -> per-sample binary spike frames (2,128,128,10) -> single
12x12 intensity frame (ON+OFF counts, normalized) -> TTFS encode, n_in = 144.
This is the best-scoring encoding in an ANN learnability oracle (0.166 vs 0.117
signed, 0.138/0.117/0.100 for 2/3/5 temporal blocks).

Modes: ref (exact engine, SP-01 + SP-02) vs stbp (tuned surrogate baseline).

Run:
  python engine/experiments/exp_sp05_dvs.py --probe
  python engine/experiments/exp_sp05_dvs.py --seed 0 --lam 5,50
Writes JSON to docs/results/sp05/sp05-dvs-results.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exp_sp05 as E  # noqa: E402  (reuses train_engine / train_baseline / cfg)
from cifar_io_dvs import load_dvs  # noqa: E402

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "..", "docs", "results", "sp05")
RESULT_PATH = os.path.join(RESULT_DIR, "sp05-dvs-results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-train", type=int, default=9000)
    ap.add_argument("--res", type=int, default=12)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--init", type=str, default="pos")
    ap.add_argument("--lam", type=str, default=None)
    ap.add_argument("--mode", type=str, default="all",
                    help="which runs: ref, stbp, all (default)")
    ap.add_argument("--slope", type=float, default=None)
    ap.add_argument("--lr-stbp", type=float, default=None)
    ap.add_argument("--T-stbp", type=int, default=None)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    if args.lam is not None:
        vals = [float(x) for x in args.lam.split(",")]
        E.LAM = vals[0] if len(vals) == 1 else vals

    c = E.cfg(epochs=args.epochs, n_train=args.n_train, res=args.res)
    if args.seed is not None:
        c["seed"] = args.seed
    if args.slope is not None:
        c["slope"] = args.slope
    if args.lr_stbp is not None:
        c["lr_stbp"] = args.lr_stbp
    if args.T_stbp is not None:
        c["T_stbp"] = args.T_stbp
    c["init"] = args.init
    c["sizes"] = [args.res * args.res, 64, 10]

    if args.probe:
        ttr, ytr, tte, yte = load_dvs(9000, res=args.res, mode="abs")
        dev = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")
        net = __import__("snn_torch", fromlist=["TTFSNetTorch"]).TTFSNetTorch(
            list(c["sizes"]), t_max=40.0, w_scale=c["w_scale"], bias_val=c["bias_val"],
            seed=c["seed"], grid_pts=c["grid_pts"], dev=dev)
        tb = __import__("torch").tensor(ttr[:c["B"]].T, dtype=net.dtype, device=dev)
        yy = __import__("torch").tensor(ytr[:c["B"]], device=dev)
        import time
        t0 = time.time()
        for _ in range(3):
            net.local_learning_grads(tb, yy, T_noise=E.T_NOISE, lam=E.LAM, mode="ref")
        __import__("torch").cuda.synchronize()
        print(f"[dvs probe] engine per-batch (ref, B={c['B']}, grid={c['grid_pts']}, "
              f"n_in={c['sizes'][0]}): {(time.time() - t0) / 3:.3f} s")
        return

    ttr, ytr, tte, yte = load_dvs(9000, res=args.res, mode="abs")
    if ttr.shape[0] > c["n_train"]:
        ttr, ytr = E.subset(c["seed"], ttr, ytr, c["n_train"])
    print(f"[dvs] data: {ttr.shape[0]} train / {tte.shape[0]} test, "
          f"res {args.res}x{args.res}, inputs {c['sizes'][0]}, arch {list(c['sizes'])}")

    modes = ("ref", "stbp") if args.mode == "all" else (args.mode,)
    runs = []
    for mode in modes:
        print(f"\n=== DVS {mode} (seed {c['seed']}, init {c['init']}, lam {E.LAM}) ===")
        if mode == "stbp":
            r = E.train_baseline(c, ttr, ytr, tte, yte)
        else:
            r = E.train_engine(c, ttr, ytr, tte, yte, mode)
        r["config_seed"] = c["seed"]
        r["dvs_encoding"] = f"abs{args.res}, tblocks=1"
        runs.append(r)

    results = {"config": {k: c[k] for k in
                          ("epochs", "n_train", "res", "sizes", "grid_pts", "B",
                           "lr", "w_scale", "bias_val", "T_stbp", "lr_stbp",
                           "slope", "beta", "seed", "init")},
               "dataset": "CIFAR10-DVS",
               "encoding": f"abs{args.res}, tblocks=1 (ON+OFF counts)",
               "lam": E.LAM,
               "runs": runs}
    os.makedirs(RESULT_DIR, exist_ok=True)
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH) as f:
            prev = json.load(f)
        prev_runs = prev.get("runs", [])
        keep = [r for r in prev_runs
                if not (r.get("config_seed") == c["seed"] and r.get("mode") in modes)]
        results["runs"] = keep + runs
    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print("summary:")
    for run in runs:
        print(f"  {run['mode']:>8}: test_acc {run['test_acc']:.3f}  "
              f"SynOps/test {run['synops_per_test']:.0f}  "
              f"latency {'1 event/neuron' if run['mode'] != 'stbp' else 'T=' + str(run['timesteps'])}  "
              f"{run['train_seconds'] / 60:.1f} min train")


if __name__ == "__main__":
    main()
