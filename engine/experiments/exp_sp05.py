"""Phase 5 (Gate E): exact engine vs surrogate baseline on CIFAR-10.

Apples-to-apples on downsampled grayscale CIFAR-10 (12x12 = 144 inputs, 15k train,
40 epochs default). Same architecture family, kernel, loss, encoding, seeds.
The ONLY difference is the learning rule:
  - engine modes: 'ref' (exact SP-01 + SP-02 existence channel, W^T transport)
                  'deep' (SP-04 per-layer local loss, no W^T transport)
  - baseline: STBP-style surrogate gradient (engine/baseline_stbp.py)

Measures: test accuracy, energy (SynOps), latency (timesteps), wall-clock train.

Run:  python engine/experiments/exp_sp05.py [--epochs N] [--n-train N] [--res R]
      python engine/experiments/exp_sp05.py --probe   # time one batch, then exit
Writes JSON to docs/results/sp05/.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snn_torch import TTFSNetTorch  # noqa: E402
from optimizers_torch import AdamTorch  # noqa: E402
from baseline_stbp import STBPNet  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cifar_io import load_cifar10, to_grayscale_resized, encode_times, subset  # noqa: E402

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "..", "docs", "results", "sp05")

T_NOISE = 1.0
LAM = 5.0
T_LO, T_HI = 0.5, 8.0
SEED = 0


def cfg(epochs=40, n_train=15000, res=12, sizes=(144, 64, 10), grid_pts=1001,
        B=128, lr=2e-2, w_scale=0.4, bias_val=0.2, T_stbp=160, lr_stbp=1e-2,
        slope=2.0, beta=3.0, seed=SEED, init="pos"):
    return dict(epochs=epochs, n_train=n_train, res=res, sizes=list(sizes),
                grid_pts=grid_pts, B=B, lr=lr, w_scale=w_scale, bias_val=bias_val,
                T_stbp=T_stbp, lr_stbp=lr_stbp, slope=slope, beta=beta, seed=seed,
                init=init)


def positive_init(sizes, bias_val, seed, dtype, dev, w_lo=0.05, w_hi=0.4):
    """Toy-style positive-uniform init: guarantees every neuron fires, so the TTFS
    latency readout is well-posed regardless of seed. Applied identically to engine
    and baseline (apples-to-apples). Same shape as the standard-normal init."""
    rng = np.random.default_rng(seed + 12345)
    W = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        w = rng.uniform(w_lo, w_hi, (b, a + 1)).astype(np.float64)
        w[:, -1] = bias_val
        W.append(torch.tensor(w, dtype=dtype, device=dev))
    return W


def load_data(c):
    Xtr, ytr, Xte, yte = load_cifar10()
    Xtr, ytr = subset(c["seed"], Xtr, ytr, c["n_train"])
    gtr = to_grayscale_resized(Xtr, c["res"])
    gte = to_grayscale_resized(Xte, c["res"])
    ttr = encode_times(gtr, T_LO, T_HI)  # (n_train, n_in)
    tte = encode_times(gte, T_LO, T_HI)
    return ttr.astype(np.float64), ytr, tte.astype(np.float64), yte


def engine_predict(net, t, y, B, n_in):
    dev, dtype = net.dev, net.dtype
    pred = np.zeros(y.shape[0], dtype=np.int64)
    for s in range(0, y.shape[0], B):
        tb = torch.tensor(t[s:s + B].T, dtype=dtype, device=dev)
        t_out = net.forward(tb).cpu().numpy()
        pred[s:s + B] = np.argmin(np.where(np.isfinite(t_out), t_out, 1e9), axis=0)
    return float(np.mean(pred == y))


def eval_subsample(t, y, n, seed):
    rng = np.random.default_rng(seed + 999)
    idx = rng.choice(y.shape[0], min(n, y.shape[0]), replace=False)
    return t[idx], y[idx]


def engine_synops(net, t, y, B, n_in):
    """SynOps = sum over layers of events_l * fanin_{l+1}, incl. input layer."""
    dev, dtype = net.dev, net.dtype
    total = 0.0
    n_batch = 0
    for s in range(0, y.shape[0], B):
        tb = torch.tensor(t[s:s + B].T, dtype=dtype, device=dev)
        net.forward(tb)
        total += B * n_in * (net.sizes[1] + 1)  # input layer: one spike per pixel
        for l in range(net.n_layers):
            fired = torch.isfinite(net._cache[l][1]).sum().item()
            fanin = net.sizes[l + 1] + 1 if l + 1 < net.n_layers else net.sizes[l + 1]
            total += fired * fanin
        n_batch += B
    return total / n_batch


def engine_silent(net, t, y, B):
    dev, dtype = net.dev, net.dtype
    tot, hid_sil = 0.0, 0.0
    for s in range(0, y.shape[0], B):
        tb = torch.tensor(t[s:s + B].T, dtype=dtype, device=dev)
        net.forward(tb)
        tp_out = net._cache[-1][1]
        tot += (~torch.isfinite(tp_out)).float().mean().item() * B
        if net.n_layers > 1:
            tp_h = net._cache[0][1]
            hid_sil += (~torch.isfinite(tp_h)).float().mean().item() * B
    n = y.shape[0]
    return tot / n, (hid_sil / n) if net.n_layers > 1 else None


def train_engine(c, ttr, ytr, tte, yte, mode, eval_every=5):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TTFSNetTorch(list(c["sizes"]), t_max=40.0, w_scale=c["w_scale"],
                       bias_val=c["bias_val"], seed=c["seed"],
                       grid_pts=c["grid_pts"], dev=dev, beta=c["beta"])
    if c.get("init", "pos") == "pos":
        net.W = positive_init(c["sizes"], c["bias_val"], c["seed"],
                              net.dtype, dev)
    params = net.W + (net.R if mode == "deep" else [])
    opt = AdamTorch(params, lr=c["lr"], clip=5.0)
    B, n_train = c["B"], ttr.shape[0]
    n_in = c["sizes"][0]
    rng = np.random.default_rng(c["seed"] + 777)
    tte_mon, yte_mon = eval_subsample(tte, yte, 2000, c["seed"])
    ttr_mon, ytr_mon = eval_subsample(ttr, ytr, 1000, c["seed"])
    t0 = time.time()
    history = []
    for ep in range(c["epochs"]):
        perm = rng.permutation(n_train)
        for s in range(0, n_train, B):
            idx = perm[s:s + B]
            t_in = torch.tensor(ttr[idx].T, dtype=net.dtype, device=dev)
            yy = torch.tensor(ytr[idx], device=dev)
            _, grads, grads_R, _ = net.local_learning_grads(t_in, yy, T_noise=T_NOISE,
                                                            lam=LAM, mode=mode)
            gs = list(grads)
            if mode == "deep":
                gs = gs + grads_R
            opt.step(params, gs)
        if ep % eval_every == 0 or ep == c["epochs"] - 1:
            acc_tr = engine_predict(net, ttr_mon, ytr_mon, B, n_in)
            acc_te = engine_predict(net, tte_mon, yte_mon, B, n_in)
            history.append({"epoch": ep, "train_acc": acc_tr, "test_acc": acc_te})
            print(f"[engine {mode}] ep {ep}: train {acc_tr:.3f} test {acc_te:.3f}")
    acc_full = engine_predict(net, tte, yte, B, n_in)
    synops = engine_synops(net, tte, yte, B, n_in)
    sil_out, sil_hid = engine_silent(net, tte, yte, B)
    return dict(mode=mode, history=history, train_seconds=time.time() - t0,
                test_acc=acc_full, synops_per_test=synops,
                silent_output_frac=sil_out, silent_hidden_frac=sil_hid)


def train_baseline(c, ttr, ytr, tte, yte, eval_every=5):
    net = STBPNet(list(c["sizes"]), t_max=40.0, T=c["T_stbp"],
                  w_scale=c["w_scale"], bias_val=c["bias_val"], seed=c["seed"],
                  slope=c["slope"])
    if c.get("init", "pos") == "pos":
        with torch.no_grad():
            for Wp, Wn in zip(positive_init(c["sizes"], c["bias_val"], c["seed"],
                                            net.dtype, net.dev), net.W):
                Wn.copy_(Wp)
    opt = torch.optim.Adam(net.parameters(), lr=c["lr_stbp"])
    B, n_train = c["B"], ttr.shape[0]
    n_in = c["sizes"][0]
    rng = np.random.default_rng(c["seed"] + 777)
    tte_mon, yte_mon = eval_subsample(tte, yte, 2000, c["seed"])
    ttr_mon, ytr_mon = eval_subsample(ttr, ytr, 1000, c["seed"])
    t0 = time.time()
    history = []
    for ep in range(c["epochs"]):
        net.train()
        perm = rng.permutation(n_train)
        for s in range(0, n_train, B):
            idx = perm[s:s + B]
            t_in = torch.tensor(ttr[idx].T, dtype=net.dtype, device=net.dev)
            yy = torch.tensor(ytr[idx], device=net.dev)
            opt.zero_grad()
            t_out = net(t_in)
            loss = net.latency_loss(t_out, yy, beta=c["beta"])
            loss.backward()
            opt.step()
        if ep % eval_every == 0 or ep == c["epochs"] - 1:
            net.eval()
            acc_tr = baseline_predict(net, ttr_mon, ytr_mon, B, n_in)
            acc_te = baseline_predict(net, tte_mon, yte_mon, B, n_in)
            history.append({"epoch": ep, "train_acc": acc_tr, "test_acc": acc_te})
            print(f"[baseline stbp] ep {ep}: train {acc_tr:.3f} test {acc_te:.3f}")
    acc_full = baseline_predict(net, tte, yte, B, n_in)
    synops = baseline_synops(net, tte, yte, B, n_in)
    return dict(mode="stbp", history=history, train_seconds=time.time() - t0,
                test_acc=acc_full, synops_per_test=synops,
                timesteps=c["T_stbp"])


def baseline_predict(net, t, y, B, n_in):
    pred = np.zeros(y.shape[0], dtype=np.int64)
    with torch.no_grad():
        for s in range(0, y.shape[0], B):
            t_in = torch.tensor(t[s:s + B].T, dtype=net.dtype, device=net.dev)
            t_out = net(t_in).cpu().numpy()
            pred[s:s + B] = np.argmin(t_out, axis=0)
    return float(np.mean(pred == y))


def baseline_synops(net, t, y, B, n_in):
    total = 0.0
    n_batch = 0
    with torch.no_grad():
        for s in range(0, y.shape[0], B):
            t_in = torch.tensor(t[s:s + B].T, dtype=net.dtype, device=net.dev)
            net(t_in)
            total += B * n_in * (net.sizes[1] + 1)
            for l, ev in enumerate(net.events_per_layer()):
                fanin = net.sizes[l + 1] + 1 if l + 1 < net.n_layers else net.sizes[l + 1]
                total += ev * fanin
            n_batch += B
    return total / n_batch


def probe(c):
    ttr, ytr, tte, yte = load_data(c)
    n_in = c["sizes"][0]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TTFSNetTorch(list(c["sizes"]), t_max=40.0, w_scale=c["w_scale"],
                       bias_val=c["bias_val"], seed=c["seed"],
                       grid_pts=c["grid_pts"], dev=dev)
    tb = torch.tensor(ttr[:c["B"]].T, dtype=net.dtype, device=dev)
    yy = torch.tensor(ytr[:c["B"]], device=dev)
    t0 = time.time()
    for _ in range(5):
        net.local_learning_grads(tb, yy, T_noise=T_NOISE, lam=LAM, mode="ref")
    torch.cuda.synchronize()
    print(f"engine per-batch (ref, B={c['B']}, grid={c['grid_pts']}): "
          f"{(time.time() - t0) / 5:.3f} s")
    netb = STBPNet(list(c["sizes"]), t_max=40.0, T=c["T_stbp"],
                   w_scale=c["w_scale"], bias_val=c["bias_val"], seed=c["seed"],
                   slope=c["slope"])
    t0 = time.time()
    for _ in range(5):
        netb(tb)
    torch.cuda.synchronize()
    print(f"stbp per-batch (T={c['T_stbp']}, B={c['B']}): {(time.time() - t0) / 5:.3f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-train", type=int, default=15000)
    ap.add_argument("--res", type=int, default=12)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--grid-pts", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--lr-stbp", type=float, default=None)
    ap.add_argument("--T-stbp", type=int, default=None)
    ap.add_argument("--slope", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--init", type=str, default="pos",
                    help="weight init: pos (positive-uniform, robust, default) or norm (standard-normal)")
    ap.add_argument("--lam", type=str, default=None,
                    help="SP-02 existence-channel strength: scalar '5' or per-layer '5,50' "
                         "(default module LAM=5.0; the robust real-data config is '5,50')")
    ap.add_argument("--mode", type=str, default="all",
                    help="which runs: ref, deep, stbp, all (default)")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    c = cfg(epochs=args.epochs, n_train=args.n_train, res=args.res)
    if args.lr is not None:
        c["lr"] = args.lr
    if args.beta is not None:
        c["beta"] = args.beta
    if args.grid_pts is not None:
        c["grid_pts"] = args.grid_pts
    if args.hidden is not None:
        c["sizes"][1] = args.hidden
    if args.lr_stbp is not None:
        c["lr_stbp"] = args.lr_stbp
    if args.T_stbp is not None:
        c["T_stbp"] = args.T_stbp
    if args.slope is not None:
        c["slope"] = args.slope
    if args.seed is not None:
        c["seed"] = args.seed
    if args.init is not None:
        c["init"] = args.init
    if args.lam is not None:
        global LAM
        lam_vals = [float(x) for x in args.lam.split(",")]
        LAM = lam_vals[0] if len(lam_vals) == 1 else lam_vals
    modes = ("ref", "deep", "stbp") if args.mode == "all" else (args.mode,)
    if args.probe:
        probe(c)
        return
    os.makedirs(RESULT_DIR, exist_ok=True)
    ttr, ytr, tte, yte = load_data(c)
    print(f"data: {c['n_train']} train / 10000 test, res {c['res']}x{c['res']}, "
          f"inputs {c['sizes'][0]}, arch {list(c['sizes'])}")
    results = {"config": c, "runs": []}
    for mode in modes:
        if mode == "stbp":
            r = train_baseline(c, ttr, ytr, tte, yte)
        else:
            r = train_engine(c, ttr, ytr, tte, yte, mode)
        r["config_seed"] = c["seed"]
        results["runs"].append(r)
    out_path = os.path.join(RESULT_DIR, "sp05-results.json")
    prev_runs = []
    if os.path.exists(out_path):
        prev = json.load(open(out_path))
        for r in prev["runs"]:
            r_seed = r.get("config_seed", 0)   # pre-multi-seed runs are seed 0
            if r_seed == c["seed"] and r["mode"] in modes:
                continue                        # re-run this (seed, mode) now
            prev_runs.append(r)
    results["runs"] = prev_runs + results["runs"]
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print("summary:")
    for run in results["runs"]:
        print(f"  {run['mode']:>10}: test_acc {run['test_acc']:.3f}  "
              f"SynOps/test {run['synops_per_test']:.0f}  "
              f"latency {'1 event/neuron' if run['mode'] != 'stbp' else 'T=' + str(run['timesteps'])}  "
              f"{run['train_seconds'] / 60:.1f} min train")


if __name__ == "__main__":
    main()
