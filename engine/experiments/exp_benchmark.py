"""Head-to-head benchmark: exact TTFS engine vs EventProp vs surrogate (STBP).

Compares three methods on the SAME model family, loss, encoding, and seeds:
  1. exact   – grid forward + IFT backward (TTFSNetTorch, mode='ref')
  2. event   – event-driven forward + IFT backward (EventTTFSNet)
  3. stbp    – surrogate gradient baseline (STBPNet)

Metrics: test accuracy, SynOps/sample, wall-clock train time, forward/backward ms.

Supported datasets (auto-downloaded where possible):
  cifar10   – CIFAR-10 grayscale downsampled (default, built-in)
  nmnist    – N-MNIST (needs tonic: pip install tonic)
  shd       – Spiking Heidelberg Digits (needs tonic: pip install tonic)

Run:
  python engine/experiments/exp_benchmark.py --dataset cifar10 --epochs 20
  python engine/experiments/exp_benchmark.py --dataset nmnist --epochs 30
  python engine/experiments/exp_benchmark.py --probe   # timing only
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snn_torch import TTFSNetTorch
from event_driven import EventTTFSNet
from optimizers_torch import AdamTorch
from baseline_stbp import STBPNet

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cifar_io import load_cifar10, to_grayscale_resized, encode_times, subset

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "..", "docs", "results", "benchmark")


def _k(tm, ts):
    import math
    s = (tm * ts / (tm - ts)) * math.log(tm / ts)
    return (math.exp(-s / tm) - math.exp(-s / ts)) / (tm - ts)


def load_cifar(c):
    Xtr, ytr, Xte, yte = load_cifar10()
    Xtr, ytr = subset(c["seed"], Xtr, ytr, c["n_train"])
    gtr = to_grayscale_resized(Xtr, c["res"])
    gte = to_grayscale_resized(Xte, c["res"])
    ttr = encode_times(gtr, 0.5, 8.0).astype(np.float64)
    tte = encode_times(gte, 0.5, 8.0).astype(np.float64)
    n_in = c["res"] ** 2
    return ttr, ytr, tte, yte, n_in


def load_nmnist(c):
    try:
        import tonic
        from tonic import transforms
        sensor_size = tonic.datasets.NMNIST.sensor_size
        transform = transforms.Compose([
            transforms.ToFrame(sensor_size=sensor_size, n_time_bins=c.get("T_stbp", 160)),
        ])
        train_ds = tonic.datasets.NMNIST(save_to="./data", train=True, transform=transform)
        test_ds = tonic.datasets.NMNIST(save_to="./data", train=False, transform=transform)
        n_in = sensor_size[0] * sensor_size[1]
        ttr_list, ytr_list, tte_list, yte_list = [], [], [], []
        n_tr = min(c["n_train"], len(train_ds))
        for i in range(n_tr):
            frames, label = train_ds[i]
            ttr_list.append(frames)
            ytr_list.append(label)
        n_te = min(2000, len(test_ds))
        for i in range(n_te):
            frames, label = test_ds[i]
            tte_list.append(frames)
            yte_list.append(label)
        ttr = np.stack(ttr_list).astype(np.float64)
        ytr = np.array(ytr_list)
        tte = np.stack(tte_list).astype(np.float64)
        yte = np.array(yte_list)
        return ttr, ytr, tte, yte, n_in
    except ImportError:
        print("tonic not installed; pip install tonic")
        sys.exit(1)


def load_shd(c):
    try:
        import tonic
        sensor_size = tonic.datasets.SHD.sensor_size
        n_in = sensor_size[0]
        train_ds = tonic.datasets.SHD(save_to="./data", train=True)
        test_ds = tonic.datasets.SHD(save_to="./data", train=False)
        n_tr = min(c["n_train"], len(train_ds))
        n_te = min(2000, len(test_ds))
        ttr_list, ytr_list = [], []
        for i in range(n_tr):
            events, label = train_ds[i]
            ttr_list.append(events)
            ytr_list.append(label)
        tte_list, yte_list = [], []
        for i in range(n_te):
            events, label = test_ds[i]
            tte_list.append(events)
            yte_list.append(label)
        return ttr_list, np.array(ytr_list), tte_list, np.array(yte_list), n_in
    except ImportError:
        print("tonic not installed; pip install tonic")
        sys.exit(1)


def predict_engine(net, t, y, B):
    dev, dtype = net.dev, net.dtype
    pred = np.zeros(y.shape[0], dtype=np.int64)
    for s in range(0, y.shape[0], B):
        tb = torch.tensor(t[s:s + B].T, dtype=dtype, device=dev)
        t_out = net.forward(tb).cpu().numpy()
        pred[s:s + B] = np.argmin(np.where(np.isfinite(t_out), t_out, 1e9), axis=0)
    return float(np.mean(pred == y))


def predict_stbp(net, t, y, B):
    pred = np.zeros(y.shape[0], dtype=np.int64)
    with torch.no_grad():
        for s in range(0, y.shape[0], B):
            tb = torch.tensor(t[s:s + B].T, dtype=net.dtype, device=net.dev)
            t_out = net(tb).cpu().numpy()
            pred[s:s + B] = np.argmin(t_out, axis=0)
    return float(np.mean(pred == y))


def measure_speed_engine(net, t, B, use_saltation=False, n_warmup=2, n_reps=5):
    dev, dtype = net.dev, net.dtype
    tb = torch.tensor(t[:B].T, dtype=dtype, device=dev)
    yy = torch.tensor(np.zeros(B, dtype=np.int64), device=dev)
    for _ in range(n_warmup):
        net.forward(tb)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_reps):
        net.forward(tb)
    torch.cuda.synchronize()
    fwd_ms = (time.time() - t0) / n_reps * 1000
    t0 = time.time()
    for _ in range(n_reps):
        if use_saltation:
            net.loss_and_grads_saltation(tb, yy)
        else:
            net.loss_and_grads(tb, yy)
    torch.cuda.synchronize()
    total_ms = (time.time() - t0) / n_reps * 1000
    return fwd_ms, total_ms - fwd_ms


def measure_speed_stbp(net, t, B, n_warmup=2, n_reps=5):
    dev = net.dev
    dtype = net.dtype
    tb = torch.tensor(t[:B].T, dtype=dtype, device=dev)
    yy = torch.tensor(np.zeros(B, dtype=np.int64), device=dev)
    for _ in range(n_warmup):
        net(tb)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_reps):
        net(tb)
    torch.cuda.synchronize()
    fwd_ms = (time.time() - t0) / n_reps * 1000
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    t0 = time.time()
    for _ in range(n_reps):
        opt.zero_grad()
        t_out = net(tb)
        loss = net.latency_loss(t_out, yy)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    total_ms = (time.time() - t0) / n_reps * 1000
    return fwd_ms, total_ms - fwd_ms


def positive_init(sizes, bias_val, seed, dtype, dev, w_lo=0.05, w_hi=0.4):
    rng = np.random.default_rng(seed + 12345)
    W = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        w = rng.uniform(w_lo, w_hi, (b, a + 1)).astype(np.float64)
        w[:, -1] = bias_val
        W.append(torch.tensor(w, dtype=dtype, device=dev))
    return W


def train_exact(c, ttr, ytr, tte, yte, n_in, engine_cls=TTFSNetTorch, label="exact"):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sizes = [n_in] + c["sizes"][1:]
    net = engine_cls(sizes, t_max=40.0, w_scale=c["w_scale"],
                     bias_val=c["bias_val"], seed=c["seed"],
                     grid_pts=c["grid_pts"], dev=dev, beta=c["beta"])
    net.W = positive_init(sizes, c["bias_val"], c["seed"], net.dtype, dev)
    params = net.W + net.R
    opt = AdamTorch(params, lr=c["lr"], clip=5.0)
    B = c["B"]
    rng = np.random.default_rng(c["seed"] + 777)
    n_eval = min(2000, yte.shape[0])
    idx_eval = rng.choice(yte.shape[0], n_eval, replace=False)
    t0 = time.time()
    for ep in range(c["epochs"]):
        perm = rng.permutation(ytr.shape[0])
        for s in range(0, ytr.shape[0], B):
            idx = perm[s:s + B]
            t_in = torch.tensor(ttr[idx].T, dtype=net.dtype, device=dev)
            yy = torch.tensor(ytr[idx], device=dev)
            _, grads, grads_R, _ = net.local_learning_grads(
                t_in, yy, T_noise=1.0, lam=5.0, mode="deep")
            gs = list(grads) + grads_R
            opt.step(params, gs)
        if ep % 5 == 0 or ep == c["epochs"] - 1:
            acc = predict_engine(net, tte[idx_eval], yte[idx_eval], B)
            print(f"  [{label}] ep {ep:3d}: test {acc:.3f}")
    acc = predict_engine(net, tte, yte, B)
    wall = time.time() - t0
    fwd_ms, bwd_ms = measure_speed_engine(net, ttr, B)
    return dict(label=label, test_acc=acc, train_s=wall,
                fwd_ms=fwd_ms, bwd_ms=bwd_ms)


def train_saltation(c, ttr, ytr, tte, yte, n_in, engine_cls=TTFSNetTorch,
                    label="saltation"):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sizes = [n_in] + c["sizes"][1:]
    net = engine_cls(sizes, t_max=40.0, w_scale=c["w_scale"],
                     bias_val=c["bias_val"], seed=c["seed"],
                     grid_pts=c["grid_pts"], dev=dev, beta=c["beta"])
    net.W = positive_init(sizes, c["bias_val"], c["seed"], net.dtype, dev)
    opt = AdamTorch(net.W, lr=c["lr"], clip=5.0)
    B = c["B"]
    rng = np.random.default_rng(c["seed"] + 777)
    n_eval = min(2000, yte.shape[0])
    idx_eval = rng.choice(yte.shape[0], n_eval, replace=False)
    t0 = time.time()
    for ep in range(c["epochs"]):
        perm = rng.permutation(ytr.shape[0])
        for s in range(0, ytr.shape[0], B):
            idx = perm[s:s + B]
            t_in = torch.tensor(ttr[idx].T, dtype=net.dtype, device=dev)
            yy = torch.tensor(ytr[idx], device=dev)
            loss, grads, _ = net.loss_and_grads_saltation(t_in, yy)
            opt.step(net.W, grads)
        if ep % 5 == 0 or ep == c["epochs"] - 1:
            acc = predict_engine(net, tte[idx_eval], yte[idx_eval], B)
            print(f"  [{label}] ep {ep:3d}: test {acc:.3f}")
    acc = predict_engine(net, tte, yte, B)
    wall = time.time() - t0
    fwd_ms, bwd_ms = measure_speed_engine(net, ttr, B, use_saltation=True)
    return dict(label=label, test_acc=acc, train_s=wall,
                fwd_ms=fwd_ms, bwd_ms=bwd_ms)


def train_stbp(c, ttr, ytr, tte, yte, n_in):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sizes = [n_in] + c["sizes"][1:]
    net = STBPNet(sizes, t_max=40.0, T=c["T_stbp"],
                  w_scale=c["w_scale"], bias_val=c["bias_val"],
                  seed=c["seed"], slope=c["slope"])
    net = net.to(dev)
    if True:
        with torch.no_grad():
            for Wp, Wn in zip(positive_init(sizes, c["bias_val"], c["seed"],
                                            net.dtype, dev), net.W):
                Wn.copy_(Wp)
    opt = torch.optim.Adam(net.parameters(), lr=c["lr_stbp"])
    B = c["B"]
    rng = np.random.default_rng(c["seed"] + 777)
    n_eval = min(2000, yte.shape[0])
    idx_eval = rng.choice(yte.shape[0], n_eval, replace=False)
    t0 = time.time()
    for ep in range(c["epochs"]):
        net.train()
        perm = rng.permutation(ytr.shape[0])
        for s in range(0, ytr.shape[0], B):
            idx = perm[s:s + B]
            t_in = torch.tensor(ttr[idx].T, dtype=net.dtype, device=dev)
            yy = torch.tensor(ytr[idx], device=dev)
            opt.zero_grad()
            t_out = net(t_in)
            loss = net.latency_loss(t_out, yy)
            loss.backward()
            opt.step()
        if ep % 5 == 0 or ep == c["epochs"] - 1:
            net.eval()
            acc = predict_stbp(net, tte[idx_eval], yte[idx_eval], B)
            print(f"  [stbp]   ep {ep:3d}: test {acc:.3f}")
    net.eval()
    acc = predict_stbp(net, tte, yte, B)
    wall = time.time() - t0
    fwd_ms, bwd_ms = measure_speed_stbp(net, ttr, B)
    return dict(label="stbp", test_acc=acc, train_s=wall,
                fwd_ms=fwd_ms, bwd_ms=bwd_ms, timesteps=c["T_stbp"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "nmnist", "shd"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=15000)
    ap.add_argument("--res", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid-pts", type=int, default=1001)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--skip-stbp", action="store_true")
    ap.add_argument("--skip-event", action="store_true")
    ap.add_argument("--skip-exact", action="store_true")
    ap.add_argument("--skip-saltation", action="store_true")
    args = ap.parse_args()

    c = dict(epochs=args.epochs, n_train=args.n_train, res=args.res,
             sizes=[args.res ** 2, args.hidden, 10], grid_pts=args.grid_pts,
             B=args.B, lr=2e-2, w_scale=0.4, bias_val=0.2,
             T_stbp=160, lr_stbp=1e-2, slope=2.0, beta=3.0, seed=args.seed)

    print(f"dataset={args.dataset}  arch={c['sizes']}  epochs={c['epochs']}  "
          f"B={c['B']}  grid={c['grid_pts']}  seed={c['seed']}")
    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    if args.dataset == "cifar10":
        ttr, ytr, tte, yte, n_in = load_cifar(c)
    elif args.dataset == "nmnist":
        ttr, ytr, tte, yte, n_in = load_nmnist(c)
    else:
        ttr, ytr, tte, yte, n_in = load_shd(c)

    c["sizes"][0] = n_in
    print(f"data: {ytr.shape[0]} train / {yte.shape[0]} test, n_in={n_in}")

    results = {"config": c, "dataset": args.dataset, "runs": []}

    if args.probe:
        print("\n--- speed probe ---")
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sizes = [n_in] + c["sizes"][1:]
        net_g = TTFSNetTorch(sizes, t_max=40.0, grid_pts=c["grid_pts"], dev=dev)
        fwd, bwd = measure_speed_engine(net_g, ttr, c["B"])
        print(f"  grid   fwd={fwd:.1f}ms  bwd={bwd:.1f}ms")
        net_e = EventTTFSNet(sizes, t_max=40.0, grid_pts=c["grid_pts"], dev=dev)
        fwd, bwd = measure_speed_engine(net_e, ttr, c["B"])
        print(f"  event  fwd={fwd:.1f}ms  bwd={bwd:.1f}ms")
        net_s = STBPNet(sizes, t_max=40.0, T=c["T_stbp"], dev=dev)
        fwd, bwd = measure_speed_stbp(net_s, ttr, c["B"])
        print(f"  stbp   fwd={fwd:.1f}ms  bwd={bwd:.1f}ms")
        return

    print("\n=== 1. Exact engine (grid + IFT) ===")
    if not args.skip_exact:
        r_exact = train_exact(c, ttr, ytr, tte, yte, n_in,
                              engine_cls=TTFSNetTorch, label="exact")
        results["runs"].append(r_exact)

    if not args.skip_event:
        print("\n=== 2. Event-driven engine ===")
        r_event = train_exact(c, ttr, ytr, tte, yte, n_in,
                              engine_cls=EventTTFSNet, label="event")
        results["runs"].append(r_event)

    if not args.skip_saltation:
        print("\n=== 3. SP-03 saltation backward ===")
        r_salt = train_saltation(c, ttr, ytr, tte, yte, n_in, label="saltation")
        results["runs"].append(r_salt)

    if not args.skip_stbp:
        print("\n=== 4. Surrogate gradient (STBP) ===")
        r_stbp = train_stbp(c, ttr, ytr, tte, yte, n_in)
        results["runs"].append(r_stbp)

    print("\n=== RESULTS ===")
    print(f"{'method':>12}  {'test_acc':>8}  {'fwd_ms':>7}  {'bwd_ms':>7}  {'train_s':>8}")
    print("-" * 50)
    for r in results["runs"]:
        print(f"{r['label']:>12}  {r['test_acc']:>8.3f}  {r['fwd_ms']:>7.1f}  "
              f"{r['bwd_ms']:>7.1f}  {r['train_s']:>8.1f}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = os.path.join(RESULT_DIR, f"benchmark-{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
