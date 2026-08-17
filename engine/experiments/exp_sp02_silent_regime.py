"""SP-02 silent-regime validation (guarded channel) on real CIFAR-10.

Every prior full-scale run used a firing-guaranteeing positive-uniform init
(silent_frac == 0 from epoch 0) or the per-layer-lam std-init fix that reached
0% silent immediately. This experiment runs the guarded existence channel in a
REGIME WHERE SILENCE REALLY EXISTS: standard-normal init scaled DOWN
(w_scale=0.30, bias_val=0.0) on real CIFAR-10, so hidden/output silent
fractions start > 0 and the channel must actually revive neurons.

  S1 initial silence present (hidden_silent_frac > 0 at epoch 0)
  S2 no NaN in any gradient/loss across the whole run
  S3 guard observable: accumulated n_edge_guarded over the run + deterministic
     degenerate-plateau unit tests (S3b: w=0/bias=0 u(t)=0 neuron -> guarded,
     no target, finite grads; S3b2: flippable-earliest-weight sub-case;
     S3b3 control: healthy neuron -> NOT guarded)
  S4 revival: final hidden silent fraction < initial (channel revives)
  S5 learning: test accuracy climbs vs epoch 0
  S6 ablation control: lam=0 (no channel) -> silence persists, accuracy lower

Run:  python engine/experiments/exp_sp02_silent_regime.py
      python engine/experiments/exp_sp02_silent_regime.py --skip-train
      (--skip-train re-reads the recorded training JSON and re-runs only the
       deterministic S3/S3c guard checks + gate assembly)
Writes JSON to docs/results/sp02-silent-regime/.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snn_torch import TTFSNetTorch  # noqa: E402
from optimizers_torch import AdamTorch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cifar_io import load_cifar10, to_grayscale_resized, encode_times, subset  # noqa: E402
from exp_sp05 import cfg, eval_subsample, engine_predict, engine_silent  # noqa: E402

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "..", "docs", "results", "sp02-silent-regime")

T_NOISE = 1.0
LAM_CHANNEL = [5.0, 50.0]          # SP-02 fix: per-layer lam (hidden/output)
LAM_ZERO = 0.0                      # ablation: no existence channel
T_LO, T_HI = 0.5, 8.0


def train_run(c, lam, eval_every=2, max_epochs=12):
    """Guarded-channel training with silent-fraction + NaN + guard bookkeeping."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TTFSNetTorch(list(c["sizes"]), t_max=40.0, w_scale=c["w_scale"],
                       bias_val=c["bias_val"], seed=c["seed"],
                       grid_pts=c["grid_pts"], dev=dev, beta=c["beta"])
    params = net.W + net.R
    opt = AdamTorch(params, lr=c["lr"], clip=5.0)
    B, n_train = c["B"], c["n_train"]
    n_in = c["sizes"][0]
    rng = np.random.default_rng(c["seed"] + 777)
    tte_mon, yte_mon = eval_subsample(c["tte"], c["yte"], 2000, c["seed"])
    ttr_mon, ytr_mon = eval_subsample(c["ttr"], c["ytr"], 1000, c["seed"])
    t0 = time.time()
    history, n_nan, n_guard = [], 0, 0
    silent0 = None
    for ep in range(max_epochs):
        perm = rng.permutation(n_train)
        for s in range(0, n_train, B):
            idx = perm[s:s + B]
            t_in = torch.tensor(c["ttr"][idx].T, dtype=net.dtype, device=dev)
            yy = torch.tensor(c["ytr"][idx], device=dev)
            loss, grads, grads_R, stats = net.local_learning_grads(
                t_in, yy, T_noise=T_NOISE, lam=lam, mode="deep")
            if not math.isfinite(float(loss)):
                n_nan += 1
            for gg in list(grads) + list(grads_R or []):
                n_nan += int(torch.isnan(gg).sum().item())
            for st in stats["silent_per_layer"]:
                n_guard += st["n_edge_guarded"]
            opt.step(params, list(grads) + list(grads_R or []))
        if ep % eval_every == 0 or ep == max_epochs - 1:
            acc_te = engine_predict(net, tte_mon, yte_mon, B, n_in)
            history.append({"epoch": ep, "test_acc": acc_te})
            print(f"  ep {ep}: test {acc_te:.3f}")
    sil_out, sil_hid = engine_silent(net, tte_mon, yte_mon, B)
    net.forward(torch.tensor(ttr_mon[:8].T, dtype=net.dtype, device=dev))
    return dict(train_seconds=time.time() - t0, history=history,
                final_silent_output=float(sil_out),
                final_silent_hidden=(float(sil_hid) if sil_hid is not None
                                     else None),
                n_nan=n_nan, n_edge_guarded=n_guard)


def initial_silence(c):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TTFSNetTorch(list(c["sizes"]), t_max=40.0, w_scale=c["w_scale"],
                       bias_val=c["bias_val"], seed=c["seed"],
                       grid_pts=c["grid_pts"], dev=dev, beta=c["beta"])
    sil_out, sil_hid = engine_silent(net, c["tte"], c["yte"], c["B"])
    return float(sil_out), float(sil_hid) if sil_hid is not None else None


def main():
    c = cfg(epochs=12, n_train=5000, res=12, sizes=(144, 64, 10),
            grid_pts=1001, B=128, lr=2e-2, w_scale=0.30, bias_val=0.0,
            beta=3.0, seed=0, init="std")
    ttr, ytr, tte, yte = load_cifar10()
    ttr, ytr = subset(c["seed"], ttr, ytr, c["n_train"])
    gtr = to_grayscale_resized(ttr, c["res"])
    gte = to_grayscale_resized(tte, c["res"])
    c["ttr"] = encode_times(gtr, T_LO, T_HI).astype(np.float64)
    c["ytr"] = ytr
    c["tte"] = encode_times(gte, T_LO, T_HI).astype(np.float64)
    c["yte"] = yte
    out, g = {}, {}

    # ---- S1: initial silence present ----
    sil0_out, sil0_hid = initial_silence(c)
    out["S1_initial_silence"] = {"output": sil0_out, "hidden": sil0_hid}
    g["S1_silence_present"] = (sil0_hid is not None and sil0_hid > 0.02)

    skip_train = "--skip-train" in sys.argv
    if skip_train:
        prev_path = os.path.join(RESULT_DIR, "sp02-silent-regime-results.json")
        with open(prev_path) as f:
            prev = json.load(f)
        res_on, res_off = prev["S_channel"], prev["S_control"]
        print("[skip-train] reusing recorded S_channel / S_control")
    else:
        print("[run] channel lam=[5,50] (silent regime, std-init)")
        res_on = train_run(c, LAM_CHANNEL)
        print("[run] ablation lam=0 (no channel)")
        res_off = train_run(c, LAM_ZERO)
    out["S_channel"] = res_on
    g["S2_no_nan"] = res_on["n_nan"] == 0
    g["S5_learns"] = (res_on["history"][-1]["test_acc"] >
                      res_on["history"][0]["test_acc"] + 0.02)
    g["S4_revives"] = (sil0_hid is not None and res_on["final_silent_hidden"] is not None
                       and res_on["final_silent_hidden"] < 0.5 * sil0_hid)

    # ---- S6: lam=0 ablation control ----
    out["S_control"] = res_off
    g["S6_control"] = (res_on["final_silent_hidden"] is not None
                       and res_off["final_silent_hidden"] is not None
                       and res_off["final_silent_hidden"] > res_on["final_silent_hidden"])

    # ---- S3b: degenerate-plateau guard unit tests ----
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net0 = TTFSNetTorch([1, 1], seed=0, w_scale=1.0, bias_val=0.0,
                        grid_pts=2001, dev=dev)
    with torch.no_grad():
        net0.W[0].zero_()  # w=0, bias=0 => u(t) = 0 identically
    tb = torch.tensor([[5.0, 6.0, 7.0]], dtype=net0.dtype, device=dev)
    yy = torch.zeros(3, dtype=torch.long, device=dev)
    loss, grads, _, stats = net0.local_learning_grads(
        tb, yy, T_noise=1.0, lam=5.0, mode="deep")
    s0 = stats["silent_per_layer"][0]
    flat_ok = (s0["n_edge_guarded"] >= 1 and s0["n_targeted"] == 0
               and not torch.isnan(grads[0]).any().item())
    out["S3b_plateau"] = {"n_edge_guarded": s0["n_edge_guarded"],
                          "n_targeted": s0["n_targeted"],
                          "grad_finite": not torch.isnan(grads[0]).any().item(),
                          "loss": float(loss)}

    netf = TTFSNetTorch([2, 1], seed=0, w_scale=1.0, bias_val=0.0,
                        grid_pts=2001, dev=dev)
    with torch.no_grad():
        netf.W[0].zero_()
        # two inputs at the SAME time with cancelling tiny weights: u(t) = 0
        # identically (nonzero weights!), so the plateau guard fires AND the
        # earliest contributing weight (1e-10) is at/below the flippable cutoff
        netf.W[0][0, 0] = 1e-10
        netf.W[0][0, 1] = -1e-10
    tb2 = torch.tensor([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0]],
                       dtype=netf.dtype, device=dev)
    yy2 = torch.zeros(3, dtype=torch.long, device=dev)
    loss, grads, _, stats = netf.local_learning_grads(
        tb2, yy2, T_noise=1.0, lam=5.0, mode="deep")
    s0f = stats["silent_per_layer"][0]
    out["S3b2_flippable"] = {"n_edge_guarded": s0f["n_edge_guarded"],
                             "n_targeted": s0f["n_targeted"]}

    neth = TTFSNetTorch([1, 1], seed=0, w_scale=1.0, bias_val=0.2,
                        grid_pts=2001, dev=dev)
    loss, grads, _, stats = neth.local_learning_grads(
        tb, yy, T_noise=1.0, lam=5.0, mode="deep")
    s0h = stats["silent_per_layer"][0]
    out["S3b3_healthy_control"] = {"n_edge_guarded": s0h["n_edge_guarded"],
                                   "n_targeted": s0h["n_targeted"]}
    g["S3_guard"] = (flat_ok and s0f["n_edge_guarded"] >= 1
                     and s0f["n_targeted"] == 0
                     and s0h["n_edge_guarded"] == 0)

    # ---- S3c: guard engages on real data (initial net, several batches) ----
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netr = TTFSNetTorch(list(c["sizes"]), t_max=40.0, w_scale=c["w_scale"],
                        bias_val=c["bias_val"], seed=c["seed"],
                        grid_pts=c["grid_pts"], dev=dev, beta=c["beta"])
    n_guard_real, n_batch = 0, 0
    for s in range(0, 1024, 128):
        tb = torch.tensor(c["ttr"][s:s + 128].T, dtype=netr.dtype, device=dev)
        yy = torch.tensor(c["ytr"][s:s + 128], device=dev)
        _, _, _, stats = netr.local_learning_grads(
            tb, yy, T_noise=1.0, lam=LAM_CHANNEL, mode="deep")
        for st in stats["silent_per_layer"]:
            n_guard_real += st["n_edge_guarded"]
        n_batch += 1
    out["S3c_guard_on_real_data"] = {"batches": n_batch,
                                     "n_edge_guarded": n_guard_real}
    g["S3c_real"] = n_guard_real >= 1

    out["gates"] = g
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, "sp02-silent-regime-results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nS1 initial silent: out={sil0_out:.3f} hid={sil0_hid:.3f}")
    print(f"S2 NaN count: channel={res_on['n_nan']} control={res_off['n_nan']} "
          f"guard-epochs-flagged={res_on['n_edge_guarded']}")
    print(f"S4/S6 final silent hidden: channel={res_on['final_silent_hidden']:.3f} "
          f"control={res_off['final_silent_hidden']:.3f}")
    print(f"S5 test acc: ep0={res_on['history'][0]['test_acc']:.3f} "
          f"final={res_on['history'][-1]['test_acc']:.3f} "
          f"(control {res_off['history'][-1]['test_acc']:.3f})")
    print(f"S3b plateau guard: {s0['n_edge_guarded']} guarded, "
          f"{s0['n_targeted']} targeted, finite={flat_ok}")
    print(f"S3c guard on real data: {n_guard_real} guarded over {n_batch} batches")
    print("\ngates:", json.dumps(g, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
