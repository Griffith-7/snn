"""SP-02 experiments: silent-neuron (birth/death) credit via the existence channel.

Run:  python engine/experiments/exp_sp02.py
Writes JSON to docs/results/sp02/ and prints a summary.

E6  far-dead revival toy: a neuron initialized m0 below threshold revives with
    the existence channel; the margin-gradient is bounded below (~1/T), and the
    revived spike time is exact (Q2.2). Control: without the channel it stays
    dead (gradient exactly 0).
E7  output-layer silence handled: training with the channel revives correct-
    class silent outputs and learns the task.
E8  no-regression: (a) SP-01 gradient checks re-run; (b) with 100% firing the
    existence channel contributes exactly zero; (c) existence-channel gradient
    verified vs finite differences (incl. the envelope theorem d(u_peak)/dW=K).
E9  ablation control (the scientific control): identical init, training WITHOUT
    the mechanism fails (dead neurons stay dead, accuracy ~chance); WITH it
    succeeds (silent fraction drops, accuracy rises).
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snn_torch import TTFSNetTorch, forward_layer_torch, peak_margin_torch, device, _K
from losses_torch import latency_cross_entropy
from optimizers_torch import AdamTorch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_sp01 import _gradcheck_torch, _build_smooth_net  # noqa: E402

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "..", "docs", "results", "sp02")

THETA = 1.0


def e6_far_dead_revival(T_noise=1.0, lam=1.0, lr=0.1, m0s=(2, 3, 5, 8),
                        max_steps=900, seed=0):
    """Single neuron, single input spike at t_in=5.0. Init w = theta - m0 < 0
    (silent). The response is a NEGATIVE bump, so the existence channel uses the
    interior minimum as the peak-extremum: u_peak = w*K(d_peak) = w (kernel peak
    1.0), and dL/dw = -(1-p)/T * K(d_peak) = -(1-p)/T. For far-dead (p->0) this
    is bounded below by -1/T -- the key property the naive sigma' surrogate
    draft lacked. m0 = 1 (w = 0) is excluded: a zero-weight neuron carries no
    signal, so its existence gradient is 0 by construction."""
    t_in_val = 5.0
    dev = device()
    dtype = torch.float64
    d_peak = (15.0 * 4.0 / (15.0 - 4.0)) * math.log(15.0 / 4.0)

    def expected_first_crossing(w):
        """First-crossing time of u(t)=w*K(t-t_in_val)=theta for w>theta.
        On the rising edge K is monotone in [0, d_peak]; bisect for
        K(d) = theta/w. Returns None if w <= theta (no crossing)."""
        if w <= THETA:
            return None
        target = THETA / w
        lo, hi = 0.0, d_peak
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            km = float(_K(torch.tensor(mid, dtype=dtype, device=dev), 15.0, 4.0,
                          False, net.k_peak).item())
            if km < target:
                lo = mid
            else:
                hi = mid
        return t_in_val + 0.5 * (lo + hi)

    rows = []
    for m0 in m0s:
        net = TTFSNetTorch([1, 1], seed=seed, w_scale=1.0, bias_val=0.0,
                           grid_pts=2001)
        W = net.W[0]
        W[0, 0] = THETA - m0
        t_in = torch.tensor([[t_in_val]], dtype=dtype, device=dev)
        y = torch.tensor([0], device=dev)
        _, grads, _ = net.existence_grads(t_in, y, T_noise=T_noise, lam=lam)
        g0 = float(grads[0][0, 0].item())

        revived_at = None
        tf = None
        w_final = None
        for step in range(max_steps):
            _, grads, _ = net.existence_grads(t_in, y, T_noise=T_noise, lam=lam)
            W[0, 0] = float(W[0, 0]) - lr * float(grads[0][0, 0])
            t_out = net.forward(t_in)
            if torch.isfinite(t_out[0, 0]):
                revived_at = step + 1
                tf = float(t_out[0, 0])
                w_final = float(W[0, 0])
                # existence channel should now be shut off (p->1)
                _, grads2, stats2 = net.existence_grads(t_in, y, T_noise=T_noise, lam=lam)
                g_final = float(grads2[0][0, 0].item())
                break
        exp_tf = expected_first_crossing(w_final) if w_final is not None else None
        rows.append({
            "margin_below_threshold_m0": m0,
            "initial_gradient_magnitude": abs(g0),
            "bounded_below_0_7": bool(abs(g0) >= 0.7),
            "revived": revived_at is not None,
            "revival_steps": revived_at,
            "spike_time": tf,
            "expected_first_crossing": exp_tf,
            "spike_time_err": abs(tf - exp_tf) if (tf is not None and exp_tf is not None) else None,
            "existence_grad_after_revival": g_final if revived_at is not None else None,
        })

    # control: same neuron, channel disabled (lam=0) -> exactly zero gradient
    net = TTFSNetTorch([1, 1], seed=seed, w_scale=1.0, bias_val=0.0, grid_pts=2001)
    W = net.W[0]
    W[0, 0] = THETA - 5.0
    t_in = torch.tensor([[t_in_val]], dtype=dtype, device=dev)
    y = torch.tensor([0], device=dev)
    w_before = float(W[0, 0])
    for _ in range(max_steps):
        _, grads, _ = net.existence_grads(t_in, y, T_noise=T_noise, lam=0.0)
        if float(grads[0][0, 0].item()) != 0.0:
            break
        W[0, 0] = float(W[0, 0]) - lr * float(grads[0][0, 0])
    control = {
        "control_channel_off": "true",
        "w_before": w_before,
        "w_after": float(W[0, 0]),
        "moved": abs(float(W[0, 0]) - w_before) > 1e-15,
        "fired_without_channel": bool(torch.isfinite(net.forward(t_in)[0, 0])),
    }
    return {"T_noise": T_noise, "lam": lam, "lr": lr, "rows": rows, "control": control}


def _make_class_task(n_train=320, n_test=96, seed=1):
    rng = np.random.default_rng(seed)
    centers = np.zeros((2, 10))
    centers[0, :4] = 1.8
    centers[1, :4] = 0.2

    def make_data(n):
        y = rng.integers(0, 2, size=n)
        x = rng.normal(loc=centers[y], scale=0.45).clip(0.0, 1.0)
        return x, y

    x, y = make_data(n_train)
    xt, yt = make_data(n_test)
    return x, y, xt, yt


def _train(net, x, y, xt, yt, epochs, B, lr, use_existence, T_noise, lam,
           record_every=5, seed=1):
    rng = np.random.default_rng(seed + 777)
    dev = net.dev
    opt = AdamTorch(net.W, lr=lr, clip=5.0)
    n_train = x.shape[0]

    def encode(xs):
        return 0.5 + 7.5 * (1.0 - xs)

    def predict(xs, ys):
        t_in = torch.tensor(encode(xs).T, dtype=net.dtype, device=dev)
        t_out = net.forward(t_in).cpu().numpy()
        pred = np.argmin(np.where(np.isfinite(t_out), t_out, 1e9), axis=0)
        return float(np.mean(pred == ys))

    def sil_stats(xs, ys):
        t_in = torch.tensor(encode(xs).T, dtype=net.dtype, device=dev)
        yy = torch.tensor(ys, device=dev)
        net.forward(t_in)
        t_out = net._cache[-1][1]
        n_out = t_out.shape[0]
        fired = torch.isfinite(t_out)
        sil = [float((~fired).float().mean().item())]
        n = torch.zeros(n_out, dtype=torch.long, device=dev)
        onehot = torch.zeros((n_out, t_in.shape[1]), dtype=torch.bool, device=dev)
        onehot[yy, torch.arange(t_in.shape[1], device=dev)] = True
        wrong_sil = float((~fired & ~onehot).sum().item() / (~onehot).sum().item())
        correct_sil = float((~fired & onehot).sum().item() / onehot.sum().item())
        hid = []
        for l in range(net.n_layers - 1):
            tp = net._cache[l][1]
            hid.append(float((~torch.isfinite(tp)).float().mean().item()))
        return sil, wrong_sil, correct_sil, hid

    history = []
    for ep in range(epochs):
        perm = rng.permutation(n_train)
        for s in range(0, n_train, B):
            idx = perm[s:s + B]
            t_in = torch.tensor(encode(x[idx].T), dtype=net.dtype, device=dev)
            yy = torch.tensor(y[idx], device=dev)
            if use_existence:
                loss, grads, _ = net.existence_grads(t_in, yy, T_noise=T_noise, lam=lam)
            else:
                loss, grads, _ = net.loss_and_grads(t_in, yy)
            opt.step(net.W, grads)
        if ep % record_every == 0 or ep == epochs - 1:
            _, w_sil, c_sil, hid = sil_stats(x, y)
            history.append({
                "epoch": ep,
                "train_acc": predict(x, y),
                "test_acc": predict(xt, yt),
                "output_silent_frac": w_sil,
                "correct_output_silent_frac": c_sil,
                "hidden_silent_frac": max(hid) if hid else None,
            })
    return history


def e7_output_layer_silence(epochs=30, B=64, lr=0.02, T_noise=1.0, lam=5.0):
    """Training WITH the existence channel. The config (small weights/bias) has
    many silent outputs at init; the correct-class silent outputs must revive
    for the task to be learnable.

    lam: the channel competes with the timing loss's push-DOWN on wrong-class
    outputs (each output is wrong for ~half the batch); a stronger lam lets the
    correct-class push win the revival race. Sensitivity (lam=1 -> stuck at
    chance) is reported in the results doc."""
    x, y, xt, yt = _make_class_task()
    for seed in range(8):
        net = TTFSNetTorch([10, 24, 2], seed=seed, w_scale=0.4, bias_val=0.15,
                           grid_pts=2001)
        # quick initial correct-output silence probe
        dev = net.dev
        t_in = torch.tensor((0.5 + 7.5 * (1.0 - x[:64].T)), dtype=net.dtype, device=dev)
        yy = torch.tensor(y[:64], device=dev)
        net.forward(t_in)
        t_out = net._cache[-1][1]
        fired = torch.isfinite(t_out)
        onehot = torch.zeros((2, 64), dtype=torch.bool, device=dev)
        onehot[yy, torch.arange(64, device=dev)] = True
        c_sil0 = float((~fired & onehot).sum().item() / onehot.sum().item())
        if c_sil0 >= 0.4:
            break
    history = _train(net, x, y, xt, yt, epochs, B, lr, use_existence=True,
                     T_noise=T_noise, lam=lam)
    return {
        "seed": seed,
        "init_correct_output_silent_frac": c_sil0,
        "epochs": epochs,
        "history": history,
        "final_test_acc": history[-1]["test_acc"],
        "final_correct_output_silent_frac": history[-1]["correct_output_silent_frac"],
    }


def e8_no_regression(T_noise=1.0, lam=1.0, eps=1e-5):
    """(a) Re-run SP-01 gradient checks (2- and 3-layer) to prove no regression.
    (b) With 100% firing the existence channel contributes exactly zero.
    (c) Existence-channel gradient check vs finite differences, including the
    envelope theorem d(u_peak)/dW_ji = K(t_peak - t_i) for a silent neuron.
    """
    out = {"SP01_gradchecks": [], "zero_contribution": None, "existence_gradcheck": None}
    for depth in (2, 3):
        for seed in range(2):
            out["SP01_gradchecks"].append(_gradcheck_torch(depth, seed))

    # (b) 100%-fired config -> existence channel adds exactly zero
    for attempt in range(30):
        net, t_in, y, stats = _build_smooth_net(2, 0, B=8)
        if stats["fired_frac"] == 1.0:
            break
    loss_a, grads_a, _ = net.loss_and_grads(t_in, y)
    loss_b, grads_b, stats_b = net.existence_grads(t_in, y, T_noise=T_noise, lam=lam)
    max_diff = max(float((ga - gb).abs().max().item())
                   for ga, gb in zip(grads_a, grads_b))
    out["zero_contribution"] = {
        "fired_frac": stats["fired_frac"],
        "loss_identical": abs(loss_a - loss_b) < 1e-12,
        "max_grad_abs_diff": max_diff,
        "targeted_silent": sum(s["n_targeted"] for s in stats_b["silent_per_layer"]),
    }

    # (c) existence-channel gradient check on a silent neuron (well-conditioned:
    # t_peak ~10 far from input times 1,2 and bias time 0 -> no kernel-onset kink)
    dev = device()
    dtype = torch.float64
    net = TTFSNetTorch([2, 1], seed=0, w_scale=1.0, bias_val=0.0, grid_pts=2001)
    W = net.W[0]
    W[0, 0] = 0.5
    W[0, 1] = 0.3
    W[0, 2] = 0.05
    t_in = torch.tensor([[1.0], [2.0]], dtype=dtype, device=dev)
    y = torch.tensor([0], device=dev)
    _, grads, _ = net.existence_grads(t_in, y, T_noise=T_noise, lam=lam)
    t_peak, u_peak = peak_margin_torch(W, t_in, 0.0, THETA, net.grid, net.tm,
                                       net.ts, net._alpha, net.k_peak)
    t_peak0 = float(t_peak[0, 0])
    u_peak0 = float(u_peak[0, 0])

    def l_exist(weights):
        Wc = W.clone()
        Wc[0, :] = torch.tensor(weights, dtype=dtype, device=dev)
        _, up = peak_margin_torch(Wc, t_in, 0.0, THETA, net.grid, net.tm,
                                  net.ts, net._alpha, net.k_peak)
        p = torch.sigmoid((up[0, 0] - THETA) / T_noise)
        return -float(torch.log(p.clamp(min=1e-12)).item())

    n_in = 2
    w0 = [float(W[0, 0].item()), float(W[0, 1].item()), float(W[0, 2].item())]
    g_an = [float(grads[0][0, i].item()) for i in range(3)]
    g_fd = []
    env_fd = []
    env_an = []
    for i in range(3):
        wp = list(w0)
        wm = list(w0)
        wp[i] += eps
        wm[i] -= eps
        g_fd.append((l_exist(wp) - l_exist(wm)) / (2.0 * eps))
        up_p = l_exist(wp)  # placeholder
        # envelope theorem: d(u_peak)/dw_i vs FD of u_peak
        up_plus = peak_margin_torch(W.clone(), t_in, 0.0, THETA, net.grid, net.tm,
                                    net.ts, net._alpha, net.k_peak)[1][0, 0]
        # recompute with perturbed weight
        Wc = W.clone()
        Wc[0, i] += eps
        up_plus = peak_margin_torch(Wc, t_in, 0.0, THETA, net.grid, net.tm,
                                    net.ts, net._alpha, net.k_peak)[1][0, 0]
        Wc = W.clone()
        Wc[0, i] -= eps
        up_minus = peak_margin_torch(Wc, t_in, 0.0, THETA, net.grid, net.tm,
                                     net.ts, net._alpha, net.k_peak)[1][0, 0]
        fd = float((up_plus - up_minus).item() / (2.0 * eps))
        t_peak_i = float(t_peak[0, 0])
        if i < n_in:
            an = float(_K(torch.tensor(t_peak_i - float(t_in[i, 0]), dtype=dtype,
                                       device=dev), net.tm, net.ts, net._alpha,
                          net.k_peak).item())
        else:
            an = float(_K(torch.tensor(t_peak_i, dtype=dtype, device=dev),
                          net.tm, net.ts, net._alpha, net.k_peak).item())
        env_fd.append(fd)
        env_an.append(an)

    rel = [abs(a - f) / (abs(f) + 1e-12) for a, f in zip(g_an, g_fd)]
    env_rel = [abs(a - f) / (abs(f) + 1e-12) for a, f in zip(env_an, env_fd)]
    out["existence_gradcheck"] = {
        "silent_neuron": True,
        "u_peak": u_peak0,
        "t_peak": t_peak0,
        "weights": w0,
        "analytic_exist_grad": g_an,
        "fd_exist_grad": g_fd,
        "rel_err_exist_grad": rel,
        "pass_exist_grad": max(rel) < 1e-4,
        "envelope_dudw_analytic": env_an,
        "envelope_dudw_fd": env_fd,
        "rel_err_envelope": env_rel,
        "pass_envelope": max(env_rel) < 1e-4,
    }
    return out


def e9_ablation_control(epochs=40, B=64, lr=0.02, T_noise=1.0, lam=5.0):
    """Scientific control: identical init, the ONLY difference is the existence
    channel. WITHOUT it the dead neurons stay dead and accuracy stays ~chance;
    WITH it neurons revive and the task is learned."""
    x, y, xt, yt = _make_class_task(seed=3)
    # find a seed with substantial initial silence (dead neurons to revive)
    for seed in range(8):
        net0 = TTFSNetTorch([10, 24, 2], seed=seed, w_scale=0.3, bias_val=0.2,
                            grid_pts=2001)
        dev = net0.dev
        t_in = torch.tensor((0.5 + 7.5 * (1.0 - x[:64].T)), dtype=net0.dtype, device=dev)
        net0.forward(t_in)
        hid_sil = float((~torch.isfinite(net0._cache[0][1])).float().mean().item())
        out_sil = float((~torch.isfinite(net0._cache[1][1])).float().mean().item())
        if hid_sil >= 0.3 and out_sil >= 0.3:
            break
    init_silence = {"hidden": hid_sil, "output": out_sil}

    netA = TTFSNetTorch([10, 24, 2], seed=seed, w_scale=0.3, bias_val=0.2, grid_pts=2001)
    histA = _train(netA, x, y, xt, yt, epochs, B, lr, use_existence=False,
                   T_noise=T_noise, lam=lam)
    netB = TTFSNetTorch([10, 24, 2], seed=seed, w_scale=0.3, bias_val=0.2, grid_pts=2001)
    histB = _train(netB, x, y, xt, yt, epochs, B, lr, use_existence=True,
                   T_noise=T_noise, lam=lam)

    finA = histA[-1]
    finB = histB[-1]
    return {
        "seed": seed,
        "init_silence": init_silence,
        "without_channel": {
            "final_train_acc": finA["train_acc"],
            "final_test_acc": finA["test_acc"],
            "final_hidden_silent_frac": finA["hidden_silent_frac"],
            "history": histA,
        },
        "with_channel": {
            "final_train_acc": finB["train_acc"],
            "final_test_acc": finB["test_acc"],
            "final_hidden_silent_frac": finB["hidden_silent_frac"],
            "history": histB,
        },
        "pass": bool(finB["test_acc"] > finA["test_acc"] + 0.15
                     and finB["hidden_silent_frac"] < finA["hidden_silent_frac"]),
    }


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    t0_all = time.perf_counter()
    report = {"experiments": {}, "meta": {"device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu"}}

    print("=== E6: far-dead revival toy (existence channel) ===")
    e6 = e6_far_dead_revival()
    for r in e6["rows"]:
        print(f"  m0={r['margin_below_threshold_m0']} |g0|={r['initial_gradient_magnitude']:.4f} "
              f"revived={r['revived']} steps={r['revival_steps']} "
              f"tf={r['spike_time']:.4f} (expect {r['expected_first_crossing']:.4f}) "
              f"err={r['spike_time_err']:.2e}")
    print(f"  control: {json.dumps(e6['control'])}")
    report["experiments"]["E6_far_dead_revival"] = e6

    print("=== E7: output-layer silence handled (training with channel) ===")
    e7 = e7_output_layer_silence()
    print(f"  seed={e7['seed']} init correct-output silent={e7['init_correct_output_silent_frac']:.3f}")
    for h in e7["history"]:
        print(f"  epoch={h['epoch']:3d} train_acc={h['train_acc']:.3f} test_acc={h['test_acc']:.3f} "
              f"correct_sil={h['correct_output_silent_frac']:.3f} hidden_sil={h['hidden_silent_frac']:.3f}")
    report["experiments"]["E7_output_layer_silence"] = e7

    print("=== E8: no-regression + existence gradient check ===")
    e8 = e8_no_regression()
    for s in e8["SP01_gradchecks"]:
        d_, w_ = s["dot_rel_err"], s["per_weight_rel_err"]
        print(f"  SP01 gradcheck depth={s['depth']} seed={s['seed']} "
              f"dot_mean={d_['mean']:.3e} w_mean={w_['mean']:.3e} pass={s['pass']}")
    zc = e8["zero_contribution"]
    print(f"  zero-contribution: fired_frac={zc['fired_frac']} loss_identical={zc['loss_identical']} "
          f"max_grad_diff={zc['max_grad_abs_diff']:.2e}")
    ec = e8["existence_gradcheck"]
    print(f"  existence gradcheck: u_peak={ec['u_peak']:.4f} t_peak={ec['t_peak']:.4f} "
          f"pass_exist_grad={ec['pass_exist_grad']} pass_envelope={ec['pass_envelope']}")
    print(f"    rel_err_exist_grad={[f'{x:.2e}' for x in ec['rel_err_exist_grad']]}")
    print(f"    rel_err_envelope  ={[f'{x:.2e}' for x in ec['rel_err_envelope']]}")
    report["experiments"]["E8_no_regression"] = e8

    print("=== E9: ablation control (without vs with channel) ===")
    e9 = e9_ablation_control()
    print(f"  seed={e9['seed']} init silence (hidden, output)="
          f"({e9['init_silence']['hidden']:.3f},{e9['init_silence']['output']:.3f})")
    print("  WITHOUT channel:")
    for h in e9["without_channel"]["history"]:
        print(f"    epoch={h['epoch']:3d} train_acc={h['train_acc']:.3f} test_acc={h['test_acc']:.3f} "
              f"hidden_sil={h['hidden_silent_frac']:.3f}")
    print("  WITH channel:")
    for h in e9["with_channel"]["history"]:
        print(f"    epoch={h['epoch']:3d} train_acc={h['train_acc']:.3f} test_acc={h['test_acc']:.3f} "
              f"hidden_sil={h['hidden_silent_frac']:.3f}")
    print(f"  pass={e9['pass']}")
    report["experiments"]["E9_ablation_control"] = e9

    report["meta"]["wall_time_s"] = time.perf_counter() - t0_all
    path = os.path.join(RESULT_DIR, "sp02-results.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
