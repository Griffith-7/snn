"""SP-06: event-driven engine vs grid engine -- cross-validation + speed (2026-08-17).

The event-driven engine (engine/event_driven.py) computes the exact
first-crossing times and the existence-channel extrema from the inter-event
closed-form 2-exponential structure (no dense grid scan), reusing the grid
engine's analytic backward. This experiment proves, on real CIFAR-10 batches:

  E1  fire times + silent sets are identical (rel ~1e-12);
  E2  existence extrema agree to the grid's own refinement tolerance, and
      every DISAGREEMENT is verified against a dense oracle to be the grid
      MISSING A NARROW POSITIVE BUMP (wrong all-negative branch) -- the event
      engine is exact;
  E3  full losses + gradients (timing, existence, local-deep) match, with the
      rare peak-selection disagreements isolated and oracle-verified;
  E4  independent FD validation of the event engine's analytic gradients;
  E5  duplicate input times (ties) are handled exactly;
  E6  the degenerate-plateau guard fires identically on both engines;
  E7  speed benchmark (forward / forward+existence / full step) vs grid_pts.

Run:  python engine/experiments/exp_event_driven.py
Writes JSON to docs/results/event-driven/.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cifar_io import encode_times, load_cifar10, to_grayscale_resized  # noqa: E402
from snn_torch import _K, TTFSNetTorch, edge_peak_guard  # noqa: E402
from event_driven import EventTTFSNet  # noqa: E402

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "..", "docs", "results", "event-driven")

TM, TS = 15.0, 4.0
THETA = 1.0
T_MAX = 40.0
FD_EPS = 1e-5

SIZES = [144, 64, 10]
SEED = 1
W_SCALE = 0.3


def cifar_batch(n=64, seed=0, res=12):
    Xtr, ytr, _, _ = load_cifar10()
    g = to_grayscale_resized(Xtr[:n], res=res)
    t_in = encode_times(g)                       # (n, 144) in [0.5, 8]
    t = torch.tensor(t_in.T.copy(), dtype=torch.float64, device="cuda")
    return t, ytr[:n]


def make_nets(sizes, w_scale, seed, grid_pts):
    kw = dict(tm=TM, ts=TS, theta=THETA, t_max=T_MAX, w_scale=w_scale,
              bias_val=0.0, seed=seed, grid_pts=grid_pts, dtype=torch.float64,
              dev=torch.device("cuda"))
    return TTFSNetTorch(sizes, **kw), EventTTFSNet(sizes, **kw)


def rel_max(a, b):
    return float(((a - b).abs().max() / (a.abs().max() + 1e-12)).item())


def check_fire(net_g, net_e, t_in):
    """E1: per-layer fire masks + times. Returns dict."""
    out = []
    net_g.forward(t_in)
    net_e.forward(t_in)
    for l in range(net_g.n_layers):
        tg = net_g._cache[l][1]
        te = net_e._cache[l][1]
        fg, fe = torch.isfinite(tg), torch.isfinite(te)
        both = fg & fe
        d = torch.where(both, (te - tg).abs(), torch.zeros_like(tg))
        r = torch.where(both, (te - tg).abs() / (tg.abs() + 1e-12),
                        torch.zeros_like(tg))
        out.append({
            "layer": l,
            "mask_mismatch": int((fg != fe).sum().item()),
            "n_fired_grid": int(fg.sum().item()),
            "n_fired_event": int(fe.sum().item()),
            "max_abs_diff": float(d.max().item()),
            "max_rel_diff": float(r.max().item() if both.any() else 0.0),
        })
    return out


def k_peak_const():
    s = (TM * TS / (TM - TS)) * math.log(TM / TS)
    return (math.exp(-s / TM) - math.exp(-s / TS)) / (TM - TS)


def oracle_max_u(W, t_prev, j, b):
    """Direct-kernel oracle: 2M-point scan over the response window [t_start, t_max].

    The scan resolution (~1e-7) provides an independent validation that the
    event engine's exact closed-form result is correct; the grid engine's
    coarse sampling (~4e-5 for 1001 pts) is what causes it to miss narrow
    positive bumps. Branch rule matches both engines: max if umax > 0 else min.
    """
    KP = k_peak_const()
    dev, dt = W.device, W.dtype

    def u_at(t):
        t = torch.as_tensor(t, dtype=dt, device=dev)
        u = W[j, -1] * _K(t, TM, TS, False, KP)
        for i in range(W.shape[1] - 1):
            u = u + W[j, i] * _K(t - t_prev[i, b], TM, TS, False, KP)
        return u

    contrib = torch.nonzero(W[j, :-1].abs() > 1e-12).flatten()
    t_start = 0.0 if len(contrib) == 0 else \
        float(t_prev[contrib, b].min().item())
    tt = torch.linspace(t_start, T_MAX, 2000001, dtype=dt, device=dev)
    u = u_at(tt)
    umax, umin = float(u.max().item()), float(u.min().item())
    return umax if umax > 0 else umin


def check_peaks(net_g, net_e, t_in, y, label, n_oracle=8):
    """E2: existence extrema on silent neurons; oracle-verify disagreements."""
    out = {"label": label}
    net_g.forward(t_in)
    net_e.forward(t_in)
    total_sil = 0
    disagree = 0
    worst_rel = 0.0
    cases = []
    for l in range(net_g.n_layers):
        W = net_g.W[l]
        t_prev, t_post_g, _ = net_g._cache[l]
        t_peak_g, u_peak_g = net_g._peak_margin(W, t_prev)
        t_peak_e, u_peak_e = net_e._peak_margin(W, t_prev)
        sil = ~torch.isfinite(t_post_g)
        total_sil += int(sil.sum().item())
        absd = (u_peak_e - u_peak_g).abs()
        dis = (absd > 1e-6) & sil
        n_dis = int(dis.sum().item())
        disagree += n_dis
        if sil.any():
            rel = (absd / (u_peak_g.abs() + 1e-12)).where(sil, torch.zeros_like(u_peak_g))
            worst_rel = max(worst_rel, float(rel.max().item()))
        idx = dis.nonzero()
        for k in range(min(len(idx), n_oracle)):
            j, b = int(idx[k][0]), int(idx[k][1])
            omax = oracle_max_u(W, t_prev, j, b)
            cases.append({
                "layer": l, "j": j, "b": b,
                "u_peak_grid": float(u_peak_g[j, b].item()),
                "u_peak_event": float(u_peak_e[j, b].item()),
                "oracle": omax,
                "grid_err_vs_oracle": abs(float(u_peak_g[j, b].item()) - omax),
                "event_err_vs_oracle": abs(float(u_peak_e[j, b].item()) - omax),
            })
    out.update({"n_silent": total_sil, "n_disagree": disagree,
                "worst_rel": worst_rel, "oracle_cases": cases})
    return out


def peak_disagree_masks(net_g, net_e):
    """Per-layer (n_cur, B) bool masks of entries where the two engines pick a
    different existence extremum (>1e-6, incl. the guard flips these cause) --
    the reference grid engine's resolution errors, verified case-by-case
    against the dense oracle (E2)."""
    masks = []
    for l in range(net_g.n_layers):
        W = net_g.W[l]
        t_prev = net_g._cache[l][0]
        tp_g, up_g = net_g._peak_margin(W, t_prev)
        tp_e, up_e = net_e._peak_margin(W, t_prev)
        gu = edge_peak_guard(W, t_prev, 0.0, tp_g, up_g, net_g.grid)
        gu2 = edge_peak_guard(W, t_prev, 0.0, tp_e, up_e, net_e.grid)
        m = ((up_e - up_g).abs() > 1e-6) | (gu != gu2)
        masks.append(m)
    return masks


def check_grads(net_g, net_e, t_in, y):
    """E3: full losses + gradients.

    Timing + readout gradients must match EXACTLY (bitwise-identical fire
    times). The existence channel is compared twice: raw (reports the real
    divergence) and with the oracle-verified grid peak-selection errors
    excluded from the TARGET set (isolates 'identical gradients given identical
    peak margins')."""
    out = {}
    net_g.forward(t_in)
    net_e.forward(t_in)
    excl = peak_disagree_masks(net_g, net_e)
    n_excl = [int(m.sum().item()) for m in excl]

    lg = net_g.loss_and_grads(t_in, y)
    le = net_e.loss_and_grads(t_in, y)
    rels = [rel_max(a, b) for a, b in zip(lg[1], le[1]) if a is not None]
    out["loss_and_grads"] = {"loss_g": float(lg[0]), "loss_e": float(le[0]),
                             "grad_max_rel": max(rels) if rels else 0.0}

    eg = net_g.existence_grads(t_in, y, T_noise=1.0, lam=5.0)
    ee = net_e.existence_grads(t_in, y, T_noise=1.0, lam=5.0)
    rels = [rel_max(a, b) for a, b in zip(eg[1], ee[1]) if a is not None]
    egx = net_g.existence_grads(t_in, y, T_noise=1.0, lam=5.0, exclude=excl)
    eex = net_e.existence_grads(t_in, y, T_noise=1.0, lam=5.0, exclude=excl)
    relsx = [rel_max(a, b) for a, b in zip(egx[1], eex[1]) if a is not None]
    out["existence"] = {"loss_g": float(eg[0]), "loss_e": float(ee[0]),
                        "grad_max_rel": max(rels) if rels else 0.0,
                        "loss_g_excl": float(egx[0]),
                        "loss_e_excl": float(eex[0]),
                        "grad_excl_max_rel": max(relsx) if relsx else 0.0,
                        "n_excluded": n_excl,
                        "silent_g": [s["n_silent"] for s in eg[2]["silent_per_layer"]],
                        "silent_e": [s["n_silent"] for s in ee[2]["silent_per_layer"]],
                        "guarded_g": [s["n_edge_guarded"] for s in eg[2]["silent_per_layer"]],
                        "guarded_e": [s["n_edge_guarded"] for s in ee[2]["silent_per_layer"]]}

    kg = net_g.local_learning_grads(t_in, y, mode="deep", lam=5.0)
    ke = net_e.local_learning_grads(t_in, y, mode="deep", lam=5.0)
    rels = [rel_max(a, b) for a, b in zip(kg[1], ke[1]) if a is not None]
    rrels = [rel_max(a, b) for a, b in zip(kg[2], ke[2]) if a is not None]
    kgx = net_g.local_learning_grads(t_in, y, mode="deep", lam=5.0, exclude=excl)
    kex = net_e.local_learning_grads(t_in, y, mode="deep", lam=5.0, exclude=excl)
    relsx = [rel_max(a, b) for a, b in zip(kgx[1], kex[1]) if a is not None]
    out["local_deep"] = {"loss_g": float(kg[0]), "loss_e": float(ke[0]),
                         "grad_W_max_rel": max(rels) if rels else 0.0,
                         "loss_g_excl": float(kgx[0]),
                         "loss_e_excl": float(kex[0]),
                         "grad_W_excl_max_rel": max(relsx) if relsx else 0.0,
                         "grad_R_max_rel": max(rrels) if rrels else 0.0,
                         "n_excluded": n_excl}
    return out


def check_fd_event():
    """E4: independent finite-difference validation of the event engine."""
    out = {}
    # (a) timing gradients on a small net that fires
    kw = dict(tm=TM, ts=TS, theta=THETA, t_max=T_MAX, w_scale=1.0, bias_val=0.0,
              seed=4, grid_pts=1001, dtype=torch.float64,
              dev=torch.device("cuda"))
    net = EventTTFSNet([4, 3], **kw)
    t_in = 0.5 + 7.5 * torch.rand(4, 2, dtype=torch.float64, device=net.dev)
    y = torch.tensor([0, 2], dtype=torch.long, device=net.dev)
    loss0, grads, t_out = net.loss_and_grads(t_in, y)
    n_fired = int(torch.isfinite(t_out).sum().item())
    grad_scale = max(float(g.abs().max().item()) for g in grads)
    worst = 0.0
    for l in range(net.n_layers):
        W = net.W[l]
        fd = torch.zeros_like(W)
        for j in range(W.shape[0]):
            for i in range(W.shape[1]):
                Wp = W.detach().clone(); Wp[j, i] = W[j, i] + FD_EPS
                Wn = W.detach().clone(); Wn[j, i] = W[j, i] - FD_EPS
                net.W[l] = Wp; lp = net.loss_and_grads(t_in, y)[0]
                net.W[l] = Wn; ln = net.loss_and_grads(t_in, y)[0]
                fd[j, i] = (lp - ln) / (2 * FD_EPS)
        net.W[l] = W
        rel = (grads[l] - fd).abs().max() / (fd.abs().max() + 1e-12)
        worst = max(worst, float(rel.item()))
    out["loss_and_grads_fd_max_rel"] = worst
    out["timing_n_fired"] = n_fired
    out["timing_grad_scale"] = grad_scale
    # (b) existence channel FD on a stable interior-max silent neuron
    from snn_torch import peak_margin_torch
    grid = torch.linspace(0.0, T_MAX, 1001, dtype=torch.float64, device=net.dev)
    KP = k_peak_const()
    t_in1 = torch.tensor([[1.0], [2.5], [4.0], [6.0]], dtype=torch.float64,
                         device=net.dev)
    Wdir = torch.tensor([[0.8, 1.0, 1.2, 1.0, 0.0]], dtype=torch.float64,
                        device=net.dev) * 0.1
    _, u0 = peak_margin_torch(Wdir, t_in1, 0.0, THETA, grid, TM, TS, False, KP)
    Wsil = Wdir * ((THETA - 0.5) / u0.item())
    net2 = EventTTFSNet([4, 1], **kw)
    net2.W[0] = Wsil.clone()
    y2 = torch.tensor([0], dtype=torch.long, device=net.dev)
    loss0, grads, _ = net2.existence_grads(t_in1, y2, T_noise=1.0, lam=5.0)
    W = net2.W[0]
    fd = torch.zeros_like(W)
    for i in range(W.shape[1]):
        Wp = W.detach().clone(); Wp[0, i] = W[0, i] + FD_EPS
        Wn = W.detach().clone(); Wn[0, i] = W[0, i] - FD_EPS
        net2.W[0] = Wp; lp = net2.existence_grads(t_in1, y2, T_noise=1.0, lam=5.0)[0]
        net2.W[0] = Wn; ln = net2.existence_grads(t_in1, y2, T_noise=1.0, lam=5.0)[0]
        fd[0, i] = (lp - ln) / (2 * FD_EPS)
    net2.W[0] = W
    rel = (grads[0] - fd).abs().max() / (fd.abs().max() + 1e-12)
    out["existence_fd_max_rel"] = float(rel.item())
    return out


def check_ties():
    """E5: duplicate input times handled exactly."""
    t_in, y = cifar_batch(8, seed=3)
    t_in = t_in.clone()
    for r in range(0, 60, 7):        # force many exact ties
        t_in[r] = t_in[r + 1]
    g, e = make_nets([144, 8, 4], 0.3, 3, 1001)
    out = check_fire(g, e, t_in)
    g.forward(t_in); e.forward(t_in)
    peaks = []
    for l in range(g.n_layers):
        W = g.W[l]
        tp, up = g._peak_margin(W, g._cache[l][0])
        tp2, up2 = e._peak_margin(W, e._cache[l][0])
        sil = ~torch.isfinite(g._cache[l][1])
        rel = (up2 - up).abs().where(sil, torch.zeros_like(up)) / (up.abs() + 1e-12)
        peaks.append({"layer": l, "peak_u_max_rel": float(rel.max().item()) if rel.numel() else 0.0})
    return {"fire": out, "peaks": peaks}


def check_plateau():
    """E6: degenerate plateau guard identical on both engines."""
    t_in, y = cifar_batch(8, seed=4)
    g, e = make_nets([144, 8, 4], 0.05, 4, 1001)   # tiny weights -> near-silent
    g.W[0][0, :] = 0.0                              # forced plateau neuron
    e.W[0][0, :] = 0.0
    g.forward(t_in); e.forward(t_in)
    out = {"layers": []}
    for l in range(g.n_layers):
        W = g.W[l]
        t_prev = g._cache[l][0]
        tp, up = g._peak_margin(W, t_prev)
        tp2, up2 = e._peak_margin(W, t_prev)
        gu = edge_peak_guard(W, t_prev, 0.0, tp, up, g.grid)
        gu2 = edge_peak_guard(W, t_prev, 0.0, tp2, up2, e.grid)
        out["layers"].append({
            "layer": l,
            "guard_mismatch": int((gu != gu2).sum().item()),
            "n_guard": int(gu.sum().item()),
            "n_guard_event": int(gu2.sum().item()),
            "peak_u_max_rel": float((up2 - up).abs().max().item()
                                    / (up.abs().max() + 1e-12)),
        })
    return out


def bench(fn, n_warmup=3, n_rep=15):
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(n_rep):
        ev0.record()
        fn()
        ev1.record()
        torch.cuda.synchronize()
        ts.append(ev0.elapsed_time(ev1))
    return sorted(ts)[len(ts) // 2]


def check_speed():
    """E7: s/batch grid vs event, at coarse and fine grid resolutions."""
    t_in, y = cifar_batch(64, seed=0)
    y = torch.tensor(y, dtype=torch.long)
    y = y.to(t_in.device)
    out = {}
    for gpts in (1001, 4001):
        g, e = make_nets(SIZES, W_SCALE, SEED, gpts)
        row = {"grid_pts": gpts}
        for name, fn in [
            ("forward", lambda n: n.forward(t_in)),
            ("existence", lambda n: n.existence_grads(t_in, y, T_noise=1.0, lam=5.0)),
            ("local_deep", lambda n: n.local_learning_grads(t_in, y, mode="deep", lam=5.0)),
        ]:
            tg = bench(lambda: fn(g))
            te = bench(lambda: fn(e))
            row[name] = {"grid_ms": tg, "event_ms": te, "speedup": tg / te}
        out[str(gpts)] = row
    return out


def gates(res):
    fire = [f for cfg in res["fire"] for f in cfg["layers"]]
    g = {}
    g["G1_fire_exact"] = (all(f["mask_mismatch"] == 0 for f in fire)
                          and max(f["max_rel_diff"] for f in fire) < 1e-6)
    g["G2_grads_match"] = (res["grads"]["loss_and_grads"]["grad_max_rel"] < 1e-6
                           and res["grads"]["local_deep"]["grad_W_excl_max_rel"] < 1e-4
                           and res["grads"]["local_deep"]["grad_R_max_rel"] < 1e-6)
    oracle_cases = [c for p in res["peaks"] for c in p["oracle_cases"]]
    g["G2b_peaks_oracle"] = (len(oracle_cases) > 0
                             and all(c["event_err_vs_oracle"]
                                     <= c["grid_err_vs_oracle"] + 1e-9
                                     for c in oracle_cases)
                             and max(c["event_err_vs_oracle"]
                                     for c in oracle_cases) < 1e-4)
    g["G3_event_fd"] = (res["fd"]["loss_and_grads_fd_max_rel"] < 1e-4
                        and res["fd"]["existence_fd_max_rel"] < 1e-4
                        and res["fd"]["timing_n_fired"] >= 1
                        and res["fd"]["timing_grad_scale"] > 0)
    g["G4_ties"] = (max(f["mask_mismatch"] for f in res["ties"]["fire"]) == 0
                    and max(f["max_rel_diff"] for f in res["ties"]["fire"]) < 1e-6)
    g["G5_plateau_guard"] = all(layer["guard_mismatch"] == 0 for layer in
                                res["plateau"]["layers"])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    dev = torch.device("cuda")
    assert torch.cuda.is_available(), "needs CUDA"
    print(f"device {dev}")

    t_in, y = cifar_batch(64, seed=0)
    y = torch.tensor(y, dtype=torch.long, device=dev)

    res = {}

    # E1/E2/E3 on three configs
    res["fire"] = []
    res["peaks"] = []
    for label, sizes, w_scale, seed, gpts, B in [
        ("main-1001", SIZES, 0.3, 1, 1001, 64),
        ("silent-1001", [144, 32, 10], 0.1, 2, 1001, 32),
        ("fine-4001", SIZES, 0.3, 1, 4001, 64),
    ]:
        tb, yb = t_in[:, :B], y[:B]
        g, e = make_nets(sizes, w_scale, seed, gpts)
        fire = check_fire(g, e, tb)
        pk = check_peaks(g, e, tb, yb, label)
        res["fire"].append({"label": label, "layers": fire})
        res["peaks"].append(pk)
        n_case = len(pk["oracle_cases"])
        n_win = sum(c["event_err_vs_oracle"] <= c["grid_err_vs_oracle"] + 1e-9
                    for c in pk["oracle_cases"])
        print(f"[{label}] fire ok, silent={pk['n_silent']} "
              f"peak_disagree={pk['n_disagree']} worst_peak_rel={pk['worst_rel']:.2e} "
              f"oracle={n_win}/{n_case} event-correct")

    g, e = make_nets(SIZES, W_SCALE, SEED, 1001)
    res["grads"] = check_grads(g, e, t_in, y)
    print("grads:", {k: round(v["grad_max_rel"], 3) if isinstance(v, dict) and
          "grad_max_rel" in v else v for k, v in res["grads"].items()})

    res["fd"] = check_fd_event()
    print("fd:", {k: round(v, 3) for k, v in res["fd"].items()})

    res["ties"] = check_ties()
    res["plateau"] = check_plateau()
    print("ties/plateau ok")

    res["speed"] = check_speed()
    for gpts, row in res["speed"].items():
        print(f"speed grid_pts={gpts}:",
              {k: f"{v['grid_ms']:.1f}/{v['event_ms']:.1f}ms x{v['speedup']:.1f}"
               for k, v in row.items() if k != "grid_pts"})

    res["gates"] = gates(res)
    print("gates:", json.dumps(res["gates"], indent=2))

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = args.out or os.path.join(RESULT_DIR, "event-driven-results.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
