"""SP-02 rigor re-validation (Gate B follow-up, 2026-08-17).

Check A -- escape-noise survival-integral gradient vs the peak-margin channel.
  The SP-02 channel p = sigmoid((u_peak - theta)/T) is the saddle-point
  approximation of the exact escape-noise P_fire = 1 - exp(-int rho dt)
  (escape_rate.py). We measure, exactly:
    * P_fire / S / Lambda and the exact gradient dL/dW by quadrature (FD
      validated to ~1e-9, float64);
    * the channel gradient (identical to snn_torch.existence_grads, lam=1,B=1);
    * the exact decomposition (logistic rate), for the dominant input:
        g_esc / g_chan = f(S) * C,   f(S) = -S log S/(1-S),   C = kernel term
      The S-dependence is EXACTLY f(S) (C is rho0-independent, verified across
      a 1e-3..1.0 rate sweep); C -> 1 as the escape spike narrows (T_esc -> 0,
      verified). So: far-dead AND narrow-spike => channel IS the exact expected
      gradient; at finite temperature the channel over-pushes by 1/(f(S) C).

Check B -- envelope theorem d(u_peak)/dW = K(t_peak - t_in) vs brute-force FD.
  Verified within a branch (interior max AND strictly-all-negative interior
  min): rel err ~1e-9. The all-negative branch's extremum selection is NON-
  SMOOTH across u_max = 0 (perturbing a neuron with u_max exactly 0 flips the
  branch, rel err ~1e5) -- a documented discontinuity of the channel.

Check C -- degenerate plateau guard (edge_peak_guard).
  All-zero weights => u(t) = 0: channel gradient ~0 (deadlock) while the exact
  escape gradient is O(1) (the escape model revives it). edge_peak_guard must
  flag it so the channel never injects a dead signal.

Run:  python engine/experiments/exp_sp02_rigor.py
Writes JSON to docs/results/sp02-rigor/.
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from escape_rate import (  # noqa: E402
    compare_channel_vs_escape,
    escape_grads,
)
from snn_torch import (  # noqa: E402
    _K,
    edge_peak_guard,
    peak_margin_torch,
)

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "..", "docs", "results", "sp02-rigor")

TM, TS = 15.0, 4.0
THETA = 1.0
T_MAX = 40.0
GRID_PTS = 4001
T_BIAS = 0.0
INPUT_TIMES = [1.0, 2.5, 4.0, 6.0]
W_DIR = [0.8, 1.0, 1.2, 1.0]
MARGINS = [-0.1, -0.25, -0.5, -0.75, -0.9]
RHOS = [1e-3, 1e-2, 1e-1, 1.0]
T_ESC = 1.0
T_ESC_SWEEP = [0.05, 0.1, 0.25, 0.5]
FD_EPS = 1e-5


def kernel_params():
    s = (TM * TS / (TM - TS)) * math.log(TM / TS)
    k_peak = (math.exp(-s / TM) - math.exp(-s / TS)) / (TM - TS)
    return TM, TS, False, k_peak


def make_grid(dev, dtype):
    return torch.linspace(0.0, T_MAX, GRID_PTS, dtype=dtype, device=dev)


def weights_for_margin(m0, t_in, grid, dev, dtype):
    """Positive-direction weights scaled so u_peak = theta + m0 (interior max)."""
    tm, ts, alpha, k_peak = kernel_params()
    W_dir = torch.tensor([W_DIR + [0.0]], dtype=dtype, device=dev)
    W_dir = W_dir * 0.1                 # shrink first so the neuron is silent
    _, u0 = peak_margin_torch(W_dir, t_in, T_BIAS, THETA, grid, tm, ts, alpha,
                              k_peak)
    c = (THETA + m0) / float(u0.item())
    return W_dir * c


def fd_u_peak(W, t_prev, t_bias, theta, grid, tm, ts, alpha, k_peak, j, i_w,
              eps=1e-5):
    """Central-difference d(u_peak)/dW[j, i_w] via peak_margin_torch."""
    def up(Wt):
        _, u = peak_margin_torch(Wt, t_prev, t_bias, theta, grid, tm, ts,
                                 alpha, k_peak)
        return u[j, 0]
    Wp = W.detach().clone()
    Wn = W.detach().clone()
    Wp[j, i_w] = W[j, i_w] + eps
    Wn[j, i_w] = W[j, i_w] - eps
    return (up(Wp) - up(Wn)) / (2.0 * eps)


def check_a(dev, dtype):
    """Survival-integral vs channel: rate sweep + saddle-limit + FD."""
    grid = make_grid(dev, dtype)
    tm, ts, alpha, k_peak = kernel_params()
    t_in = torch.tensor(INPUT_TIMES, dtype=dtype, device=dev).view(-1, 1)
    rows = []
    for m0 in MARGINS:
        for rho0 in RHOS:
            W = weights_for_margin(m0, t_in, grid, dev, dtype)
            cfg = {"m0": m0, "rho0": rho0, "T_esc": T_ESC, "kind": "logistic"}
            for r in compare_channel_vs_escape(W, t_in, T_BIAS, THETA, grid,
                                               tm, ts, alpha, k_peak,
                                               T_esc=T_ESC, rho0=rho0,
                                               kind="logistic", fd_eps=FD_EPS):
                r["config"] = cfg
                rows.append(r)
    for T_esc in T_ESC_SWEEP:
        W = weights_for_margin(-0.25, t_in, grid, dev, dtype)
        for r in compare_channel_vs_escape(W, t_in, T_BIAS, THETA, grid,
                                           tm, ts, alpha, k_peak, T_esc=T_esc,
                                           rho0=0.1, kind="logistic",
                                           fd_eps=FD_EPS):
            r["config"] = {"m0": -0.25, "rho0": 0.1, "T_esc": T_esc,
                           "kind": "logistic"}
            rows.append(r)
    W = weights_for_margin(-0.25, t_in, grid, dev, dtype)
    for r in compare_channel_vs_escape(W, t_in, T_BIAS, THETA, grid,
                                       tm, ts, alpha, k_peak, T_esc=1.0,
                                       rho0=0.1, kind="exponential",
                                       fd_eps=FD_EPS):
        r["config"] = {"m0": -0.25, "rho0": 0.1, "T_esc": 1.0,
                       "kind": "exponential"}
        rows.append(r)
    return rows


def check_b(dev, dtype):
    """Envelope d(u_peak)/dW = K(t_peak - t_in) at interior + boundary extrema."""
    grid = make_grid(dev, dtype)
    tm, ts, alpha, k_peak = kernel_params()
    t_in = torch.tensor(INPUT_TIMES, dtype=dtype, device=dev).view(-1, 1)
    results = []
    configs = []
    W_max = weights_for_margin(-0.5, t_in, grid, dev, dtype)
    configs.append(("interior-max", W_max))
    W_min = torch.tensor([[-1.0, -0.6, -0.4, -0.2, -0.1]], dtype=dtype,
                         device=dev)   # u_max < 0 strictly (stable all_neg)
    configs.append(("interior-min-allneg", W_min))
    for name, W in configs:
        ch = compare_channel_vs_escape(W, t_in, T_BIAS, THETA, grid, tm, ts,
                                       alpha, k_peak, T_esc=T_ESC, rho0=0.1,
                                       kind="logistic")
        t_peak, u_peak = peak_margin_torch(W, t_in, T_BIAS, THETA, grid, tm,
                                           ts, alpha, k_peak)
        n_in = W.shape[1] - 1
        env = []
        for i in range(n_in + 1):
            d = t_peak[0, 0] - (t_in[i, 0] if i < n_in else T_BIAS)
            env_val = float(_K(d.reshape(1), tm, ts, alpha, k_peak)[0].item())
            fd_val = float(fd_u_peak(W, t_in, T_BIAS, THETA, grid, tm, ts,
                                     alpha, k_peak, 0, i, FD_EPS).item())
            env.append({"input": i, "envelope": env_val, "fd": fd_val,
                        "rel_err": abs(fd_val - env_val)
                        / (abs(env_val) + 1e-12)})
        guard = edge_peak_guard(W, t_in, T_BIAS, t_peak, u_peak, grid)
        results.append({"config": name, "t_peak": float(t_peak[0, 0].item()),
                        "u_peak": float(u_peak[0, 0].item()),
                        "guard": bool(guard[0, 0].item()), "envelope": env,
                        "cosine": ch[0]["cosine"], "S": ch[0]["S"]})
    W_edge = torch.tensor([[-1.0, -0.6, -0.4, -0.2, 0.0]], dtype=dtype,
                          device=dev)  # u_max = 0 exactly: branch boundary
    t_peak, u_peak = peak_margin_torch(W_edge, t_in, T_BIAS, THETA, grid,
                                       tm, ts, alpha, k_peak)
    fd_val = float(fd_u_peak(W_edge, t_in, T_BIAS, THETA, grid, tm, ts,
                             alpha, k_peak, 0, n_in, FD_EPS).item())
    env_val = float(_K((t_peak[0, 0] - T_BIAS).reshape(1), tm, ts, alpha,
                       k_peak)[0].item())
    results.append({
        "config": "cross-branch (u_max=0)",
        "t_peak": float(t_peak[0, 0].item()),
        "u_peak": float(u_peak[0, 0].item()),
        "bias_fd": fd_val, "bias_envelope": env_val,
        "bias_rel_err": abs(fd_val - env_val) / (abs(env_val) + 1e-12),
        "note": "all_neg<->max branch flip under +eps bias; documented "
                "channel non-smoothness, not an envelope failure within branch",
    })
    return results


def check_c(dev, dtype):
    """Degenerate plateau: channel deadlock vs escape gradient + guard."""
    grid = make_grid(dev, dtype)
    tm, ts, alpha, k_peak = kernel_params()
    t_in = torch.tensor(INPUT_TIMES, dtype=dtype, device=dev).view(-1, 1)
    W = torch.zeros((1, len(INPUT_TIMES) + 1), dtype=dtype, device=dev)
    t_peak, u_peak = peak_margin_torch(W, t_in, T_BIAS, THETA, grid, tm, ts,
                                       alpha, k_peak)
    guard = edge_peak_guard(W, t_in, T_BIAS, t_peak, u_peak, grid)
    ch = compare_channel_vs_escape(W, t_in, T_BIAS, THETA, grid, tm, ts,
                                   alpha, k_peak, T_esc=T_ESC, rho0=0.1,
                                   kind="logistic")
    res = escape_grads(W, t_in, T_BIAS, THETA, grid, tm, ts, alpha, k_peak,
                       T_esc=T_ESC, rho0=0.1, kind="logistic")
    g_chan = torch.tensor(ch[0]["g_chan"], dtype=dtype, device=dev)
    g_esc = res["dL_dW"][0, 0]
    return {
        "guard_flagged": bool(guard[0, 0].item()),
        "channel_grad_max_abs": float(g_chan.abs().max().item()),
        "escape_grad_max_abs": float(g_esc.abs().max().item()),
        "gap_ratio": float(g_esc.abs().max().item()
                           / (g_chan.abs().max().item() + 1e-12)),
        "escape_dL_dW": g_esc.tolist(),
        "P_fire": ch[0]["P_fire"], "S": ch[0]["S"],
    }


def gates(rows, env):
    g = {}
    fd_p = [r["fd_dP_rel"] for r in rows if r["config"]["kind"] == "logistic"]
    fd_l = [r["fd_dL_rel"] for r in rows]
    g["A_fd_gradient"] = max(fd_p) < 1e-4 and max(fd_l) < 1e-4
    main = [r for r in rows if r["config"]["kind"] == "logistic"
            and r["config"]["T_esc"] == T_ESC]
    cols = {}
    for r in main:
        cols.setdefault(r["config"]["m0"], []).append(r)
    g["A_s_dependence"] = all(
        max(abs(r["decomp_err"]) for r in col) < 1e-6 and
        (max(r["C_emp"] for r in col) - min(r["C_emp"] for r in col)) < 0.02
        for col in cols.values())
    saddle = [r for r in rows if r["config"]["T_esc"] == 0.05
              and r["config"]["kind"] == "logistic"]
    g["A_saddle_limit_C_to_1"] = len(saddle) > 0 and max(
        abs(r["C_emp"] - 1.0) for r in saddle) < 0.05
    far_narrow = [r for r in main if r["S"] >= 0.98]
    g["A_far_dead_bounded"] = len(far_narrow) > 0 and min(
        r["cosine"] for r in far_narrow) > 0.99 and all(
            r["mag_ratio_esc_chan"] < 1.0 for r in far_narrow)
    g["B_envelope_within_branch"] = all(
        max(abs(e["rel_err"]) for e in c["envelope"]) < 1e-6 and not c["guard"]
        for c in env if c["config"] != "cross-branch (u_max=0)")
    cross = [c for c in env if c["config"] == "cross-branch (u_max=0)"][0]
    g["B_cross_branch_discontinuity_documented"] = cross["bias_rel_err"] > 1.0
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", type=str, default="float64")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {dev} dtype {args.dtype}")

    rows = check_a(dev, dtype)
    env = check_b(dev, dtype)
    deg = check_c(dev, dtype)
    g = gates(rows, env)

    os.makedirs(RESULT_DIR, exist_ok=True)
    out = {"inputs": INPUT_TIMES, "kernel": {"tm": TM, "ts": TS, "theta": THETA,
           "t_max": T_MAX, "grid_pts": GRID_PTS}, "T_esc": T_ESC,
           "T_esc_sweep": T_ESC_SWEEP, "fd_eps": FD_EPS, "check_a": rows,
           "check_b": env, "check_c": deg, "gates": g}
    out_path = args.out or os.path.join(RESULT_DIR, "sp02-rigor-results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\nCheck A: channel vs exact escape-noise gradient (logistic, T_esc=1)")
    hdr = (f"{'m0':>6} {'rho0':>6} {'P':>6} {'S':>6} {'cos':>5} {'C_emp':>7} "
           f"{'C_th':>7} {'decomp':>8} {'fdP':>7} {'fdL':>7}")
    print(hdr)
    for r in rows:
        if r["config"]["kind"] != "logistic" or r["config"]["T_esc"] != 1.0:
            continue
        c = r["config"]
        print(f"{c['m0']:>6} {c['rho0']:>6.0e} {r['P_fire']:>6.2f} {r['S']:>6.2f} "
              f"{r['cosine']:>5.3f} {r['C_emp']:>7.3f} {r['C_theory']:>7.3f} "
              f"{r['decomp_err']:>8.1e} {r['fd_dP_rel']:>7.1e} {r['fd_dL_rel']:>7.1e}")
    print("\nCheck A: saddle limit (m0=-0.25, rho0=0.1): C_emp -> 1 as T_esc -> 0")
    for r in rows:
        if r["config"]["kind"] != "logistic" or r["config"]["T_esc"] == 1.0:
            continue
        if r["config"]["T_esc"] != 1.0:
            c = r["config"]
            if c["m0"] == -0.25 and c["rho0"] == 0.1:
                print(f"  T_esc={c['T_esc']:>5} P={r['P_fire']:>6.2f} "
                      f"S={r['S']:>6.3f} C_emp={r['C_emp']:>7.3f} "
                      f"f(S)={r['f_pred']:>7.3f}")
    print("\nCheck B: envelope d(u_peak)/dW vs brute-force FD")
    for c in env:
        if c["config"] == "cross-branch (u_max=0)":
            print(f"  {c['config']:<24} bias fd={c['bias_fd']:.3e} "
                  f"env={c['bias_envelope']:.3e} rel={c['bias_rel_err']:.1e}")
            continue
        worst = max(e["rel_err"] for e in c["envelope"])
        print(f"  {c['config']:<24} t_peak={c['t_peak']:6.3f} "
              f"u_peak={c['u_peak']:7.4f} guard={c['guard']} "
              f"worst envelope rel_err={worst:.1e} cos={c['cosine']:.3f}")
    print("\nCheck C: degenerate plateau")
    print(f"  guard_flagged={deg['guard_flagged']} "
          f"channel_grad_max_abs={deg['channel_grad_max_abs']:.1e} "
          f"escape_grad_max_abs={deg['escape_grad_max_abs']:.3f} "
          f"gap_ratio={deg['gap_ratio']:.0f}x S={deg['S']:.3f}")
    print("\ngates:", json.dumps(g, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
