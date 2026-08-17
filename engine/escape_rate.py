"""SP-02 rigorous upgrade: full escape-noise survival-integral gradient.

The SP-02 existence channel (snn_torch.existence_grads / local_learning_grads)
uses the deterministic peak-margin surrogate

    p_j = sigmoid((u_peak_j - theta) / T_noise)

which is the saddle-point (peak) approximation of the true escape-noise firing
probability that the SP-02 research doc defers as future work (Q2.4):

    P_fire_j = 1 - exp(-int_0^tmax rho(u_j(t)) dt),
    rho(u) = rho0 * sigmoid((u - theta)/T_esc)      (logistic rate, default)
    rho(u) = rho0 * exp((u - theta)/T_esc)          (exponential rate)

This module implements that deferred form exactly (trapezoid quadrature on the
engine grid) together with its exact expected gradient dL/dW (L = -log P_fire),
and the measurement harness that quantifies where and by how much the channel
deviates from it.

Main result (verified empirically by exp_sp02_rigor.py): for a targeted silent
neuron with survival S = exp(-Lambda), Lambda = int rho dt, the loss-gradient
ratio is, in the saddle-point (weak-coupling) limit,

    g_escape / g_channel = f(S) = -S*log(S)/(1-S)

  * far-dead neurons (Lambda << 1, S -> 1): f(S) -> 1, so the channel IS the
    exact expected gradient of the escape-noise model (this makes rigorous the
    SP-02 doc's "bounded ~1/T far-dead gradient" claim: it is the limit of the
    exact escape-noise gradient, not just a convenient surrogate);
  * near-threshold neurons (S -> 0): f(S) -> 0, so the channel over-pushes by
    up to 1/f(S) -- a monotone, closed-form bias the channel applies to
    near-firing neurons (acceptable as a stronger revival prior, but it is NOT
    the true expected gradient there).

All quantities are per-neuron / per-sample; the experiment uses B=1.
"""
import math

import torch

from snn_torch import _K, peak_margin_torch


def trapezoid_weights(grid):
    """Trapezoid quadrature weights over the engine grid (G,)."""
    h = grid[1] - grid[0]
    w = torch.full_like(grid, h)
    if w.numel():
        w[0] = 0.5 * h
        w[-1] = 0.5 * h
    return w


def _hazard(u, theta, T_esc, rho0, kind):
    if kind == "exponential":
        return rho0 * torch.exp((u - theta) / T_esc)
    return rho0 * torch.sigmoid((u - theta) / T_esc)


def _hazard_deriv(u, theta, T_esc, rho0, kind):
    """du/d lambda(u): logistic -> rho*sigma(1-sigma)/T, exp -> lambda/T."""
    lam = _hazard(u, theta, T_esc, rho0, kind)
    if kind == "exponential":
        return lam / T_esc
    s = torch.sigmoid((u - theta) / T_esc)
    return lam * (1.0 - s) / T_esc


def _U_over_grid(W, t_prev, t_bias, grid, tm, ts, alpha, k_peak):
    """Membrane potential over the response grid: (n_cur, B, G)."""
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_prev.shape[1]
    G = grid.numel()
    dev, dtype = W.device, W.dtype
    g = grid.view(1, 1, -1)
    U = torch.zeros((n_cur, B, G), dtype=dtype, device=dev)
    for i in range(n_in):
        d = g - t_prev[i].view(1, -1, 1)
        U += W[:, i].view(n_cur, 1, 1) * _K(d, tm, ts, alpha, k_peak)
    U += W[:, n_in].view(n_cur, 1, 1) * _K(g - t_bias, tm, ts, alpha, k_peak)
    return U


def survival_and_p_fire(W, t_prev, t_bias, theta, grid, tm, ts, alpha, k_peak,
                        T_esc=1.0, rho0=1.0, kind="logistic"):
    """Escape-noise survival S(T), cumulative hazard Lambda, firing probability
    P = 1 - S, plus the potential and hazard over the grid.

    Returns dict with keys: U (n_cur,B,G), lam (n_cur,B,G), w (G,),
    Lambda (n_cur,B), S (n_cur,B), P (n_cur,B).
    """
    w = trapezoid_weights(grid)
    U = _U_over_grid(W, t_prev, t_bias, grid, tm, ts, alpha, k_peak)
    lam = _hazard(U, theta, T_esc, rho0, kind)
    Lambda = (lam * w.view(1, 1, -1)).sum(-1)
    S = torch.exp(-Lambda)
    return {"U": U, "lam": lam, "w": w, "Lambda": Lambda, "S": S, "P": 1.0 - S}


def escape_grads(W, t_prev, t_bias, theta, grid, tm, ts, alpha, k_peak,
                 T_esc=1.0, rho0=1.0, kind="logistic"):
    """Exact escape-noise expected gradients for L = -log P_fire.

    dP_fire/dW_ji = S(T) * int_0^tmax rho'(u(t)) K(t - t_in_i) dt   (chain rule
    through P = 1 - exp(-int rho dt); exact, no REINFORCE needed because the
    model is a time-inhomogeneous Poisson process and the expectation of the
    gradient is the gradient of the expectation).

    Returns dict: P, S, Lambda (n_cur,B); dP_dW (n_cur,B,n_in+1);
    dL_dW (n_cur,B,n_in+1); U, lam (n_cur,B,G); w (G,).
    """
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_prev.shape[1]
    dev, dtype = W.device, W.dtype
    r = survival_and_p_fire(W, t_prev, t_bias, theta, grid, tm, ts, alpha,
                            k_peak, T_esc, rho0, kind)
    S, P, lam, w = r["S"], r["P"], r["lam"], r["w"]
    U = r["U"]
    dlam = _hazard_deriv(U, theta, T_esc, rho0, kind)
    integrand = dlam * w.view(1, 1, -1)          # (n_cur, B, G)
    dP_dW = torch.zeros((n_cur, B, n_inp), dtype=dtype, device=dev)
    for i in range(n_in):
        d = grid.view(1, 1, -1) - t_prev[i].view(1, -1, 1)
        Kd = _K(d, tm, ts, alpha, k_peak)
        dP_dW[:, :, i] = S * (integrand * Kd).sum(-1)
    d = grid.view(1, 1, -1) - t_bias
    Kd = _K(d, tm, ts, alpha, k_peak)
    dP_dW[:, :, n_in] = S * (integrand * Kd).sum(-1)
    dL_dW = -dP_dW / P.clamp(min=1e-12).unsqueeze(-1)
    return {"P": P, "S": S, "Lambda": r["Lambda"], "dP_dW": dP_dW,
            "dL_dW": dL_dW, "U": U, "lam": lam, "w": w}


def channel_grads(W, t_prev, t_bias, theta, grid, tm, ts, alpha, k_peak,
                  T_esc=1.0):
    """The SP-02 channel gradient + envelope d(u_peak)/dW, for comparison.

    Replicates snn_torch.existence_grads for lam=1, B=1, target=1:
      p = sigmoid((u_peak - theta)/T_esc)
      dL/dW_ji = -(1-p)/T_esc * K(t_peak - t_in_i)
    The envelope-theorem quantity d(u_peak)/dW_ji = K(t_peak - t_in_i) is also
    returned for the boundary check.

    Returns dict: t_peak, u_peak, p (n_cur,B); g_chan, envelope (n_cur,B,n_in+1).
    """
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_prev.shape[1]
    dev, dtype = W.device, W.dtype
    t_peak, u_peak = peak_margin_torch(W, t_prev, t_bias, theta, grid, tm, ts,
                                       alpha, k_peak)
    p = torch.sigmoid((u_peak - theta) / T_esc)
    g = -(1.0 - p) / T_esc                       # dL/d(u_peak), per sample
    g_chan = torch.zeros((n_cur, B, n_inp), dtype=dtype, device=dev)
    envelope = torch.zeros_like(g_chan)
    for i in range(n_in):
        d = t_peak - t_prev[i].view(1, -1)
        Kd = _K(d, tm, ts, alpha, k_peak)
        g_chan[:, :, i] = g * Kd
        envelope[:, :, i] = Kd
    Kd = _K(t_peak - t_bias, tm, ts, alpha, k_peak)
    g_chan[:, :, n_in] = g * Kd
    envelope[:, :, n_in] = Kd
    return {"t_peak": t_peak, "u_peak": u_peak, "p": p, "g_chan": g_chan,
            "envelope": envelope}


def fd_dP_dW(W, t_prev, t_bias, theta, grid, tm, ts, alpha, k_peak,
             T_esc, rho0, kind, j, i_w, eps=1e-5):
    """Central-difference dP_fire/dW[j, i_w], per sample. Returns (B,) tensor."""
    def P_of(Wt):
        return survival_and_p_fire(Wt, t_prev, t_bias, theta, grid, tm, ts,
                                   alpha, k_peak, T_esc, rho0, kind)["P"]
    Wp = W.detach().clone()
    Wn = W.detach().clone()
    Wp[j, i_w] = W[j, i_w] + eps
    Wn[j, i_w] = W[j, i_w] - eps
    return (P_of(Wp)[j] - P_of(Wn)[j]) / (2.0 * eps)


def fd_dL_dW(W, t_prev, t_bias, theta, grid, tm, ts, alpha, k_peak,
             T_esc, rho0, kind, j, i_w, eps=1e-5):
    """Central-difference d(-log P_fire)/dW[j, i_w], per sample. Returns (B,)."""
    def L_of(Wt):
        P = survival_and_p_fire(Wt, t_prev, t_bias, theta, grid, tm, ts,
                                alpha, k_peak, T_esc, rho0, kind)["P"]
        return -torch.log(P.clamp(min=1e-12))
    Wp = W.detach().clone()
    Wn = W.detach().clone()
    Wp[j, i_w] = W[j, i_w] + eps
    Wn[j, i_w] = W[j, i_w] - eps
    return (L_of(Wp)[j] - L_of(Wn)[j]) / (2.0 * eps)


def compare_channel_vs_escape(W, t_prev, t_bias, theta, grid, tm, ts, alpha,
                              k_peak, T_esc=1.0, rho0=1.0, kind="logistic",
                              fd_eps=1e-5):
    """Full per-config comparison. Returns a JSON-able dict.

    B must be 1. Flattens the (n_cur, B, n_in+1) gradient vectors per neuron.

    Besides the raw gradients it reports the closed-form decomposition
        g_esc,dom / g_chan,dom = f(S) * C_dom,
        f(S) = -S log S / (1-S)                     (S-dependence, exact),
        C_dom = int sig(1-sig) K_dom ds
                / (int sig ds * (1-sig_peak) * K_dom,peak)   (kernel geometry),
    and the residual |ratio_dom - f(S)*C_dom|/|ratio_dom| (exact algebra, so
    ~machine precision; it validates the decomposition numerically).
    """
    res = escape_grads(W, t_prev, t_bias, theta, grid, tm, ts, alpha, k_peak,
                       T_esc, rho0, kind)
    ch = channel_grads(W, t_prev, t_bias, theta, grid, tm, ts, alpha, k_peak,
                       T_esc)
    n_cur, B, n_inp = res["dL_dW"].shape
    n_in = n_inp - 1
    sig = torch.sigmoid((res["U"] - theta) / T_esc)
    out = []
    for j in range(n_cur):
        for b in range(B):
            g_e = res["dL_dW"][j, b]
            g_c = ch["g_chan"][j, b]
            epsg = 1e-12
            cos = float((g_e * g_c).sum() / (g_e.norm() * g_c.norm() + epsg))
            mag_ratio = float(g_e.norm() / (g_c.norm() + epsg))
            S = float(res["S"][j, b])
            P = float(res["P"][j, b])
            f_pred = -S * math.log(max(S, 1e-12)) / max(1.0 - S, 1e-12)
            dom = int(g_e[:n_in].abs().argmax().item())
            fd_p = fd_dP_dW(W, t_prev, t_bias, theta, grid, tm, ts, alpha,
                            k_peak, T_esc, rho0, kind, j, dom, fd_eps)
            fd_l = fd_dL_dW(W, t_prev, t_bias, theta, grid, tm, ts, alpha,
                            k_peak, T_esc, rho0, kind, j, dom, fd_eps)
            an_p = float(res["dP_dW"][j, b, dom])
            an_l = float(res["dL_dW"][j, b, dom])
            ratio_dom = float(g_e[dom] / (g_c[dom] + epsg))
            c_theory = None
            decomp_err = None
            if kind == "logistic" and abs(g_c[dom].item()) > 1e-12:
                d = grid.view(1, 1, -1) - t_prev[dom].view(1, -1, 1)
                Kd = _K(d, tm, ts, alpha, k_peak)
                wg = res["w"].view(1, 1, -1)
                num = float((sig[j, b].unsqueeze(0)
                             * (1.0 - sig[j, b].unsqueeze(0)) * Kd[j, b]
                             * wg).sum().item())
                den = (float((sig[j, b] * wg[0, 0]).sum().item())
                       * (1.0 - float(ch["p"][j, b].item()))
                       * abs(float(ch["envelope"][j, b, dom].item())))
                c_theory = num / (den + 1e-12)
                decomp_err = abs(ratio_dom - f_pred * c_theory) / (abs(ratio_dom) + 1e-12)
            out.append({
                "neuron": j,
                "t_peak": float(ch["t_peak"][j, b]),
                "u_peak": float(ch["u_peak"][j, b]),
                "p_peak": float(ch["p"][j, b]),
                "P_fire": P,
                "S": S,
                "Lambda": float(res["Lambda"][j, b]),
                "cosine": cos,
                "mag_ratio_esc_chan": mag_ratio,
                "f_pred": f_pred,
                "ratio_dom_esc_chan": ratio_dom,
                "C_theory": c_theory,
                "C_emp": None if c_theory is None else ratio_dom / f_pred,
                "decomp_err": decomp_err,
                "fd_dP_rel": float(abs(fd_p[b].item() - an_p)
                                   / (abs(an_p) + 1e-12)),
                "fd_dL_rel": float(abs(fd_l[b].item() - an_l)
                                   / (abs(an_l) + 1e-12)),
                "g_esc": res["dL_dW"][j, b].tolist(),
                "g_chan": g_c.tolist(),
            })
    return out
