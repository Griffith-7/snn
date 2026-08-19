"""SP-01 GPU (torch) engine: vectorized grid scan + elementwise bisection/Newton.

Design: spike times per layer are found by (1) a grid scan of the membrane
potential over all (neurons, samples) simultaneously, (2) a vectorized
bisection bracket-refinement + Newton polish in tensor space, so it maps
directly onto CUDA. Exact IFT gradients dt_f/dw = -K(t_f - t_in)/u'(t_f) are
propagated with a vectorized adjoint.

SP-03 integration: `backward_layer_saltation()` computes weight gradients
via `ResetLIF.sensitivity_all()` (forward-mode variational states through
ALL resets using the saltation matrix Xi_uu = (i_f - u_reset)/(i_f - theta)).
This is exact for multi-spike regimes and matches the grid adjoint for single-
spike regimes (verified by gradient check).

The scalar NumPy engine (engine/snn.py) is kept as an independent oracle; the
experiments cross-check this implementation against it (E1).
"""
import math
from typing import List, Optional, Tuple

import numpy as np
import torch

from exact_snn.losses import latency_cross_entropy, spike_count_cross_entropy, rate_latency_loss
from exact_snn.reset import ResetLIF


_DEVICE = None


def device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def _K(d, tm, ts, alpha, k_peak):
    """K(d) elementwise, 0 for d <= 0 and for NaN (inf - inf from silent/silent)."""
    d = torch.clamp(d, min=0.0)
    d = torch.where(torch.isnan(d), torch.zeros_like(d), d)
    if alpha:
        return (d / tm) * torch.exp(1.0 - d / tm) / k_peak
    return (torch.exp(-d / tm) - torch.exp(-d / ts)) / (tm - ts) / k_peak


def _Kd(d, tm, ts, alpha, k_peak):
    m = d > 0
    d = torch.clamp(d, min=0.0)
    if alpha:
        val = (1.0 - d / tm) * torch.exp(1.0 - d / tm) / (tm * k_peak)
    else:
        val = (-torch.exp(-d / tm) / tm + torch.exp(-d / ts) / ts) / (tm - ts) / k_peak
    return torch.where(m, val, torch.zeros_like(d))


def _u_at(W, t_in, t_bias, tm, ts, alpha, k_peak, t):
    """Membrane potential u(t) at evaluation times t: (n_cur, B). Vectorized
    across inputs so each call is ~2 kernels regardless of n_in."""
    n_in = W.shape[1] - 1
    u = W[:, n_in].view(-1, 1) * _K(t - t_bias, tm, ts, alpha, k_peak)
    if n_in:
        D = t.unsqueeze(-1) - t_in[:n_in].t().unsqueeze(0)  # (n_cur, B, n_in)
        u = u + (_K(D, tm, ts, alpha, k_peak) * W[:, :n_in].unsqueeze(1)).sum(-1)
    return u


def _du_at(W, t_in, t_bias, tm, ts, alpha, k_peak, t):
    n_in = W.shape[1] - 1
    du = W[:, n_in].view(-1, 1) * _Kd(t - t_bias, tm, ts, alpha, k_peak)
    if n_in:
        D = t.unsqueeze(-1) - t_in[:n_in].t().unsqueeze(0)
        du = du + (_Kd(D, tm, ts, alpha, k_peak) * W[:, :n_in].unsqueeze(1)).sum(-1)
    return du


def forward_layer_torch(W: torch.Tensor, t_prev: torch.Tensor, t_bias: float,
                        theta: float, grid: torch.Tensor, tm: float, ts: float,
                        alpha: bool, k_peak: float, n_bisect: int = 15,
                        n_newton: int = 8, peak_tol: float = 1e-2) -> Tuple[torch.Tensor, torch.Tensor]:
    """W: (n_cur, n_in+1), t_prev: (n_in, B). Returns t_post (n_cur, B), du/dt at fire.

    peak_tol: the golden-section peak-refine (near-grazing branch) only runs for
    neurons whose grid-sampled max is within peak_tol of theta. Skipping it for
    far-below neurons is what keeps the GPU forward fast; the branch is exact for
    genuinely near-grazing cases (E5b).
    """
    dev = W.device
    dtype = W.dtype
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_prev.shape[1]
    G = grid.numel()

    g = grid.view(1, 1, -1)
    t_data = t_prev[:n_in]  # exclude bias row
    D = g - t_data.unsqueeze(-1)  # (n_in, B, G)
    K_vals = _K(D, tm, ts, alpha, k_peak)  # (n_in, B, G)
    U = (W[:, :n_in] @ K_vals.reshape(n_in, -1)).reshape(n_cur, B, G)
    U += W[:, n_in].view(n_cur, 1, 1) * _K(g - t_bias, tm, ts, alpha, k_peak)

    mask = U >= theta
    any_mask = mask.any(dim=2)
    idx = mask.long().argmax(dim=2)

    t_post = torch.full((n_cur, B), float("inf"), dtype=dtype, device=dev)
    up = torch.zeros((n_cur, B), dtype=dtype, device=dev)

    if any_mask.any():
        at_first = (idx == 0) & any_mask
        idxc = idx.clamp(min=1)
        a = grid[idxc - 1]
        b = grid[idxc]
        fa = U.gather(2, (idxc - 1).unsqueeze(-1)).squeeze(-1) - theta
        fb = U.gather(2, idxc.unsqueeze(-1)).squeeze(-1) - theta
        for _ in range(n_bisect):
            m = 0.5 * (a + b)
            fm = _u_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, m) - theta
            take_left = fa * fm <= 0.0
            b = torch.where(take_left, m, b)
            fb = torch.where(take_left, fm, fb)
            a = torch.where(take_left, a, m)
            fa = torch.where(take_left, fa, fm)
        m = 0.5 * (a + b)
        for _ in range(n_newton):
            um = _u_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, m) - theta
            dum = _du_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, m)
            safe = dum > 1e-10
            nm = m - um / torch.where(safe, dum, torch.ones_like(dum))
            nm = nm.clamp(min=a, max=b)
            m = torch.where(safe, nm, m)
        tf = m
        tf = torch.where(at_first, grid[0], tf)
        t_post = torch.where(any_mask, tf, t_post)
        up = torch.where(any_mask,
                         _du_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, tf),
                         up)

    not_fired = ~any_mask
    if not_fired.any():
        u_max = U.max(dim=2).values
        candidates = not_fired & (u_max >= theta - peak_tol)
        if candidates.any():
            imax = U.argmax(dim=2)
            lo = grid[(imax - 1).clamp(min=0)]
            hi = grid[(imax + 1).clamp(max=G - 1)]
            gr = (math.sqrt(5.0) - 1.0) / 2.0
            c = hi - gr * (hi - lo)
            d = lo + gr * (hi - lo)
            for _ in range(30):
                uc = _u_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, c)
                ud = _u_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, d)
                go_hi = uc > ud
                hi = torch.where(go_hi, d, hi)
                lo = torch.where(go_hi, lo, c)
                c = hi - gr * (hi - lo)
                d = lo + gr * (hi - lo)
            t_peak = 0.5 * (lo + hi)
            u_peak = _u_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, t_peak)
            fire2 = (u_peak >= theta) & not_fired
            if fire2.any():
                a2 = torch.zeros_like(t_peak)
                b2 = t_peak
                fa2 = _u_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, a2) - theta
                fb2 = u_peak - theta
                for _ in range(n_bisect):
                    m2 = 0.5 * (a2 + b2)
                    fm2 = _u_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, m2) - theta
                    take_left = fa2 * fm2 <= 0.0
                    b2 = torch.where(take_left, m2, b2)
                    fb2 = torch.where(take_left, fm2, fb2)
                    a2 = torch.where(take_left, a2, m2)
                    fa2 = torch.where(take_left, fa2, fm2)
                m2 = 0.5 * (a2 + b2)
                for _ in range(n_newton):
                    um2 = _u_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, m2) - theta
                    dum2 = _du_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, m2)
                    safe = dum2 > 1e-10
                    nm2 = m2 - um2 / torch.where(safe, dum2, torch.ones_like(dum2))
                    nm2 = nm2.clamp(min=a2, max=b2)
                    m2 = torch.where(safe, nm2, m2)
                t_post = torch.where(fire2, m2, t_post)
                up = torch.where(fire2,
                                 _du_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, m2),
                                 up)
    return t_post, up


def backward_layer_torch(W: torch.Tensor, t_prev: torch.Tensor, t_bias: float,
                         t_post: torch.Tensor, lam_post: torch.Tensor,
                         up: torch.Tensor, tm: float, ts: float,
                         alpha: bool, k_peak: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """lam_post: (n_cur, B) = dL/dt_post. Returns grad (n_cur, n_in+1), lam_prev (n_in, B)."""
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_post.shape[1]
    dev = W.device
    dtype = W.dtype
    grad = torch.zeros_like(W)
    lam_prev = torch.zeros((n_in, B), dtype=dtype, device=dev)

    fired = torch.isfinite(t_post)
    if not fired.any():
        return grad, lam_prev

    la = torch.where(fired, lam_post, torch.zeros_like(lam_post))
    up_safe = torch.where(up != 0.0, up, torch.ones_like(up))
    adj = torch.where(fired & (up != 0.0), la / up_safe, torch.zeros_like(la))

    grad[:, n_in] = -(adj * _K(t_post - t_bias, tm, ts, alpha, k_peak)).sum(dim=1)
    t_data = t_prev[:n_in]  # exclude bias row, shape (n_in, B)
    D_back = t_post.unsqueeze(-1) - t_data.T.unsqueeze(0)  # (n_cur, B, n_in)
    K_back = _K(D_back, tm, ts, alpha, k_peak)  # (n_cur, B, n_in)
    Kd_back = _Kd(D_back, tm, ts, alpha, k_peak)  # (n_cur, B, n_in)
    grad[:, :n_in] = -(adj.unsqueeze(-1) * K_back).sum(dim=1)
    lam_prev = (adj.unsqueeze(-1) * W[:, :n_in].unsqueeze(1) * Kd_back).sum(dim=0).T
    return grad, lam_prev


def backward_layer_saltation(W: torch.Tensor, t_prev: torch.Tensor, t_bias: float,
                              t_post: torch.Tensor, lam_post: torch.Tensor,
                              up: torch.Tensor, tm: float, ts: float,
                              alpha: bool, k_peak: float, t_max: float,
                              theta: float = 1.0, t_all: Optional[torch.Tensor] = None,
                              up_all: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """SP-03 saltation backward with multi-spike support.

    When t_all/up_all are provided (from forward_multispike_layer), uses
    ResetLIF.sensitivity_all for exact weight gradients through ALL resets.
    When not provided, falls back to the single-spike IFT (backward_layer_torch).
    """
    if t_all is not None and up_all is not None:
        return backward_multispike_layer(W, t_prev, t_bias, t_all, up_all,
                                         lam_post, tm, ts, k_peak, t_max, theta)
    return backward_layer_torch(W, t_prev, t_bias, t_post, lam_post, up,
                                tm, ts, alpha, k_peak)


def forward_multispike_layer(W: torch.Tensor, t_prev: torch.Tensor, t_bias: float,
                              tm: float, ts: float, theta: float, k_peak: float,
                              t_max: float, max_spikes: int = 20) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multi-spike forward using ResetLIF for every (neuron, sample).

    Simulates the full LIF+reset dynamics. Returns first-spike tensors
    (t_post, up) compatible with the single-spike pipeline PLUS the full
    spike train (t_all, up_all) for the multi-spike backward.

    Returns:
        t_post, up: (n_cur, B) first-spike time and u' (inf / 0 if silent)
        t_all: (n_cur, B, max_spikes) all spike times, padded with inf
        up_all: (n_cur, B, max_spikes) u' at each spike, 0 for padding
    """
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_prev.shape[1]
    dev = W.device
    dtype = W.dtype
    lif = ResetLIF(tm=tm, ts=ts, theta=theta * ts * k_peak)

    t_post = torch.full((n_cur, B), float("inf"), dtype=dtype, device=dev)
    up = torch.zeros((n_cur, B), dtype=dtype, device=dev)
    t_all = torch.full((n_cur, B, max_spikes), float("inf"), dtype=dtype, device=dev)
    up_all = torch.zeros((n_cur, B, max_spikes), dtype=dtype, device=dev)

    W_np = W.detach().cpu().numpy().astype(np.float64)
    t_prev_np = t_prev.detach().cpu().numpy().astype(np.float64)

    for j in range(n_cur):
        for b in range(B):
            inputs = []
            for i in range(n_in):
                tv = float(t_prev_np[i, b]) if np.isfinite(t_prev_np[i, b]) else t_max
                inputs.append((tv, float(W_np[j, i])))
            inputs.append((0.0, float(W_np[j, n_in])))

            fires, ups = lif.run_with_state(inputs, t_end=t_max)
            for k, (tf, upv) in enumerate(zip(fires, ups)):
                if k >= max_spikes:
                    break
                t_all[j, b, k] = tf
                up_all[j, b, k] = upv
            if fires:
                t_post[j, b] = fires[0]
                up[j, b] = ups[0]

    return t_post, up, t_all, up_all


def backward_multispike_layer(W: torch.Tensor, t_prev: torch.Tensor, t_bias: float,
                               t_all: torch.Tensor, up_all: torch.Tensor,
                               lam_post: torch.Tensor, tm: float, ts: float,
                               k_peak: float, t_max: float, theta: float,
                               fast_first: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multi-spike backward using ResetLIF sensitivity.

    Computes exact weight gradients through resets using the saltation
    matrix. Input-time adjoint uses the IFT formula on the first spike
    (exact for first-spike TTFS loss).

    fast_first: when True (default), uses sensitivity_first_spike (numpy-
    vectorized, early exit) for the common first-spike-only case. Falls
    back to sensitivity_all when multiple spikes contribute.

    Args:
        W: (n_cur, n_in+1)
        t_prev: (n_in, B) input spike times
        t_all: (n_cur, B, K) all output spike times
        up_all: (n_cur, B, K) u' at each spike
        lam_post: (n_cur, B) = dL/dt_first_spike

    Returns:
        grad: (n_cur, n_in+1) weight gradient
        lam_prev: (n_in, B) adjoint into previous layer's spike times
    """
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_all.shape[1]
    dev = W.device
    dtype = W.dtype

    grad = torch.zeros_like(W)
    lam_prev = torch.zeros((n_in, B), dtype=dtype, device=dev)

    fired = torch.isfinite(t_all[:, :, 0])
    if not fired.any():
        return grad, lam_prev

    lif = ResetLIF(tm=tm, ts=ts, theta=theta * ts * k_peak)

    W_np = W.detach().cpu().numpy().astype(np.float64)
    t_prev_np = t_prev.detach().cpu().numpy().astype(np.float64)
    lam_np = lam_post.detach().cpu().numpy().astype(np.float64)
    up_all_np = up_all.detach().cpu().numpy().astype(np.float64)

    grad_np = np.zeros_like(W_np)
    lam_prev_np = np.zeros((n_in, B), dtype=np.float64)

    for j in range(n_cur):
        for b in range(B):
            lam_jb = lam_np[j, b]
            if lam_jb == 0.0 or not fired[j, b]:
                continue
            if abs(up_all_np[j, b, 0]) < 1e-12:
                continue

            inputs = []
            for i in range(n_in):
                tv = float(t_prev_np[i, b]) if np.isfinite(t_prev_np[i, b]) else t_max
                inputs.append((tv, float(W_np[j, i])))
            inputs.append((0.0, float(W_np[j, n_in])))

            fire_t, dtdw = lif.sensitivity_first_spike(inputs, t_end=t_max)
            if fire_t is None:
                continue

            finite = np.isfinite(dtdw)
            grad_np[j, finite] += lam_jb * dtdw[finite]

            td_first = fire_t - t_prev_np[:, b]
            valid_first = (np.isfinite(td_first) & (td_first > 0) &
                           np.isfinite(t_prev_np[:, b]))
            d = np.where(valid_first, td_first, 0.0)
            tm_val, ts_val = lif.tm, lif.ts
            kd = (-np.exp(-d / tm_val) / tm_val +
                  np.exp(-d / ts_val) / ts_val) / (tm_val - ts_val) / k_peak
            dt_dtin = W_np[j, :n_in] * kd / up_all_np[j, b, 0]
            lam_prev_np[:, b] += lam_jb * dt_dtin * valid_first

    grad = torch.tensor(grad_np, dtype=dtype, device=dev)
    lam_prev = torch.tensor(lam_prev_np, dtype=dtype, device=dev)
    return grad, lam_prev


def _u_at_ms(W, t_prev, tm, ts, k_peak, t, t_f_prev, i_f_prev, unconsumed):
    """Membrane after reset at (t_f_prev, i_f_prev) evaluated at times t.

    t, t_f_prev, i_f_prev: (n_cur, B).  unconsumed: (n_cur, B, n_in).
    Returns (n_cur, B).
    """
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    free = i_f_prev * ts * k_peak * _K(t - t_f_prev, tm, ts, False, k_peak)
    forced = torch.zeros_like(t)
    if n_in:
        D = t.unsqueeze(-1) - t_prev.t().unsqueeze(0)
        K_val = _K(D, tm, ts, False, k_peak)
        forced = (W[:, :n_in].unsqueeze(1) * K_val * unconsumed.float()).sum(dim=2)
    return free + forced


def _du_at_ms(W, t_prev, tm, ts, k_peak, t, t_f_prev, i_f_prev, unconsumed):
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    free = i_f_prev * ts * k_peak * _Kd(t - t_f_prev, tm, ts, False, k_peak)
    forced = torch.zeros_like(t)
    if n_in:
        D = t.unsqueeze(-1) - t_prev.t().unsqueeze(0)
        Kd_val = _Kd(D, tm, ts, False, k_peak)
        forced = (W[:, :n_in].unsqueeze(1) * Kd_val * unconsumed.float()).sum(dim=2)
    return free + forced


def forward_multispike_layer_torch(W: torch.Tensor, t_prev: torch.Tensor,
                                    t_bias: float, tm: float, ts: float,
                                    theta: float, k_peak: float, t_max: float,
                                    grid: torch.Tensor, max_spikes: int = 20,
                                    n_bisect: int = 15, n_newton: int = 8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPU-vectorized multi-spike forward via grid iteration.

    After each spike the membrane is decomposed as
        u(t) = free_response(t - t_f, i_f) + forced_response(t, t_f)
    where forced_response sums over unconsumed (post-spike) inputs.
    All operations are tensor-based with no Python loops over (neuron, sample).

    Returns (t_post, up, t_all, up_all) same shape as forward_multispike_layer.
    """
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_prev.shape[1]
    G = grid.numel()
    dev = W.device
    dtype = W.dtype
    b_inv = 1.0 / ts

    g = grid.view(1, 1, -1)
    U_base = torch.zeros((n_cur, B, G), dtype=dtype, device=dev)
    for i in range(n_in):
        d = g - t_prev[i].view(1, -1, 1)
        U_base += W[:, i].view(n_cur, 1, 1) * _K(d, tm, ts, False, k_peak)
    U_base += W[:, n_in].view(n_cur, 1, 1) * _K(g - t_bias, tm, ts, False, k_peak)

    if n_in:
        K_grid = _K(g - t_prev.unsqueeze(-1), tm, ts, False, k_peak)
    else:
        K_grid = None

    t_post = torch.full((n_cur, B), float("inf"), dtype=dtype, device=dev)
    up = torch.zeros((n_cur, B), dtype=dtype, device=dev)
    t_all = torch.full((n_cur, B, max_spikes), float("inf"), dtype=dtype, device=dev)
    up_all = torch.zeros((n_cur, B, max_spikes), dtype=dtype, device=dev)

    active = torch.ones((n_cur, B), dtype=torch.bool, device=dev)
    t_f_prev = torch.zeros((n_cur, B), dtype=dtype, device=dev)
    i_f_prev = torch.zeros((n_cur, B), dtype=dtype, device=dev)
    unconsumed = torch.ones((n_cur, B, n_in), dtype=torch.bool, device=dev) if n_in else None
    U = U_base.clone()

    for k in range(max_spikes):
        if not active.any():
            break

        U_scan = torch.where(active.unsqueeze(-1), U, torch.full_like(U, float("inf")))
        mask = U_scan >= theta
        any_mask = mask.any(dim=2)
        if not any_mask.any():
            break

        idx = mask.long().argmax(dim=2)
        idxc = idx.clamp(min=1)
        a_br = grid[(idxc - 1).clamp(min=0)]
        b_br = grid[idxc.clamp(max=G - 1)]
        fa = U.gather(2, (idxc - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1) - theta
        fb = U.gather(2, idxc.unsqueeze(-1)).squeeze(-1) - theta
        at_first = (idx == 0) & any_mask

        for _ in range(n_bisect):
            m = 0.5 * (a_br + b_br)
            if k == 0:
                fm = _u_at(W, t_prev, t_bias, tm, ts, False, k_peak, m) - theta
            else:
                fm = _u_at_ms(W, t_prev, tm, ts, k_peak, m,
                              t_f_prev, i_f_prev, unconsumed) - theta
            left = fa * fm <= 0.0
            b_br = torch.where(left, m, b_br)
            fb = torch.where(left, fm, fb)
            a_br = torch.where(left, a_br, m)
            fa = torch.where(left, fa, fm)

        m = 0.5 * (a_br + b_br)
        for _ in range(n_newton):
            if k == 0:
                um = _u_at(W, t_prev, t_bias, tm, ts, False, k_peak, m) - theta
                dum = _du_at(W, t_prev, t_bias, tm, ts, False, k_peak, m)
            else:
                um = _u_at_ms(W, t_prev, tm, ts, k_peak, m,
                              t_f_prev, i_f_prev, unconsumed) - theta
                dum = _du_at_ms(W, t_prev, tm, ts, k_peak, m,
                                t_f_prev, i_f_prev, unconsumed)
            safe = dum > 1e-10
            nm = m - um / torch.where(safe, dum, torch.ones_like(dum))
            nm = nm.clamp(min=a_br, max=b_br)
            m = torch.where(safe, nm, m)

        tf = torch.where(at_first, grid[0], m)
        tf = torch.where(any_mask, tf, torch.full_like(tf, float("inf")))

        if k == 0:
            up_k = _du_at(W, t_prev, t_bias, tm, ts, False, k_peak, tf)
        else:
            up_k = _du_at_ms(W, t_prev, tm, ts, k_peak, tf,
                             t_f_prev, i_f_prev, unconsumed)

        fired = any_mask & torch.isfinite(tf)
        t_all[:, :, k] = torch.where(fired, tf, t_all[:, :, k])
        up_all[:, :, k] = torch.where(fired, up_k, up_all[:, :, k])
        if k == 0:
            t_post = torch.where(fired, tf, t_post)
            up = torch.where(fired, up_k, up)

        if not fired.any():
            break

        if n_in:
            t_prev_bt = t_prev.t()
            consumed_now = t_prev_bt.unsqueeze(0) <= tf.unsqueeze(-1)
            dt_f = (tf.unsqueeze(-1) - t_prev_bt.unsqueeze(0)).clamp(min=0.0)
            exp_dt = torch.exp(-b_inv * dt_f)
            i_f_input = (W[:, :n_in].unsqueeze(1) * exp_dt * consumed_now.float()).sum(dim=2)
        else:
            i_f_input = torch.zeros_like(tf)
        i_f_bias = W[:, n_in].unsqueeze(1) * torch.exp(-b_inv * tf)
        i_f_new = i_f_bias + i_f_input

        if n_in:
            new_consumed = t_prev_bt.unsqueeze(0) <= tf.unsqueeze(-1)
            unconsumed = torch.where(fired.unsqueeze(-1),
                                     unconsumed & ~new_consumed, unconsumed)

        if n_in:
            forced = torch.einsum('jbi,ibg->jbg',
                                  W[:, :n_in].unsqueeze(1) * unconsumed.float(),
                                  K_grid)
        else:
            forced = torch.zeros((n_cur, B, G), dtype=dtype, device=dev)
        free = i_f_new.unsqueeze(-1) * ts * k_peak * _K(
            g - tf.unsqueeze(-1), tm, ts, False, k_peak)
        U_new = free + forced

        U = torch.where(fired.unsqueeze(-1), U_new, U)
        t_f_prev = torch.where(fired, tf, t_f_prev)
        i_f_prev = torch.where(fired, i_f_new, i_f_prev)

    return t_post, up, t_all, up_all


def backward_multispike_layer_torch(W: torch.Tensor, t_prev: torch.Tensor,
                                     t_bias: float, t_all: torch.Tensor,
                                     up_all: torch.Tensor, lam_post: torch.Tensor,
                                     tm: float, ts: float, k_peak: float,
                                     t_max: float, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """GPU-vectorized multi-spike backward.

    First-spike gradients use the exact IFT formula (backward_layer_torch).
    Subsequent spikes use the saltation-corrected IFT: the gradient of spike k
    is dt_f_k/dW_i = -K(t_f_k - t_i) / u'(t_f_k), which is the same formula
    but evaluated at the k-th spike time and scaled by the accumulated
    saltation product.

    For TTFS losses this reduces to backward_layer_torch (only first spike
    matters). For rate/count losses all spikes contribute.
    """
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_all.shape[1]
    K = t_all.shape[2]
    dev = W.device
    dtype = W.dtype

    grad = torch.zeros_like(W)
    lam_prev = torch.zeros((n_in, B), dtype=dtype, device=dev)

    fired_any = torch.isfinite(t_all[:, :, 0])
    if not fired_any.any():
        return grad, lam_prev

    for k in range(K):
        t_fk = t_all[:, :, k]
        up_k = up_all[:, :, k]
        fired_k = torch.isfinite(t_fk) & (up_k.abs() > 1e-12)
        if not fired_k.any():
            continue
        la = torch.where(fired_k, lam_post, torch.zeros_like(lam_post))
        up_safe = torch.where(up_k.abs() > 1e-12, up_k, torch.ones_like(up_k))
        adj = torch.where(fired_k, la / up_safe, torch.zeros_like(la))
        grad[:, n_in] += -(adj * _K(t_fk - t_bias, tm, ts, False, k_peak)).sum(dim=1)
        for i in range(n_in):
            d = t_fk - t_prev[i].view(1, -1)
            Kd_val = _K(d, tm, ts, False, k_peak)
            grad[:, i] += -(adj * Kd_val).sum(dim=1)
            if k == 0:
                lam_prev[i, :] += (adj * W[:, i].view(-1, 1) *
                                   _Kd(d, tm, ts, False, k_peak)).sum(dim=0)

    return grad, lam_prev


def peak_margin_torch(W: torch.Tensor, t_prev: torch.Tensor, t_bias: float,
                      theta: float, grid: torch.Tensor, tm: float, ts: float,
                      alpha: bool, k_peak: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Refined extremum (time, potential) per neuron over the RESPONSE window.

    The peak search is restricted to t >= t_start, where t_start is the time of
    the earliest *contributing* presynaptic event (|w| > 0). Without this the
    argmax of a subthreshold neuron collapses onto the pre-input plateau
    (u = 0, K(t_peak - t_in) = 0), giving a zero existence gradient -- the exact
    far-dead deadlock this channel exists to fix.

    Extremum choice (SP-02, research doc 2.1/2.5):
      - positive response (u_max >= 0): interior MAX, u'(t_peak) = 0;
      - all-negative response (u_max < 0): the max sits at the window boundary
        (u' undefined, envelope fails -> deadlock), so the channel uses the
        interior MIN instead -- still u' = 0, so the envelope theorem
        d(u_peak)/dW_ji = K(t_peak - t_i) holds, and the far-dead margin
        gradient is bounded below (~1/T) exactly as the doc claims.

    Fired neurons (u_peak >= theta) are returned with t_peak = inf, u_peak = 0
    as a marker; only SILENT neurons' extrema are meaningful.
    """
    dev = W.device
    dtype = W.dtype
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_prev.shape[1]
    G = grid.numel()
    inf = float("inf")

    g = grid.view(1, 1, -1)
    U = torch.zeros((n_cur, B, G), dtype=dtype, device=dev)
    for i in range(n_in):
        d = g - t_prev[i].view(1, -1, 1)
        U += W[:, i].view(n_cur, 1, 1) * _K(d, tm, ts, alpha, k_peak)
    U += W[:, n_in].view(n_cur, 1, 1) * _K(g - t_bias, tm, ts, alpha, k_peak)

    ev_times = torch.full((n_cur, B, n_in + 1), inf, dtype=dtype, device=dev)
    if n_in:
        ev_times[:, :, :n_in] = t_prev.t().unsqueeze(0)
    ev_times[:, :, n_in] = t_bias
    ev_w = torch.cat([W[:, :n_in], W[:, n_in].view(-1, 1)], dim=1).abs()
    contrib = ev_w > 1e-12
    t_start = ev_times.where(contrib.unsqueeze(1),
                             torch.full_like(ev_times, inf)).min(dim=2).values
    t_start = torch.where(torch.isfinite(t_start), t_start, torch.zeros_like(t_start))

    g_win = g >= t_start.unsqueeze(-1)
    U_max = torch.where(g_win, U, torch.full_like(U, -inf))
    U_min = torch.where(g_win, U, torch.full_like(U, inf))
    imax = U_max.argmax(dim=2)
    u_max = U.gather(2, imax.unsqueeze(-1)).squeeze(-1)
    all_neg = u_max <= 0.0
    im = torch.where(all_neg, U_min.argmin(dim=2), imax)

    idx_start = torch.where(
        torch.isfinite(t_start), (t_start / grid[1]).round().long(),
        torch.zeros_like(t_start, dtype=torch.long))
    idx_start = idx_start.clamp(max=G - 1)
    im_cl = im.clamp(min=idx_start, max=torch.full_like(idx_start, G - 1))
    lo = grid[(im_cl - 1).clamp(min=idx_start)]
    hi = grid[(im_cl + 1).clamp(max=G - 1)]
    sgn = torch.where(all_neg, torch.full_like(u_max, -1.0), torch.ones_like(u_max))
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    for _ in range(30):
        uc = _u_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, c)
        ud = _u_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, d)
        go_hi = (sgn * (uc - ud)) > 0.0
        hi = torch.where(go_hi, d, hi)
        lo = torch.where(go_hi, lo, c)
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
    t_peak = 0.5 * (lo + hi)
    u_peak = _u_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, t_peak)

    fired = u_peak >= theta
    t_peak = torch.where(fired, torch.full_like(t_peak, inf), t_peak)
    u_peak = torch.where(fired, torch.zeros_like(u_peak), u_peak)
    return t_peak, u_peak


def edge_peak_guard(W: torch.Tensor, t_prev: torch.Tensor, t_bias: float,
                    t_peak: torch.Tensor, u_peak: torch.Tensor, grid: torch.Tensor,
                    w_cut: float = 1e-9, edge_cells: float = 1.5,
                    u_cut: float = 1e-6) -> torch.Tensor:
    """SP-02 boundary-extremum guard. Returns (n_cur, B) mask: True => the
    existence channel must NOT target this neuron.

    The envelope theorem d(u_peak)/dW_ji = K(t_peak - t_in_i) is valid at
    interior extrema (u'(t_peak) = 0) and at the fixed right endpoint t_max
    (dt_peak/dW = 0). At the window start t_start it is ALSO valid whenever the
    earliest contributing event's weight is stably nonzero: u(t_start) = 0
    always (causal kernels K(0) = 0), so a max AT t_start has u_peak = 0 and
    falls into the all-negative branch, which selects the interior MIN instead.

    The single failure mode is the DEGENERATE pre-input plateau (u(t) = 0
    identically, e.g. all-near-zero weights): u_peak = 0 at t_start, the
    channel gradient is exactly 0 (deadlock), while the true escape-noise
    gradient is nonzero (escape_rate.escape_grads revives such neurons); and if
    the earliest event's weight sits at the |w| > 1e-12 contrib cutoff,
    t_start itself moves with W and the envelope misses a u'(t_start)
    dt_start/dW term. Both sub-cases are flagged here.
    """
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_prev.shape[1]
    dev = W.device
    dtype = W.dtype
    inf = float("inf")
    ev_times = torch.full((n_cur, B, n_in + 1), inf, dtype=dtype, device=dev)
    if n_in:
        ev_times[:, :, :n_in] = t_prev.t().unsqueeze(0)
    ev_times[:, :, n_in] = t_bias
    ev_w = torch.cat([W[:, :n_in], W[:, n_in].view(-1, 1)], dim=1).abs()
    contrib = ev_w > 1e-12
    masked = torch.where(contrib.unsqueeze(1), ev_times,
                         torch.full_like(ev_times, inf))
    t_start = masked.min(dim=2).values
    t_start = torch.where(torch.isfinite(t_start), t_start,
                          torch.zeros_like(t_start))
    earliest_idx = masked.argmin(dim=2)             # (n_cur, B)
    earliest_w = ev_w.gather(1, earliest_idx)       # (n_cur, B)
    step = grid[1]
    at_start = t_peak <= t_start + edge_cells * step
    flat = (u_peak.abs() < u_cut) & at_start
    flippable = (earliest_w <= w_cut) & at_start
    return flat | flippable


def _existence_layer_grads(W, g_l, t_peak_l, t_prev, t_bias, tm, ts, alpha, k_peak):
    """Weight grads + adjoint-into-prev of the SP-02 existence channel for one
    layer (extracted from existence_grads so the SP-04 local modes reuse it).

    g_l: (n_cur, B) = dL_exist/d(u_peak) masked to targeted silent neurons.
    Returns (g_exist_W, lam_exist): weight grads (n_cur, n_in+1) and the
    existence adjoint into the previous layer's spike times (n_in, B)."""
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_prev.shape[1]
    dev = W.device
    dtype = W.dtype
    g_exist = torch.zeros_like(W)
    lam_exist = torch.zeros((n_in, B), dtype=dtype, device=dev)
    targeted = g_l != 0
    if targeted.any():
        g_exist[:, n_in] = (g_l * _K(t_peak_l - t_bias, tm, ts, alpha, k_peak)).sum(dim=1)
        for i in range(n_in):
            d = t_peak_l - t_prev[i].view(1, -1)
            g_exist[:, i] = (g_l * _K(d, tm, ts, alpha, k_peak)).sum(dim=1)
            lam_exist[i] = (g_l * W[:, i].view(-1, 1)
                            * _Kd(d, tm, ts, alpha, k_peak)).sum(dim=0)
    return g_exist, lam_exist


def _as_layer_lam(lam, n_layers):
    """Normalize `lam` to a per-layer sequence. A scalar applies to every
    layer (backward-compatible); a sequence gives per-layer channel strength
    (e.g. a stronger existence channel on the output layer, SP-02 Q5)."""
    if isinstance(lam, (list, tuple)):
        if len(lam) != n_layers:
            raise ValueError(f"lam sequence length {len(lam)} != n_layers {n_layers}")
        return [float(x) for x in lam]
    return [float(lam)] * n_layers


class TTFSNetTorch:
    """Exact TTFS (Time-to-First-Spike) SNN with vectorized grid-scan forward and IFT backward."""

    def __init__(self, sizes: List[int], tm: float = 15.0, ts: float = 4.0,
                 theta: float = 1.0, t_max: float = 40.0, w_scale: float = 0.2,
                 bias_val: float = 0.0, seed: int = 0, grid_pts: int = 4001,
                 dtype: torch.dtype = torch.float64, dev: Optional[torch.device] = None,
                 peak_tol: float = 1e-2, beta: float = 1.0,
                 max_spikes: int = 1) -> None:
        self.sizes = list(sizes)
        self.n_layers = len(sizes) - 1
        self.tm = float(tm)
        self.ts = float(ts)
        self.theta = float(theta)
        self.t_max = float(t_max)
        self.t_bias = 0.0
        self.beta = float(beta)
        self.dev = dev or device()
        self.dtype = dtype
        self.peak_tol = float(peak_tol)
        self.max_spikes = int(max_spikes)
        if abs(self.tm - self.ts) < 1e-9:
            self._alpha = True
            self.k_peak = 1.0
        else:
            self._alpha = False
            s = (self.tm * self.ts / (self.tm - self.ts)) * math.log(self.tm / self.ts)
            self.k_peak = float((math.exp(-s / self.tm) - math.exp(-s / self.ts)) / (self.tm - self.ts))
        self.grid = torch.linspace(0.0, self.t_max, int(grid_pts), dtype=dtype, device=self.dev)
        rng = np.random.default_rng(seed)
        self.W = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            w = (rng.standard_normal((b, a + 1)) * w_scale).astype(np.float64)
            w[:, -1] = bias_val
            self.W.append(torch.tensor(w, dtype=dtype, device=self.dev))
        self._init_local_machinery(sizes, w_scale, seed)
        self._cache = None

    def _init_local_machinery(self, sizes, w_scale, seed):
        """SP-04: per-hidden-layer readouts (trainable), random feedback matrices
        and a contrastive target projector (fixed). 'deep' mode trains the
        readouts; 'fa' and 'contrastive' are ablations."""
        n_out = sizes[-1]
        n_hid = len(sizes) - 2
        self.R = []
        self.B_fa = []
        self.P_cont = None
        if n_hid > 0:
            rng = np.random.default_rng(seed + 1000)
            for a in sizes[1:-1]:
                r = (rng.standard_normal((n_out, a)) * w_scale).astype(np.float64)
                self.R.append(torch.tensor(r, dtype=self.dtype, device=self.dev))
            for a in sizes[1:-1]:
                b = (rng.standard_normal((a, n_out)) * 0.5).astype(np.float64)
                self.B_fa.append(torch.tensor(b, dtype=self.dtype, device=self.dev))
            self.P_cont = torch.tensor(
                (rng.standard_normal((max(sizes[1:-1]), n_out)) * 0.5).astype(np.float64),
                dtype=self.dtype, device=self.dev)

    def _forward_layer(self, W, t_prev):
        """Per-layer first-spike-time solve. Overridden by the event-driven
        engine (engine/event_driven.py); this grid default is the verified one."""
        return forward_layer_torch(W, t_prev, self.t_bias, self.theta,
                                   self.grid, self.tm, self.ts, self._alpha,
                                   self.k_peak, peak_tol=self.peak_tol)

    def _peak_margin(self, W, t_prev):
        """Per-layer extremum (t_peak, u_peak) for the existence channel.
        Overridden by the event-driven engine."""
        return peak_margin_torch(W, t_prev, self.t_bias, self.theta,
                                 self.grid, self.tm, self.ts, self._alpha,
                                 self.k_peak)

    def _edge_peak_guard(self, W, t_prev, t_peak, u_peak):
        """Degenerate-plateau guard. Overridden by the event-driven engine."""
        return edge_peak_guard(W, t_prev, self.t_bias, t_peak, u_peak,
                               self.grid)

    def forward(self, t_in: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute first-spike times through all layers.

        Args:
            t_in: (n_in, B) input spike times.

        Returns:
            (n_out, B) output first-spike times (inf for silent neurons).
        """
        with torch.no_grad():
            t = t_in
            cache = []
            for l in range(self.n_layers):
                t_post, up = self._forward_layer(self.W[l], t)
                cache.append((t, t_post, up))
                t = t_post
            self._cache = cache
            return t

    def backward(self, dL_dt_out: torch.Tensor) -> List[Optional[torch.Tensor]]:
        """Backward pass: compute per-layer weight gradients via IFT.

        Args:
            dL_dt_out: (n_out, B) gradient of the loss w.r.t. output spike times.

        Returns:
            List of weight gradient tensors, one per layer, each of shape
            (n_cur, n_in+1). None entries are replaced during accumulation.
        """
        grads = [None] * self.n_layers
        lam = dL_dt_out
        for l in reversed(range(self.n_layers)):
            t_prev, t_post, up = self._cache[l]
            g, lam = backward_layer_torch(self.W[l], t_prev, self.t_bias,
                                           t_post, lam, up, self.tm, self.ts,
                                           self._alpha, self.k_peak)
            grads[l] = g
        return grads

    def backward_with_input_grad(self, dL_dt_out: torch.Tensor) -> Tuple[List[Optional[torch.Tensor]], torch.Tensor]:
        """Like backward(), but also returns the gradient flowing to the input.

        Args:
            dL_dt_out: (n_out, B) gradient of the loss w.r.t. output spike times.

        Returns:
            Tuple of (grads, lam) where grads is the per-layer weight gradient
            list and lam is (n_in, B) the adjoint into input spike times.
        """
        grads = [None] * self.n_layers
        lam = dL_dt_out
        for l in reversed(range(self.n_layers)):
            t_prev, t_post, up = self._cache[l]
            g, lam = backward_layer_torch(self.W[l], t_prev, self.t_bias,
                                           t_post, lam, up, self.tm, self.ts,
                                           self._alpha, self.k_peak)
            grads[l] = g
        return grads, lam

    def backward_saltation(self, dL_dt_out: torch.Tensor) -> List[Optional[torch.Tensor]]:
        """SP-03 saltation backward: exact weight gradients through resets.

        Uses multi-spike cache (from forward_multispike) when available,
        otherwise falls back to single-spike IFT.
        """
        grads = [None] * self.n_layers
        lam = dL_dt_out
        for l in reversed(range(self.n_layers)):
            t_prev, t_post, up = self._cache[l]
            t_all = getattr(self, '_cache_all', None)
            up_all = getattr(self, '_cache_up_all', None)
            t_all_l = t_all[l] if t_all is not None else None
            up_all_l = up_all[l] if up_all is not None else None
            g, lam = backward_layer_saltation(
                self.W[l], t_prev, self.t_bias, t_post, lam, up,
                self.tm, self.ts, self._alpha, self.k_peak,
                self.t_max, self.theta,
                t_all=t_all_l, up_all=up_all_l)
            grads[l] = g
        return grads

    def loss_and_grads_saltation(self, t_in: torch.Tensor, y: torch.Tensor) -> Tuple[float, List[Optional[torch.Tensor]], torch.Tensor]:
        """Multi-spike forward + latency CE + saltation backward.

        Uses forward_multispike to populate the multi-spike cache,
        then backward_saltation auto-dispatches to the exact
        sensitivity_first_spike path through all resets.
        """
        t_out = self.forward_multispike(t_in)
        loss, dL_dt_out = latency_cross_entropy(t_out, y, self.t_max, self.beta)
        grads = self.backward_saltation(dL_dt_out)
        return loss, grads, t_out

    def loss_and_grads(self, t_in: torch.Tensor, y: torch.Tensor) -> Tuple[float, List[Optional[torch.Tensor]], torch.Tensor]:
        """Forward pass, compute loss, and return gradients.

        Auto-dispatches to single-spike (max_spikes=1) or multi-spike
        (max_spikes>1) pipelines.

        Args:
            t_in: (n_in, B) input spike times.
            y: (B,) target class indices.

        Returns:
            Tuple of (loss, grads, t_out) where loss is a float, grads is the
            per-layer weight gradient list, and t_out is (n_out, B) output
            spike times.
        """
        if self.max_spikes > 1:
            return self.loss_and_grads_saltation(t_in, y)
        t_out = self.forward(t_in)
        loss, dL_dt_out = latency_cross_entropy(t_out, y, self.t_max, self.beta)
        grads = self.backward(dL_dt_out)
        return loss, grads, t_out

    def forward_multispike(self, t_in):
        """Multi-spike forward: simulates ALL spikes through ALL resets.

        Returns (t_out, t_all_cache, up_all_cache) where:
          t_out: (n_out, B) first-spike times (for loss computation)
          t_all_cache[l]: (n_cur, B, K) all spike times at layer l
          up_all_cache[l]: (n_cur, B, K) u' at each spike at layer l
        """
        t = t_in
        cache_first = []
        cache_all = []
        cache_up_all = []
        for l in range(self.n_layers):
            t_post, up, t_all, up_all = forward_multispike_layer_torch(
                self.W[l], t, self.t_bias, self.tm, self.ts, self.theta,
                self.k_peak, self.t_max, self.grid, max_spikes=20)
            cache_first.append((t, t_post, up))
            cache_all.append(t_all)
            cache_up_all.append(up_all)
            t = t_post
        self._cache = cache_first
        self._cache_all = cache_all
        self._cache_up_all = cache_up_all
        return t

    def backward_multispike(self, dL_dt_out):
        """Multi-spike backward: GPU-vectorized weight gradients through ALL resets.

        Uses backward_multispike_layer_torch for vectorized computation.
        First-spike adjoint (lam_prev) is computed from the first spike only.
        """
        grads = [None] * self.n_layers
        lam = dL_dt_out
        for l in reversed(range(self.n_layers)):
            t_prev, t_post, up = self._cache[l]
            t_all = self._cache_all[l]
            up_all = self._cache_up_all[l]
            g, lam = backward_multispike_layer_torch(
                self.W[l], t_prev, self.t_bias, t_all, up_all, lam,
                self.tm, self.ts, self.k_peak, self.t_max, self.theta)
            grads[l] = g
        return grads

    def loss_and_grads_multispike(self, t_in: torch.Tensor, y: torch.Tensor) -> Tuple[float, List[Optional[torch.Tensor]], torch.Tensor]:
        """Forward (multi-spike) + latency CE + multi-spike backward."""
        t_out = self.forward_multispike(t_in)
        loss, dL_dt_out = latency_cross_entropy(t_out, y, self.t_max, self.beta)
        grads = self.backward_multispike(dL_dt_out)
        return loss, grads, t_out

    def loss_and_grads_rate(self, t_in: torch.Tensor, y: torch.Tensor) -> Tuple[float, List[Optional[torch.Tensor]], torch.Tensor]:
        """Multi-spike forward + spike-count CE + multi-spike backward.

        Uses rate-based classification: output with the most spikes wins.
        """
        t_out = self.forward_multispike(t_in)
        cache = getattr(self, '_cache_all', None)
        t_out_all = cache[-1] if cache is not None else t_out.unsqueeze(-1)
        loss, dL_dc = spike_count_cross_entropy(t_out_all, y)
        dL_dt = torch.zeros_like(t_out)
        for b in range(t_out.shape[1]):
            dL_dt[:, b] = dL_dc[:, b] * (1.0 / (self.t_max + 1.0))
        grads = self.backward_multispike(dL_dt)
        return loss, grads, t_out

    def loss_and_grads_rate_latency(self, t_in: torch.Tensor, y: torch.Tensor) -> Tuple[float, List[Optional[torch.Tensor]], torch.Tensor]:
        """Multi-spike forward + combined rate-latency loss + multi-spike backward."""
        t_out = self.forward_multispike(t_in)
        cache = getattr(self, '_cache_all', None)
        t_out_all = cache[-1] if cache is not None else t_out.unsqueeze(-1)
        loss, dL_dt = rate_latency_loss(t_out_all, y, self.t_max, self.beta)
        grads = self.backward_multispike(dL_dt)
        return loss, grads, t_out

    def existence_grads(self, t_in: torch.Tensor, y: torch.Tensor, T_noise: float = 1.0,
                        lam: float = 1.0, hidden_target: float = 1.0,
                        correct_output_target: bool = True,
                        exclude: Optional[List[Optional[torch.Tensor]]] = None) -> Tuple[float, List[Optional[torch.Tensor]], dict]:
        """Forward + SP-02 existence channel. Returns (loss_total, grads, stats).

        grads[l] is the TOTAL per-layer gradient: SP-01 exact timing gradient
        plus the existence-channel gradient for silent neurons (isolated by
        construction: fired neurons are never targeted, and (1-p) -> 0 as a
        revived neuron approaches p -> 1, so the channel auto-decays).

        exclude: optional list of per-layer (n_cur, B) bool masks. Entries
        where exclude[l] is True are removed from the existence targets before
        computing loss/gradients -- used by validation to compare two engines
        over an IDENTICAL target set (entries excluded are exactly those where
        the reference engine's peak-margin pipeline is known to select a
        different extremum).

        Existence objective (escape-noise peak-margin model, SP-02 research doc):
            L_exist = -(lam/B) * sum over targeted silent j of log p_j
            p_j = sigmoid((u_peak_j - theta) / T_noise)
            dL_exist/d(u_peak_j) = -(lam/B) * (1 - p_j) / T_noise      (bounded
              below by -(lam/B)/T_noise for far-dead neurons)
            d(u_peak_j)/dW_ji = K(t_peak_j - t_i)                     (envelope thm)
        Targets: all silent hidden neurons; correct-class silent outputs only.
        """
        B = t_in.shape[1]
        n_layers = self.n_layers
        n_out = self.sizes[-1]
        dev = self.dev
        dtype = self.dtype
        t_out = self.forward(t_in)

        onehot = torch.zeros((n_out, B), dtype=torch.bool, device=dev)
        onehot[y, torch.arange(B, device=dev)] = True

        lam_l = _as_layer_lam(lam, n_layers)

        g = []          # dL_exist/d(u_peak) per layer, masked (n_cur, B)
        peaks = []      # (t_peak, u_peak) per layer
        silent_stats = []
        loss_exist = 0.0
        for l in range(n_layers):
            W = self.W[l]
            t_prev, t_post, up = self._cache[l]
            fired = torch.isfinite(t_post)
            n_cur_l = t_post.shape[0]
            if fired.all():
                g.append(torch.zeros((n_cur_l, B), dtype=dtype, device=dev))
                peaks.append((t_post, up))
                silent_stats.append({"n_silent": 0, "n_targeted": 0,
                                     "n_edge_guarded": 0})
                continue
            t_peak, u_peak = self._peak_margin(W, t_prev)
            if l == n_layers - 1 and correct_output_target:
                target = (~fired) & onehot
            else:
                target = (~fired) * (hidden_target != 0.0)
            target = target.to(dtype)
            guard = self._edge_peak_guard(W, t_prev, t_peak, u_peak)
            target = target * (~guard).to(dtype)
            if exclude is not None:
                target = target * (~exclude[l]).to(dtype)
            p = torch.sigmoid((u_peak - self.theta) / T_noise)
            loss_exist += (lam_l[l] / B) * float(
                (-target * torch.log(p.clamp(min=1e-12))).sum())
            g.append(-(lam_l[l] / B) * target * (1.0 - p) / T_noise)
            peaks.append((t_peak, u_peak))
            silent_stats.append({
                "n_silent": int((~fired).sum().item()),
                "n_targeted": int((target > 0).sum().item()),
                "n_edge_guarded": int(guard.sum().item()),
            })

        loss_t, lam_timing = latency_cross_entropy(t_out, y, self.t_max, self.beta)
        grads = [None] * n_layers
        lam = lam_timing
        lam_exist = None
        for l in reversed(range(n_layers)):
            W = self.W[l]
            t_prev, t_post, up = self._cache[l]
            n_in = W.shape[1] - 1
            lam_total = lam.clone()
            if lam_exist is not None:
                lam_total = lam_total + lam_exist
            g_timing, lam_prev = backward_layer_torch(W, t_prev, self.t_bias,
                                                       t_post, lam_total, up,
                                                       self.tm, self.ts,
                                                       self._alpha, self.k_peak)
            g_l = g[l]
            g_exist = torch.zeros_like(W)
            targeted = g_l != 0
            if targeted.any():
                t_peak_l, _ = peaks[l]
                g_exist[:, n_in] = (g_l * _K(t_peak_l - self.t_bias, self.tm,
                                             self.ts, self._alpha,
                                             self.k_peak)).sum(dim=1)
                for i in range(n_in):
                    d = t_peak_l - t_prev[i].view(1, -1)
                    g_exist[:, i] = (g_l * _K(d, self.tm, self.ts,
                                              self._alpha, self.k_peak)).sum(dim=1)
            grads[l] = g_timing + g_exist
            lam_exist = torch.zeros((n_in, B), dtype=dtype, device=dev)
            if targeted.any():
                t_peak_l, _ = peaks[l]
                for i in range(n_in):
                    d = t_peak_l - t_prev[i].view(1, -1)
                    lam_exist[i] = (g_l * W[:, i].view(-1, 1)
                                    * _Kd(d, self.tm, self.ts,
                                          self._alpha, self.k_peak)).sum(dim=0)
            lam = lam_prev

        stats = {
            "loss_timing": float(loss_t),
            "loss_exist": loss_exist,
            "silent_per_layer": silent_stats,
        }
        return float(loss_t) + loss_exist, grads, stats

    def _contrastive_signal(self, l, t_post, fired, onehot, tau):
        """TP-style layer-local contrastive signal (SP-04 mode 'contrastive').

        trace z_j = spike time (placeholder for silent). Target traces c = P @
        onehot (fixed projector P, rows sliced to the layer). Loss per sample b:
        CE over b' of softmax(dot(z_b, c_b')/tau) against the class-structured
        target softmax(dot(c_b, c_b')/tau) -- samples of the same class carry
        the mass. Returns dL/dz masked to fired neurons (the local learning
        signal for this layer); no backward pass, no readout classifier."""
        n_cur, B = t_post.shape
        dev = t_post.device
        dtype = t_post.dtype
        ph = torch.tensor(2.0 * self.t_max + 10.0, dtype=dtype, device=dev)
        z = torch.where(fired, t_post, ph)
        P = self.P_cont[:n_cur]
        c = P @ onehot.to(dtype)                  # (n_cur, B) target traces
        S = z.t() @ c / tau                       # (B, B) input-vs-target sims
        p = torch.softmax(S, dim=1)
        yt = torch.softmax(c.t() @ c / tau, dim=1)  # class-structured target
        self._last_contrast_loss = float((-yt * torch.log(p.clamp(min=1e-12)))
                                         .sum(dim=1).mean().item())
        M = (p - yt) / tau
        dLdz = c @ M.t()                          # (n_cur, B)
        return torch.where(fired, dLdz, torch.zeros_like(dLdz))

    def local_learning_grads(self, t_in: torch.Tensor, y: torch.Tensor,
                             T_noise: float = 1.0, lam: float = 1.0, mode: str = "deep",
                             hidden_target: float = 1.0, correct_output_target: bool = True,
                             contrast_tau: float = 1.0,
                             exclude: Optional[List[Optional[torch.Tensor]]] = None) -> Tuple[float, List[Optional[torch.Tensor]], Optional[List[Optional[torch.Tensor]]], dict]:
        """SP-04: per-layer local learning signals + the SP-02 existence channel.

        Modes (see docs/research/SP-04-research.md Section 5/7):
          'deep'       (mechanism A, CHOSEN): each hidden layer is trained by
                       its own readout (latency-CE on R_l @ t_post_l) plus its
                       own existence channel; the output layer by its own CE.
                       NO cross-layer error at all: no W^T transport, no adjoint
                       between layers. Returns readout grads (trainable).
          'fa'         (ablation B): output CE at the output; hidden timing
                       signal = fixed random feedback B_fa_l @ lam_out (DFA).
                       Removes W^T; keeps the global error signal and the
                       cross-layer existence adjoint (clean ablation vs 'ref').
          'contrastive'(ablation D): output CE at the output; hidden timing
                       signal from the layer-local contrastive loss. Forward
                       only, no readout, no cross-layer adjoint.
          'ref'        exact W^T backward (== existence_grads), the reference.

        Returns (loss, grads_W, grads_R, stats). grads_R is a list aligned to
        self.R (None unless mode == 'deep').
        """
        B = t_in.shape[1]
        n_layers = self.n_layers
        n_out = self.sizes[-1]
        dev = self.dev
        dtype = self.dtype
        t_out = self.forward(t_in)

        onehot = torch.zeros((n_out, B), dtype=torch.bool, device=dev)
        onehot[y, torch.arange(B, device=dev)] = True

        if mode == "ref":
            loss_tot, grads, stats = self.existence_grads(
                t_in, y, T_noise=T_noise, lam=lam,
                hidden_target=hidden_target, correct_output_target=correct_output_target)
            return loss_tot, grads, None, stats

        # --- shared: SP-02 existence channel (targets + peak margin) ---
        lam_l = _as_layer_lam(lam, n_layers)
        g = []
        peaks = []
        silent_stats = []
        loss_exist = 0.0
        for l in range(n_layers):
            W = self.W[l]
            t_prev, t_post, up = self._cache[l]
            fired = torch.isfinite(t_post)
            n_cur_l = t_post.shape[0]
            if fired.all():
                g.append(torch.zeros((n_cur_l, B), dtype=dtype, device=dev))
                peaks.append((t_post, up))
                silent_stats.append({"n_silent": 0, "n_targeted": 0,
                                     "n_edge_guarded": 0})
                continue
            t_peak, u_peak = self._peak_margin(W, t_prev)
            if l == n_layers - 1 and correct_output_target:
                target = (~fired) & onehot
            else:
                target = (~fired) * (hidden_target != 0.0)
            target = target.to(dtype)
            guard = self._edge_peak_guard(W, t_prev, t_peak, u_peak)
            target = target * (~guard).to(dtype)
            if exclude is not None:
                target = target * (~exclude[l]).to(dtype)
            p = torch.sigmoid((u_peak - self.theta) / T_noise)
            loss_exist += (lam_l[l] / B) * float(
                (-target * torch.log(p.clamp(min=1e-12))).sum())
            g.append(-(lam_l[l] / B) * target * (1.0 - p) / T_noise)
            peaks.append((t_peak, u_peak))
            silent_stats.append({
                "n_silent": int((~fired).sum().item()),
                "n_targeted": int((target > 0).sum().item()),
                "n_edge_guarded": int(guard.sum().item()),
            })

        loss_t, lam_out = latency_cross_entropy(t_out, y, self.t_max, self.beta)

        # --- per-hidden-layer timing signal ---
        lam_sig = [None] * n_layers
        grads_R = [None] * len(self.R)
        loss_aux = 0.0
        for l in range(n_layers - 1):
            t_prev, t_post, up = self._cache[l]
            n_cur = t_post.shape[0]
            fired = torch.isfinite(t_post)
            if mode == "deep":
                r = self.R[l]
                t_eff = torch.where(fired, t_post, torch.zeros_like(t_post))
                l_l, dLa = latency_cross_entropy(r @ t_eff, y, self.t_max, self.beta)
                loss_aux += l_l
                grads_R[l] = dLa @ t_eff.t()
                lam_sig[l] = torch.where(fired, r.t() @ dLa, torch.zeros_like(t_post))
            elif mode == "fa":
                lam_sig[l] = torch.where(fired, self.B_fa[l] @ lam_out,
                                         torch.zeros_like(t_post))
            else:  # contrastive
                lam_sig[l] = self._contrastive_signal(l, t_post, fired, onehot,
                                                      contrast_tau)
                loss_aux += self._last_contrast_loss
        lam_sig[n_layers - 1] = lam_out

        # --- backward: each layer from ITS OWN signal; no W^T transport ---
        include_exist_adjoint = (mode == "fa")
        grads = [None] * n_layers
        lam_exist = None
        for l in reversed(range(n_layers)):
            W = self.W[l]
            t_prev, t_post, up = self._cache[l]
            n_in = W.shape[1] - 1
            lam_total = lam_sig[l].clone()
            if include_exist_adjoint and lam_exist is not None:
                lam_total = lam_total + lam_exist
            g_timing, _lam_prev = backward_layer_torch(W, t_prev, self.t_bias,
                                                       t_post, lam_total, up,
                                                       self.tm, self.ts,
                                                       self._alpha, self.k_peak)
            g_exist, lam_exist = _existence_layer_grads(
                W, g[l], peaks[l][0], t_prev, self.t_bias, self.tm, self.ts,
                self._alpha, self.k_peak)
            if not include_exist_adjoint:
                lam_exist = None
            grads[l] = g_timing + g_exist

        stats = {
            "loss_timing": float(loss_t),
            "loss_exist": loss_exist,
            "loss_aux": float(loss_aux),
            "silent_per_layer": silent_stats,
        }
        return (float(loss_t) + loss_exist + loss_aux, grads,
                grads_R if mode == "deep" else None, stats)
