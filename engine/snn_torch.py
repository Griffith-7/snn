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

import numpy as np
import torch

from losses_torch import latency_cross_entropy
from reset_lif import ResetLIF


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
        D = t.unsqueeze(-1) - t_in.t().unsqueeze(0)  # (n_cur, B, n_in)
        u = u + (_K(D, tm, ts, alpha, k_peak) * W[:, :n_in].unsqueeze(1)).sum(-1)
    return u


def _du_at(W, t_in, t_bias, tm, ts, alpha, k_peak, t):
    n_in = W.shape[1] - 1
    du = W[:, n_in].view(-1, 1) * _Kd(t - t_bias, tm, ts, alpha, k_peak)
    if n_in:
        D = t.unsqueeze(-1) - t_in.t().unsqueeze(0)
        du = du + (_Kd(D, tm, ts, alpha, k_peak) * W[:, :n_in].unsqueeze(1)).sum(-1)
    return du


def forward_layer_torch(W, t_prev, t_bias, theta, grid, tm, ts, alpha, k_peak,
                        n_bisect=15, n_newton=8, peak_tol=1e-2):
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
    U = torch.zeros((n_cur, B, G), dtype=dtype, device=dev)
    for i in range(n_in):
        d = g - t_prev[i].view(1, -1, 1)
        U += W[:, i].view(n_cur, 1, 1) * _K(d, tm, ts, alpha, k_peak)
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


def backward_layer_torch(W, t_prev, t_bias, t_post, lam_post, up, tm, ts, alpha, k_peak):
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
    for i in range(n_in):
        d = t_post - t_prev[i].view(1, -1)
        grad[:, i] = -(adj * _K(d, tm, ts, alpha, k_peak)).sum(dim=1)
        lam_prev[i, :] = (adj * W[:, i].view(-1, 1) * _Kd(d, tm, ts, alpha, k_peak)).sum(dim=0)
    return grad, lam_prev


def backward_layer_saltation(W, t_prev, t_bias, t_post, lam_post, up, tm, ts,
                              alpha, k_peak, t_max, theta=1.0,
                              t_all=None, up_all=None):
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


def forward_multispike_layer(W, t_prev, t_bias, tm, ts, theta, k_peak,
                              t_max, max_spikes=20):
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


def backward_multispike_layer(W, t_prev, t_bias, t_all, up_all, lam_post,
                               tm, ts, k_peak, t_max, theta):
    """Multi-spike backward using ResetLIF.sensitivity_all.

    Computes exact weight gradients through ALL resets using the saltation
    matrix. Input-time adjoint uses the IFT formula on the first spike
    (exact for first-spike TTFS loss).

    Args:
        W: (n_cur, n_in+1)
        t_prev: (n_in, B) input spike times
        t_all: (n_cur, B, K) all output spike times (from forward_multispike)
        up_all: (n_cur, B, K) u' at each spike
        lam_post: (n_cur, B) = dL/dt_first_spike (from loss)

    Returns:
        grad: (n_cur, n_in+1) weight gradient
        lam_prev: (n_in, B) adjoint into previous layer's spike times
    """
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_all.shape[1]
    K = t_all.shape[2]
    dev = W.device
    dtype = W.dtype

    grad = torch.zeros_like(W)
    lam_prev = torch.zeros((n_in, B), dtype=dtype, device=dev)

    lif = ResetLIF(tm=tm, ts=ts, theta=theta * ts * k_peak)

    W_np = W.detach().cpu().numpy().astype(np.float64)
    t_prev_np = t_prev.detach().cpu().numpy().astype(np.float64)
    lam_np = lam_post.detach().cpu().numpy().astype(np.float64)
    t_all_np = t_all.detach().cpu().numpy().astype(np.float64)
    up_all_np = up_all.detach().cpu().numpy().astype(np.float64)

    grad_np = np.zeros_like(W_np)
    lam_prev_np = np.zeros((n_in, B), dtype=np.float64)

    for j in range(n_cur):
        for b in range(B):
            lam_jb = lam_np[j, b]
            if lam_jb == 0.0:
                continue
            if not np.isfinite(t_all_np[j, b, 0]):
                continue

            inputs = []
            for i in range(n_in):
                tv = float(t_prev_np[i, b]) if np.isfinite(t_prev_np[i, b]) else t_max
                inputs.append((tv, float(W_np[j, i])))
            inputs.append((0.0, float(W_np[j, n_in])))

            fires, dtdw_matrix = lif.sensitivity_all(inputs, t_end=t_max)

            n_fires = len(fires)
            dL_dt_first = lam_jb
            if n_fires > 0 and abs(up_all_np[j, b, 0]) > 1e-12:
                dtdw_first = np.array(dtdw_matrix[0], dtype=np.float64)
                finite = np.isfinite(dtdw_first)
                grad_np[j, finite] += dL_dt_first * dtdw_first[finite]

            td_first = fires[0] - t_prev_np[:, b] if n_fires > 0 else np.full(n_in, np.inf)
            valid_first = (np.isfinite(td_first) & (td_first > 0) &
                           np.isfinite(t_prev_np[:, b]))
            if n_fires > 0 and abs(up_all_np[j, b, 0]) > 1e-12:
                d = np.where(valid_first, td_first, 0.0)
                tm_val, ts_val = lif.tm, lif.ts
                kd = (-np.exp(-d / tm_val) / tm_val +
                      np.exp(-d / ts_val) / ts_val) / (tm_val - ts_val) / k_peak
                dt_dtin = W_np[j, :n_in] * kd / up_all_np[j, b, 0]
                lam_prev_np[:, b] += dL_dt_first * dt_dtin * valid_first

    grad = torch.tensor(grad_np, dtype=dtype, device=dev)
    lam_prev = torch.tensor(lam_prev_np, dtype=dtype, device=dev)
    return grad, lam_prev


def peak_margin_torch(W, t_prev, t_bias, theta, grid, tm, ts, alpha, k_peak):
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


def edge_peak_guard(W, t_prev, t_bias, t_peak, u_peak, grid, w_cut=1e-9,
                    edge_cells=1.5, u_cut=1e-6):
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
    def __init__(self, sizes, tm=15.0, ts=4.0, theta=1.0, t_max=40.0,
                 w_scale=0.2, bias_val=0.0, seed=0, grid_pts=4001,
                 dtype=torch.float64, dev=None, peak_tol=1e-2, beta=1.0):
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

    def forward(self, t_in):
        t = t_in
        cache = []
        for l in range(self.n_layers):
            t_post, up = self._forward_layer(self.W[l], t)
            cache.append((t, t_post, up))
            t = t_post
        self._cache = cache
        return t

    def backward(self, dL_dt_out):
        grads = [None] * self.n_layers
        lam = dL_dt_out
        for l in reversed(range(self.n_layers)):
            t_prev, t_post, up = self._cache[l]
            g, lam = backward_layer_torch(self.W[l], t_prev, self.t_bias,
                                           t_post, lam, up, self.tm, self.ts,
                                           self._alpha, self.k_peak)
            grads[l] = g
        return grads

    def backward_saltation(self, dL_dt_out):
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

    def loss_and_grads_saltation(self, t_in, y):
        """Forward + latency CE + SP-03 saltation backward."""
        t_out = self.forward(t_in)
        loss, dL_dt_out = latency_cross_entropy(t_out, y, self.t_max, self.beta)
        grads = self.backward_saltation(dL_dt_out)
        return loss, grads, t_out

    def loss_and_grads(self, t_in, y):
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
            t_post, up, t_all, up_all = forward_multispike_layer(
                self.W[l], t, self.t_bias, self.tm, self.ts, self.theta,
                self.k_peak, self.t_max, max_spikes=20)
            cache_first.append((t, t_post, up))
            cache_all.append(t_all)
            cache_up_all.append(up_all)
            t = t_post
        self._cache = cache_first
        self._cache_all = cache_all
        self._cache_up_all = cache_up_all
        return t

    def backward_multispike(self, dL_dt_out):
        """Multi-spike backward: exact weight gradients through ALL resets.

        Uses ResetLIF.sensitivity_all for weight gradients (saltation matrix
        at every reset) and IFT formula for input-time adjoint (exact for
        first-spike TTFS loss).
        """
        grads = [None] * self.n_layers
        lam = dL_dt_out
        for l in reversed(range(self.n_layers)):
            t_prev, t_post, up = self._cache[l]
            t_all = self._cache_all[l]
            up_all = self._cache_up_all[l]
            g, lam = backward_multispike_layer(
                self.W[l], t_prev, self.t_bias, t_all, up_all, lam,
                self.tm, self.ts, self.k_peak, self.t_max, self.theta)
            grads[l] = g
        return grads

    def loss_and_grads_multispike(self, t_in, y):
        """Forward (multi-spike) + latency CE + multi-spike backward."""
        t_out = self.forward_multispike(t_in)
        loss, dL_dt_out = latency_cross_entropy(t_out, y, self.t_max, self.beta)
        grads = self.backward_multispike(dL_dt_out)
        return loss, grads, t_out

    def existence_grads(self, t_in, y, T_noise=1.0, lam=1.0,
                        hidden_target=1.0, correct_output_target=True,
                        exclude=None):
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
            t_peak_l, _ = peaks[l]
            g_exist = torch.zeros_like(W)
            targeted = g_l != 0
            if targeted.any():
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

    def local_learning_grads(self, t_in, y, T_noise=1.0, lam=1.0, mode="deep",
                             hidden_target=1.0, correct_output_target=True,
                             contrast_tau=1.0, exclude=None):
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
