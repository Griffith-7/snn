"""SP-06: event-driven exact engine (no dense grid scan).

Same math as the verified grid engine (engine/snn_torch.py); the per-layer
spike-time solve and the existence-channel extremum are computed from the
INTER-EVENT CLOSED-FORM structure of the double-exponential kernel instead of
a 1001..4001-point uniform scan.

Structure: between consecutive input events the set of active presynaptic
kernels is fixed, so the membrane potential is a sum of TWO exponentials,

    u(t) = A * exp(-t/tm) + B * exp(-t/ts),

with A, B constant on each inter-event interval (prefix sums of the sorted
event times, times the per-neuron weights). A sum of two exponentials has at
most one critical point, in CLOSED FORM:

    u'(t) = 0  =>  t* = (tm*ts/(tm-ts)) * ln(-B*tm/(A*ts))      (A*B < 0),

so each interval's crossing structure is decided analytically and the first
root is polished with vectorized bisection + clamped Newton. The existence
channel's peak is the exact max/min over {interior critical points} U {window
boundaries} -- no golden-section refinement at all.

The gradient side (backward_layer_torch, existence grads, edge_peak_guard) is
already analytic in the grid engine and is reused unchanged. Cross-validation
of fire times, peaks, silent sets, and gradients against the grid engine is in
engine/experiments/exp_event_driven.py.

alpha kernels (tm == ts) are not handled by the closed form and fall back to
the grid engine (never used by the repo's experiments, tm=15, ts=4).
"""
import torch

from exact_snn.core import TTFSNetTorch, _du_at, _u_at

N_BISECT = 20
N_NEWTON = 8


def _interval_coeffs(
    W: torch.Tensor,
    t_prev: torch.Tensor,
    t_bias: float,
    t_max: float,
    tm: float,
    ts: float,
    k_peak: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interval decomposition of u(t).

    Sorted per sample (dim 0), with ties allowed: interval k spans
    (lo[k], hi[k]), k = 0..n_in, where lo = [t_bias, tsorted_0, ..., tsorted_{n-1}]
    and hi = [tsorted_0, ..., tsorted_{n-1}, t_max]. Its coefficients are the
    bias plus the prefix sum of the first k sorted inputs -- i.e. exactly the
    set of kernels active for t in (lo[k], hi[k]). Zero-length intervals (ties)
    are harmless: they contribute no interior root/extremum.

    Returns A, Bv (n_cur, n_in+1, B) with u(t) = A*e^{-t/tm} + B*e^{-t/ts},
    and lo, hi (n_in+1, B).
    """
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_prev.shape[1]
    dev = W.device
    dtype = W.dtype
    denom = (tm - ts) * k_peak
    tsorted, perm = torch.sort(t_prev, dim=0)  # (n_in, B), per-sample
    wb = W[:, n_in]                            # (n_cur,)
    A_bias = wb / denom
    B_bias = -wb / denom
    zero = torch.zeros((n_cur, 1, B), dtype=dtype, device=dev)
    if n_in:
        index = perm.unsqueeze(0).expand(n_cur, n_in, B)
        wg = W[:, :n_in].unsqueeze(-1).expand(n_cur, n_in, B).gather(1, index)
        ts_u = tsorted.unsqueeze(0)
        A_terms = wg * torch.exp(ts_u / tm) / denom
        B_terms = -wg * torch.exp(ts_u / ts) / denom
        A_cs = torch.cumsum(A_terms, dim=1)
        B_cs = torch.cumsum(B_terms, dim=1)
        A_int = A_bias.view(n_cur, 1, 1) + torch.cat([zero, A_cs], dim=1)
        B_int = B_bias.view(n_cur, 1, 1) + torch.cat([zero, B_cs], dim=1)
        lo = torch.cat([torch.full((1, B), t_bias, dtype=dtype, device=dev),
                        tsorted[:-1]], dim=0)
        lo = torch.cat([lo, tsorted[-1:]], dim=0)
        hi = torch.cat([tsorted,
                        torch.full((1, B), t_max, dtype=dtype, device=dev)],
                       dim=0)
    else:
        A_int = A_bias.view(n_cur, 1, 1)
        B_int = B_bias.view(n_cur, 1, 1)
        lo = torch.full((1, B), t_bias, dtype=dtype, device=dev)
        hi = torch.full((1, B), t_max, dtype=dtype, device=dev)
    return A_int, B_int, lo, hi


def forward_layer_event(
    W: torch.Tensor,
    t_prev: torch.Tensor,
    t_bias: float,
    theta: float,
    t_max: float,
    tm: float,
    ts: float,
    k_peak: float,
    n_bisect: int = N_BISECT,
    n_newton: int = N_NEWTON,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact first-crossing times, event-driven. Returns (t_post, up).

    t_post (n_cur, B): first t in [0, t_max] with u(t) >= theta (inf if none);
    up (n_cur, B): du/dt at t_post, evaluated with the SAME causal analytic
    kernels as the grid engine (engine/snn_torch._du_at).
    """
    n_cur, n_inp = W.shape
    B = t_prev.shape[1]
    dev = W.device
    dtype = W.dtype
    inf = float("inf")

    A, Bv, lo, hi = _interval_coeffs(W, t_prev, t_bias, t_max, tm, ts, k_peak)
    lo_u = lo.unsqueeze(0)   # (1, n_in+1, B)
    hi_u = hi.unsqueeze(0)

    def u_val(a, b, t):
        return a * torch.exp(-t / tm) + b * torch.exp(-t / ts)

    def up_val(a, b, t):
        return -a / tm * torch.exp(-t / tm) - b / ts * torch.exp(-t / ts)

    crit = (A * Bv < 0) & (A != 0)
    disc = -Bv * tm / (A * ts)
    tstar = torch.where(crit, (tm * ts / (tm - ts)) * torch.log(disc),
                        torch.zeros_like(A))
    crit_in = crit & (tstar > lo_u) & (tstar < hi_u)
    u_star = u_val(A, Bv, tstar)

    u_lo = u_val(A, Bv, lo_u)
    u_hi = u_val(A, Bv, hi_u)
    f_lo = u_lo - theta
    f_hi = u_hi - theta

    fired_at_start = f_lo >= 0.0
    start_time = torch.where(fired_at_start, lo_u,
                             torch.full_like(lo_u, inf))

    root_exists = (f_hi >= 0.0) | (crit_in & (u_star >= theta))
    root_exists = root_exists & (f_lo < 0.0)
    b_lo = torch.where(crit_in & (u_star >= theta), lo_u,
                       torch.where(crit_in, tstar, lo_u))
    b_hi = torch.where(crit_in & (u_star >= theta), tstar, hi_u)

    a = torch.where(root_exists, b_lo, torch.zeros_like(b_lo))
    b = torch.where(root_exists, b_hi, torch.zeros_like(b_hi))
    fa = u_val(A, Bv, a) - theta
    fb = u_val(A, Bv, b) - theta
    for _ in range(n_bisect):
        m = 0.5 * (a + b)
        fm = u_val(A, Bv, m) - theta
        take_left = fa * fm <= 0.0
        b = torch.where(take_left, m, b)
        fb = torch.where(take_left, fm, fb)
        a = torch.where(take_left, a, m)
        fa = torch.where(take_left, fa, fm)
    m = 0.5 * (a + b)
    for _ in range(n_newton):
        um = u_val(A, Bv, m) - theta
        dum = up_val(A, Bv, m)
        safe = dum > 1e-10
        nm = m - um / torch.where(safe, dum, torch.ones_like(dum))
        nm = torch.clamp(nm, min=a, max=b)
        m = torch.where(safe, nm, m)

    root_time = torch.where(root_exists, m, torch.full_like(m, inf))
    t_cand = torch.minimum(root_time, start_time)      # (n_cur, n_in+1, B)
    t_cand = t_cand.min(dim=1).values                  # earliest over intervals
    t_post = torch.where(torch.isfinite(t_cand), t_cand,
                         torch.full_like(t_cand, inf))

    up = _du_at(W, t_prev, t_bias, tm, ts, False, k_peak, t_post)
    return t_post, up


def peak_margin_event(
    W: torch.Tensor,
    t_prev: torch.Tensor,
    t_bias: float,
    theta: float,
    t_max: float,
    tm: float,
    ts: float,
    k_peak: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact extremum (t_peak, u_peak) over the response window [t_start, t_max].

    The maximum (or, for an all-negative response, the minimum) of a
    piecewise-2-exponential function over a window lies among {interior
    critical points} U {window boundaries}, all in closed form. Same selection
    rule and same fired-marker convention as peak_margin_torch.
    """
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_prev.shape[1]
    dev = W.device
    dtype = W.dtype
    inf = float("inf")

    A, Bv, lo, hi = _interval_coeffs(W, t_prev, t_bias, t_max, tm, ts, k_peak)
    lo_u = lo.unsqueeze(0)
    hi_u = hi.unsqueeze(0)

    crit = (A * Bv < 0) & (A != 0)
    disc = -Bv * tm / (A * ts)
    tstar = torch.where(crit, (tm * ts / (tm - ts)) * torch.log(disc),
                        torch.zeros_like(A))
    u_star = (A * torch.exp(-tstar / tm) + Bv * torch.exp(-tstar / ts))

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
    ts_u = t_start.unsqueeze(1)                       # (n_cur, 1, B)
    tmax_bc = torch.full((n_cur, 1, B), t_max, dtype=dtype, device=dev)

    in_win = (crit & (tstar > lo_u) & (tstar < hi_u)
              & (tstar > ts_u) & (tstar < tmax_bc))

    u_tmax = _u_at(W, t_prev, t_bias, tm, ts, False, k_peak,
                   tmax_bc.squeeze(1)).unsqueeze(1)   # (n_cur, 1, B)

    cand_t = torch.cat([tstar, ts_u, tmax_bc], dim=1)
    cand_v = torch.cat([u_star,
                        torch.zeros_like(ts_u),
                        u_tmax], dim=1)
    valid = torch.cat([in_win,
                       torch.ones_like(ts_u, dtype=torch.bool),
                       torch.ones_like(tmax_bc, dtype=torch.bool)], dim=1)

    if n_in:
        # Extrema can also sit at a KINK (an input event time), where u is
        # continuous but u' is not. u at event k uses interval-k coefficients
        # (inputs strictly before it; K(0) = 0, so the event itself contributes
        # nothing at its own time). The grid engine samples every grid point,
        # which is how it sees these; include them exactly here.
        tsorted, _ = torch.sort(t_prev, dim=0)
        ts_ev = torch.cat([tsorted.unsqueeze(0),
                           torch.full((1, 1, B), inf, dtype=dtype, device=dev)],
                          dim=1)                       # (1, n_in+1, B), dummy pad
        u_ev = torch.cat([
            (A[:, :n_in, :] * torch.exp(-tsorted.unsqueeze(0) / tm)
             + Bv[:, :n_in, :] * torch.exp(-tsorted.unsqueeze(0) / ts)),
            torch.zeros((n_cur, 1, B), dtype=dtype, device=dev)], dim=1)
        ev_valid = torch.cat([
            (tsorted.unsqueeze(0) >= ts_u) & (tsorted.unsqueeze(0) <= tmax_bc),
            torch.zeros((n_cur, 1, B), dtype=torch.bool, device=dev)], dim=1)
        ts_ev = ts_ev.expand(n_cur, -1, -1)
        cand_t = torch.cat([tstar, ts_ev, ts_u, tmax_bc], dim=1)
        cand_v = torch.cat([u_star, u_ev,
                            torch.zeros_like(ts_u),
                            u_tmax], dim=1)
        valid = torch.cat([in_win, ev_valid,
                           torch.ones_like(ts_u, dtype=torch.bool),
                           torch.ones_like(tmax_bc, dtype=torch.bool)], dim=1)
    v_max = torch.where(valid, cand_v, torch.full_like(cand_v, -inf))
    max_v = v_max.max(dim=1).values
    all_neg = max_v <= 0.0
    v_min = torch.where(valid, cand_v, torch.full_like(cand_v, inf))
    min_v = v_min.min(dim=1).values

    u_peak = torch.where(all_neg, min_v, max_v)
    # For degenerate ties (e.g. the u==0 plateau, where every point attains the
    # extremum) pick the EARLIEST time among candidates at the extremum value --
    # the window start -- matching the grid engine's first-argmax semantics and
    # the edge_peak_guard's t_start convention.
    best_v = u_peak.unsqueeze(1)
    match = valid & (cand_v == best_v)
    t_peak = torch.where(match, cand_t,
                         torch.full_like(cand_t, inf)).min(dim=1).values
    fired = u_peak >= theta
    t_peak = torch.where(fired, torch.full_like(t_peak, inf), t_peak)
    u_peak = torch.where(fired, torch.zeros_like(u_peak), u_peak)
    return t_peak, u_peak


def edge_peak_guard_event(
    W: torch.Tensor,
    t_prev: torch.Tensor,
    t_bias: float,
    t_peak: torch.Tensor,
    u_peak: torch.Tensor,
    t_max: float,
    w_cut: float = 1e-9,
    u_cut: float = 1e-6,
    n_inferred: int = 4001,
) -> torch.Tensor:
    """Grid-free degenerate-plateau guard for the event-driven engine.

    Same logic as edge_peak_guard but takes a step-size estimate
    (t_max / n_inferred) instead of requiring the actual grid tensor.
    Flags neurons where the peak sits at the window start with near-zero
    potential (degenerate plateau → zero existence gradient).
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
    earliest_idx = masked.argmin(dim=2)
    earliest_w = ev_w.gather(1, earliest_idx)
    step = t_max / n_inferred
    at_start = t_peak <= t_start + 1.5 * step
    flat = (u_peak.abs() < u_cut) & at_start
    flippable = (earliest_w <= w_cut) & at_start
    return flat | flippable


class EventTTFSNet(TTFSNetTorch):
    """Event-driven exact engine. Drop-in for TTFSNetTorch (same weights, same
    seeds, same interface); alpha kernels fall back to the grid engine."""

    def _forward_layer(self, W: torch.Tensor, t_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._alpha:
            return super()._forward_layer(W, t_prev)
        return forward_layer_event(W, t_prev, self.t_bias, self.theta,
                                   self.t_max, self.tm, self.ts, self.k_peak)

    def _peak_margin(self, W: torch.Tensor, t_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._alpha:
            return super()._peak_margin(W, t_prev)
        return peak_margin_event(W, t_prev, self.t_bias, self.theta,
                                 self.t_max, self.tm, self.ts, self.k_peak)

    def _edge_peak_guard(self, W: torch.Tensor, t_prev: torch.Tensor,
                         t_peak: torch.Tensor, u_peak: torch.Tensor) -> torch.Tensor:
        if self._alpha:
            return super()._edge_peak_guard(W, t_prev, t_peak, u_peak)
        return edge_peak_guard_event(W, t_prev, self.t_bias, t_peak, u_peak,
                                     self.t_max)
