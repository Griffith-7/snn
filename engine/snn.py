"""SP-01 minimal exact TTFS engine.

Implements:
  - normalized double-exponential PSP kernel K(d), K'(d)
  - exact first-spike-time root finding (grid bracket + Newton polish)
  - exact IFT gradients:  dt_f/dw = -K(t_f - t_in) / u'(t_f)
  - adjoint propagation across layers
  - latency cross-entropy loss (TTFS)
  - Adam optimizer

Correctness is validated by experiments/exp_sp01.py against finite differences.
"""
import numpy as np

from losses import latency_cross_entropy


class DoubleExpKernel:
    """K(d) = (e^{-d/tm} - e^{-d/ts}) / (tm - ts) / k_peak, causal (0 for d <= 0)."""

    def __init__(self, tm, ts):
        self.tm = float(tm)
        self.ts = float(ts)
        if abs(self.tm - self.ts) < 1e-9:
            self._alpha = True
            self.k_peak = 1.0
        else:
            self._alpha = False
            s = (self.tm * self.ts / (self.tm - self.ts)) * np.log(self.tm / self.ts)
            # peak of the raw double-exponential INCLUDING the (tm-ts) factor,
            # so that max_d K(d) == 1.0
            self.k_peak = float((np.exp(-s / self.tm) - np.exp(-s / self.ts)) / (self.tm - self.ts))

    def K(self, d):
        d = np.asarray(d, dtype=float)
        out = np.zeros_like(d)
        m = d > 0
        if not np.any(m):
            return out
        if self._alpha:
            out[m] = (d[m] / self.tm) * np.exp(1.0 - d[m] / self.tm) / self.k_peak
        else:
            out[m] = (np.exp(-d[m] / self.tm) - np.exp(-d[m] / self.ts)) / (self.tm - self.ts) / self.k_peak
        return out

    def Kd(self, d):
        d = np.asarray(d, dtype=float)
        out = np.zeros_like(d)
        m = d > 0
        if not np.any(m):
            return out
        if self._alpha:
            out[m] = (1.0 - d[m] / self.tm) * np.exp(1.0 - d[m] / self.tm) / (self.tm * self.k_peak)
        else:
            out[m] = (-(1.0 / self.tm) * np.exp(-d[m] / self.tm)
                      + (1.0 / self.ts) * np.exp(-d[m] / self.ts)) / (self.tm - self.ts) / self.k_peak
        return out


def _refine_peak(a, b, u_fun, tol=1e-10):
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    for _ in range(90):
        if abs(b - a) < tol:
            break
        if u_fun(c) > u_fun(d):
            b = d
        else:
            a = c
        c = b - gr * (b - a)
        d = a + gr * (b - a)
    return 0.5 * (a + b)


def _bisect_newton(a, b, u_fun, du_fun, theta, rtol=1e-13):
    fa = u_fun(a) - theta
    fb = u_fun(b) - theta
    if fa > 0.0:
        return a
    if fb < 0.0:
        return np.inf
    for _ in range(100):
        m = 0.5 * (a + b)
        if abs(b - a) < rtol:
            break
        fm = u_fun(m) - theta
        if fa * fm <= 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    m = 0.5 * (a + b)
    for _ in range(8):
        du = du_fun(m)
        if du > 1e-10:
            nm = m - (u_fun(m) - theta) / du
            if a < nm < b:
                m = nm
    return m


def _first_fire(u_grid, grid, u_fun, du_fun, theta):
    mask = u_grid >= theta
    if np.any(mask):
        idx = int(np.argmax(mask))
        if idx == 0:
            return grid[0], du_fun(grid[0])
        tf = _bisect_newton(grid[idx - 1], grid[idx], u_fun, du_fun, theta)
        if np.isfinite(tf):
            return tf, du_fun(tf)
        return np.inf, 0.0
    imax = int(np.argmax(u_grid))
    lo = grid[max(imax - 1, 0)]
    hi = grid[min(imax + 1, len(grid) - 1)]
    t_peak = _refine_peak(lo, hi, u_fun)
    if u_fun(t_peak) >= theta:
        tf = _bisect_newton(0.0, t_peak, u_fun, du_fun, theta)
        if np.isfinite(tf):
            return tf, du_fun(tf)
    return np.inf, 0.0


def forward_layer_batch(W, t_prev, t_bias, theta, t_max, k, grid):
    """W: (n_cur, n_in+1), bias is the last column. t_prev: (n_in, B). inf = silent input.

    Returns t_post (n_cur, B) firing times (inf = silent), up (n_cur, B) = du/dt at firing.
    """
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_prev.shape[1]
    t_post = np.full((n_cur, B), np.inf)
    up = np.zeros((n_cur, B))
    for b in range(B):
        t_in = t_prev[:, b]
        U = np.zeros((n_cur, len(grid)))
        for i in range(n_in):
            if np.isfinite(t_in[i]):
                U += W[:, i:i + 1] * k.K(grid - t_in[i])[None, :]
        U += W[:, n_in:n_in + 1] * k.K(grid - t_bias)[None, :]
        for j in range(n_cur):
            wj = W[j]
            def u_fun(t):
                return float(np.dot(wj[:n_in], k.K(t - t_in)) + wj[n_in] * float(k.K(t - t_bias)))
            def du_fun(t):
                return float(np.dot(wj[:n_in], k.Kd(t - t_in)) + wj[n_in] * float(k.Kd(t - t_bias)))
            tf, du = _first_fire(U[j], grid, u_fun, du_fun, theta)
            t_post[j, b] = tf
            up[j, b] = du
    return t_post, up


def backward_layer_batch(W, t_prev, t_bias, t_post, lam_post, up, k):
    """lam_post: (n_cur, B) = dL/dt_post. Returns grad (n_cur, n_in+1) and lam_prev (n_in, B)."""
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_post.shape[1]
    grad = np.zeros_like(W)
    lam_prev = np.zeros((n_in, B))
    for b in range(B):
        t_in = t_prev[:, b]
        for j in range(n_cur):
            la = lam_post[j, b]
            if la == 0.0 or not np.isfinite(t_post[j, b]):
                continue
            alpha = la / up[j, b]
            tj = t_post[j, b]
            grad[j, n_in] -= alpha * float(k.K(tj - t_bias))
            for i in range(n_in):
                if np.isfinite(t_in[i]):
                    d = tj - t_in[i]
                    Kv = float(k.K(d))
                    Kdv = float(k.Kd(d))
                    grad[j, i] -= alpha * Kv
                    lam_prev[i, b] += alpha * W[j, i] * Kdv
    return grad, lam_prev


class TTFSNet:
    def __init__(self, sizes, tm=15.0, ts=4.0, theta=1.0, t_max=40.0,
                 w_scale=0.2, bias_val=0.0, seed=0, grid_pts=4001):
        self.sizes = list(sizes)
        self.n_layers = len(sizes) - 1
        self.k = DoubleExpKernel(tm, ts)
        self.theta = float(theta)
        self.t_max = float(t_max)
        self.t_bias = 0.0
        self.grid = np.linspace(0.0, self.t_max, int(grid_pts))
        rng = np.random.default_rng(seed)
        self.W = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            w = (rng.standard_normal((b, a + 1)) * w_scale).astype(float)
            w[:, -1] = bias_val
            self.W.append(w)
        self._cache = None

    def forward(self, t_in):
        t = t_in
        cache = []
        for l in range(self.n_layers):
            t_post, up = forward_layer_batch(self.W[l], t, self.t_bias,
                                             self.theta, self.t_max, self.k, self.grid)
            cache.append((t, t_post, up))
            t = t_post
        self._cache = cache
        return t

    def backward(self, dL_dt_out):
        grads = [None] * self.n_layers
        lam = dL_dt_out
        for l in reversed(range(self.n_layers)):
            t_prev, t_post, up = self._cache[l]
            g, lam = backward_layer_batch(self.W[l], t_prev, self.t_bias,
                                          t_post, lam, up, self.k)
            grads[l] = g
        return grads

    def loss_and_grads(self, t_in, y):
        t_out = self.forward(t_in)
        loss, dL_dt_out = latency_cross_entropy(t_out, y, self.t_max)
        grads = self.backward(dL_dt_out)
        return loss, grads, t_out
