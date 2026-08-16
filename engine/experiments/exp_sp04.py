"""SP-04 experiments: exact + cheap + local temporal AND spatial credit.

Run:  python engine/experiments/exp_sp04.py
Writes JSON to docs/results/sp04/ and prints a summary.

D3 (recorded in MEMORY.md): per-layer local loss (deep supervision) is the
mechanism; feedback alignment and forward-only contrastive are ablations that
measure the cost of stricter locality (Q4.1). Research + mapping:
docs/research/SP-04-research.md.

E1  gradcheck of the per-layer-loss objective (depth 3 and 4, incl. readout
    weights), plus a mixed fired/silent depth-3 config with the existence
    channel active. Proves the local mechanism is EXACT (each local loss is a
    real objective).
E2  memory: retained state per neuron is O(1) in the time grid (measured),
    vs BPTT-over-grid which is O(G); the transient grid scan is the only thing
    that grows.
E3  no accuracy regression vs the SP-02 solid state: 'deep' (local) vs 'ref'
    (exact W^T) on the E9-equivalent task, both with the existence channel.
E4  deep net (4 hidden layers): trains with 'deep' mode; diagnostics per
    arXiv:2606.21126 -- per-layer cosine of the local credit vs the exact
    reference, scale stability, reference validity, depth utility (accuracy
    vs a frozen-lower-blocks baseline).
E5  ablations (Q4.1): 'fa' (random feedback, keeps global error) and
    'contrastive' (forward-only, no backward) on the same deep net; reports
    accuracy + per-layer cosine -- the measured cost of stricter locality.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snn_torch import TTFSNetTorch, device, peak_margin_torch
from losses_torch import latency_cross_entropy
from optimizers_torch import AdamTorch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_sp01 import _build_smooth_net, _torch_fired_sets, _status_flip  # noqa: E402
from exp_sp02 import _make_class_task  # noqa: E402

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "..", "docs", "results", "sp04")

T_NOISE = 1.0
LAM = 5.0


# --------------------------------------------------------------------------
# E1: gradcheck of the per-layer-loss objective
# --------------------------------------------------------------------------

def _layer_objective(net, l, t_in, y, T_noise, lam):
    """Layer l's OWN local objective: its readout CE (hidden) or output CE,
    plus its own SP-02 existence loss. The 'deep' mechanism is DECOUPLED: each
    layer's weights are updated by the exact gradient of THIS objective only
    (no W^T transport, no adjoint between layers), so the gradcheck perturbs
    W_l/R_l and compares against the FD of this scalar."""
    net.forward(t_in)
    W = net.W[l]
    t_prev, t_post, up = net._cache[l]
    fired = torch.isfinite(t_post)
    if l == net.n_layers - 1:
        lv, _ = latency_cross_entropy(t_post, y, net.t_max)
    else:
        t_eff = torch.where(fired, t_post, torch.zeros_like(t_post))
        lv, _ = latency_cross_entropy(net.R[l] @ t_eff, y, net.t_max)
    B = t_in.shape[1]
    n_out = net.sizes[-1]
    onehot = torch.zeros(n_out, B, dtype=torch.bool, device=t_post.device)
    onehot[y, torch.arange(B, device=t_post.device)] = True
    t_peak, u_peak = peak_margin_torch(W, t_prev, net.t_bias, net.theta,
                                       net.grid, net.tm, net.ts,
                                       net._alpha, net.k_peak)
    if l == net.n_layers - 1:
        target = (~fired) & onehot
    else:
        target = ~fired
    target = target.to(torch.float64)
    p = torch.sigmoid((u_peak - net.theta) / T_noise)
    lex = -float((target * torch.log(p.clamp(min=1e-12))).sum().item()) * (lam / B)
    return lv + lex


def _gradcheck_local(depth, seed, mode="deep", n_in=8, B=8, eps=1e-5,
                     n_dir=5, n_w=12, target_rel=1e-4, atol=1e-6,
                     lam=LAM, net=None, t_in=None, y=None):
    """Per-layer gradcheck of the decoupled local mechanism.

    For each layer l: perturb W_l and R_l, finite-difference layer l's OWN
    objective (_layer_objective), and compare against the analytic gradient
    returned for that layer. This is the correct check for a per-layer
    decoupled rule (each layer minimizes its own objective exactly). R does
    not affect the forward, so R perturbations never flip status and are
    exactly linear (arrival = R @ t_eff).
    """
    if net is None:
        net, t_in, y, stats = _build_smooth_net(depth, seed, n_in=n_in, B=B)
    else:
        stats = {}
    dev = net.dev
    rng = np.random.default_rng(seed + 4242)
    _, grads, grads_R, _ = net.local_learning_grads(
        t_in, y, T_noise=T_NOISE, lam=lam, mode=mode)
    base_fired = _torch_fired_sets(net, t_in)

    def obj(l):
        return _layer_objective(net, l, t_in, y, T_NOISE, lam)

    def fired_sets():
        net.forward(t_in)
        return [torch.isfinite(c[1]).cpu().numpy() for c in net._cache]

    dot_errors, w_errors, w_abs_errors = [], [], []
    n_near_zero, skipped_flips = 0, 0
    for l in range(net.n_layers):
        Wl = net.W[l]
        an_W = grads[l]
        for _ in range(n_dir):
            V = torch.randn_like(Wl)
            V.div_(V.norm() + 1e-12)
            Wl.add_(eps * V)
            l_plus, f_plus = obj(l), fired_sets()
            Wl.sub_(2.0 * eps * V)
            l_minus, f_minus = obj(l), fired_sets()
            Wl.add_(eps * V)
            if _status_flip(base_fired, f_plus, f_minus):
                continue
            fd = (l_plus - l_minus) / (2.0 * eps)
            an = float((an_W * V).sum())
            dot_errors.append(abs(an - fd) / (abs(fd) + 1e-12))
        flat = rng.choice(int(Wl.numel()), size=min(n_w, int(Wl.numel())),
                          replace=False)
        for fi in flat:
            idx = np.unravel_index(int(fi), Wl.shape)
            orig = float(Wl[idx])
            Wl[idx] = orig + eps
            l_plus, f_plus = obj(l), fired_sets()
            Wl[idx] = orig - eps
            l_minus, f_minus = obj(l), fired_sets()
            Wl[idx] = orig
            if _status_flip(base_fired, f_plus, f_minus):
                skipped_flips += 1
                continue
            fd = (l_plus - l_minus) / (2.0 * eps)
            an = float(an_W[idx])
            if max(abs(an), abs(fd)) < atol:
                n_near_zero += 1
                w_abs_errors.append(abs(an - fd))
            else:
                w_errors.append(abs(an - fd) / (abs(fd) + 1e-12))
        if grads_R is not None and l < len(grads_R) and grads_R[l] is not None:
            an_R = grads_R[l]
            Rl = net.R[l]
            flat = rng.choice(int(Rl.numel()), size=min(n_w, int(Rl.numel())),
                              replace=False)
            for fi in flat:
                idx = np.unravel_index(int(fi), Rl.shape)
                orig = float(Rl[idx])
                Rl[idx] = orig + eps
                l_plus = obj(l)
                Rl[idx] = orig - eps
                l_minus = obj(l)
                Rl[idx] = orig
                fd = (l_plus - l_minus) / (2.0 * eps)
                an = float(an_R[idx])
                if max(abs(an), abs(fd)) < atol:
                    n_near_zero += 1
                    w_abs_errors.append(abs(an - fd))
                else:
                    w_errors.append(abs(an - fd) / (abs(fd) + 1e-12))

    def _stat(a):
        a = np.array(a)
        return {"mean": float(a.mean()) if a.size else None,
                "median": float(np.median(a)) if a.size else None,
                "max": float(a.max()) if a.size else None}

    dot_stat, w_stat, w_abs_stat = _stat(dot_errors), _stat(w_errors), _stat(w_abs_errors)
    fired_frac = float(np.mean([s.mean() for s in base_fired]))
    return {
        "depth": depth,
        "seed": seed,
        "sizes": net.sizes,
        "lam": lam,
        "n_dot_dirs": len(dot_errors),
        "dot_rel_err": dot_stat,
        "n_weights_checked": len(w_errors),
        "n_near_zero_grad": n_near_zero,
        "skipped_flips": skipped_flips,
        "per_weight_rel_err": w_stat,
        "near_zero_abs_err": w_abs_stat,
        "fired_frac": fired_frac,
        "conditioning": stats,
        "pass": bool(dot_stat["mean"] is not None and w_stat["mean"] is not None
                     and dot_stat["mean"] < target_rel and w_stat["mean"] < target_rel
                     and (w_abs_stat["max"] is None or w_abs_stat["max"] < atol)),
    }


def _build_mixed_net(depth, seed, n_in=8, B=8, n_out=3):
    """Net with a mix of fired and silent neurons (existence channel active).

    Start from the all-positive smooth-net recipe (uniform positive weights,
    early spread input times), then deliberately silence 2 neurons per hidden
    layer by negating their incoming rows (all-negative weights =>
    u <= 0 < theta, guaranteed silent). The output layer stays all-fired: the
    latency CE's gradient is deliberately the *unclamped* softmax gradient
    (SP-02: pushes silent targets to fire), which only equals the FD of the
    clamped loss when the label's probability is real, so labels must never
    point at a silent output neuron in a gradcheck. Fired neurons keep a
    positive margin against input arrivals so both FD sides stay smooth
    (residual status flips are skipped by the flip check)."""
    for attempt in range(40):
        rng = np.random.default_rng(seed + 555 * attempt)
        sizes = [n_in] + [12] * (depth - 1) + [n_out]
        net = TTFSNetTorch(sizes, seed=seed, w_scale=1.0, bias_val=0.2,
                           peak_tol=0.0)
        for W in net.W:
            a, b = W.shape
            W[:, :b - 1] = torch.tensor(rng.uniform(0.8, 1.2, (a, b - 1)),
                                        dtype=W.dtype, device=W.device)
        for l in range(net.n_layers - 1):
            for j in (0, 3):
                if j < net.W[l].shape[0]:
                    net.W[l][j, :] = -net.W[l][j, :]
        t_in = torch.tensor(rng.uniform(0.1, 0.4, (n_in, B)),
                            dtype=net.dtype, device=net.dev)
        y = torch.tensor(rng.integers(0, n_out, size=B), device=net.dev)
        net.forward(t_in)
        fired = np.concatenate([torch.isfinite(c[1]).cpu().numpy().ravel()
                                for c in net._cache])
        ff = float(fired.mean())
        margins = []
        for l in range(1, net.n_layers):
            tp = net._cache[l][1].cpu().numpy()
            tp_prev = net._cache[l][0].cpu().numpy()
            for j in range(tp.shape[0]):
                for b in range(tp.shape[1]):
                    tf = tp[j, b]
                    if np.isfinite(tf):
                        fin = tp_prev[:, b][np.isfinite(tp_prev[:, b])]
                        if fin.size:
                            margins.append(tf - float(np.max(fin)))
        mm = float(min(margins)) if margins else -1.0
        if 0.35 <= ff <= 0.97 and mm >= 0.03:
            return net, t_in, y, {"fired_frac": ff, "min_margin": mm,
                                  "attempts": attempt + 1}
    raise RuntimeError("no mixed fired/silent config found")


# --------------------------------------------------------------------------
# E2: memory O(1) vs O(G)
# --------------------------------------------------------------------------

def e2_memory(grids=(401, 1001, 4001, 16001), sizes=(10, 24, 24, 24, 24, 2)):
    B = 8
    dev = device()
    rows = []
    total_neurons = int(sum(sizes[1:]))
    for G in grids:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        net = TTFSNetTorch(list(sizes), seed=0, w_scale=0.4, bias_val=0.2,
                           grid_pts=G)
        t_in = torch.rand(sizes[0], B, dtype=net.dtype, device=net.dev) * 8.0
        net.forward(t_in)
        retained_bytes = sum(t.numel() * t.element_size()
                             for c in net._cache for t in c)
        retained_per_neuron = retained_bytes / (total_neurons * B)
        peak_mb = ((torch.cuda.max_memory_allocated() / 2 ** 20)
                   if torch.cuda.is_available() else 0.0)
        bptt_elems = sum(l_n * B * G for l_n in sizes[1:]) + sum(l_n * B for l_n in sizes)
        rows.append({
            "grid_pts": G,
            "engine_retained_bytes": retained_bytes,
            "engine_retained_bytes_per_neuron": float(retained_per_neuron),
            "engine_peak_gpu_mb_forward": float(peak_mb),
            "bptt_over_grid_stored_elements": int(bptt_elems),
        })
    retained_flat = [r["engine_retained_bytes"] for r in rows]
    stable = bool(max(retained_flat) - min(retained_flat) <= 0.05 * max(retained_flat))
    return {"rows": rows,
            "retained_O1_in_grid": stable,
            "note": "retained state = per-layer (t_prev, t_post, u'); the grid "
                    "scan is a transient recomputation (peak GPU memory grows), "
                    "not a stored trajectory. BPTT-over-grid stores O(n*B*G)."}


# --------------------------------------------------------------------------
# training helpers (E3/E4/E5)
# --------------------------------------------------------------------------

def _train_mode(net, x, y, xt, yt, epochs, B, lr, mode, T_noise=1.0,
                lam=LAM, record_every=5, seed=1, frozen_layers=()):
    rng = np.random.default_rng(seed + 777)
    dev = net.dev
    params = net.W + (net.R if mode == "deep" else [])
    opt = AdamTorch(params, lr=lr, clip=5.0)
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
        net.forward(t_in)
        hid = []
        for l in range(net.n_layers - 1):
            tp = net._cache[l][1]
            hid.append(float((~torch.isfinite(tp)).float().mean().item()))
        return max(hid) if hid else None

    history = []
    for ep in range(epochs):
        perm = rng.permutation(n_train)
        for s in range(0, n_train, B):
            idx = perm[s:s + B]
            t_in = torch.tensor(encode(x[idx].T), dtype=net.dtype, device=dev)
            yy = torch.tensor(y[idx], device=dev)
            _, grads, grads_R, _ = net.local_learning_grads(
                t_in, yy, T_noise=T_noise, lam=lam, mode=mode)
            gs = list(grads)
            for l in frozen_layers:
                gs[l] = torch.zeros_like(gs[l]) if gs[l] is not None else None
            if mode == "deep":
                gs = gs + grads_R
            opt.step(params, gs)
        if ep % record_every == 0 or ep == epochs - 1:
            history.append({
                "epoch": ep,
                "train_acc": predict(x, y),
                "test_acc": predict(xt, yt),
                "hidden_silent_frac": sil_stats(x, y),
            })
    return history


def _per_layer_cosine(net, t_in, y, T_noise=T_NOISE, lam=LAM):
    """Per-layer cosine of each local mode's per-layer weight gradient vs the
    exact 'ref' gradient on the same net/batch (arXiv:2606.21126: per-layer,
    not aggregate). Also reports per-layer gradient norms (scale stability)."""
    _, grads_ref, _, _ = net.local_learning_grads(
        t_in, y, T_noise=T_noise, lam=lam, mode="ref")
    out = {"ref_grad_norm_per_layer": []}
    for l in range(net.n_layers):
        out["ref_grad_norm_per_layer"].append(
            float(grads_ref[l].norm().item()))
    for mode in ("deep", "fa", "contrastive"):
        _, grads_m, _, _ = net.local_learning_grads(
            t_in, y, T_noise=T_noise, lam=lam, mode=mode)
        cos = []
        norms = []
        for l in range(net.n_layers):
            a = grads_ref[l].reshape(-1)
            b = grads_m[l].reshape(-1)
            cos.append(float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)))
            norms.append(float(b.norm().item()))
        out[mode] = {"cosine_per_layer": cos,
                     "mean_cosine": float(np.mean(cos)),
                     "grad_norm_per_layer": norms}
    return out


# --------------------------------------------------------------------------
# E3 / E4 / E5
# --------------------------------------------------------------------------

def _seed_with_silence(depth_sizes, seed_lo=0, n_try=8):
    x, y, xt, yt = _make_class_task(seed=3)
    for seed in range(seed_lo, seed_lo + n_try):
        net0 = TTFSNetTorch(list(depth_sizes), seed=seed, w_scale=0.3,
                            bias_val=0.2, grid_pts=1001)
        dev = net0.dev
        t_in = torch.tensor((0.5 + 7.5 * (1.0 - x[:64].T)),
                            dtype=net0.dtype, device=dev)
        net0.forward(t_in)
        hid_sil = max(float((~torch.isfinite(c[1])).float().mean().item())
                      for c in net0._cache[:-1])
        out_sil = float((~torch.isfinite(net0._cache[-1][1])).float().mean().item())
        if 0.1 <= hid_sil <= 0.95 and out_sil >= 0.15:
            return seed, hid_sil, out_sil
    return seed_lo, 0.0, 0.0


def e3_no_regression(epochs=40, B=64, lr=0.02):
    """'deep' (local) vs 'ref' (exact W^T), both with the existence channel,
    identical init. Gate D: no accuracy regression vs the SP-02 solid state."""
    x, y, xt, yt = _make_class_task(seed=3)
    seed, hid0, out0 = _seed_with_silence([10, 24, 2], seed_lo=0, n_try=8)
    net_ref = TTFSNetTorch([10, 24, 2], seed=seed, w_scale=0.3, bias_val=0.2)
    hist_ref = _train_mode(net_ref, x, y, xt, yt, epochs, B, lr, mode="ref")
    net_local = TTFSNetTorch([10, 24, 2], seed=seed, w_scale=0.3, bias_val=0.2)
    hist_local = _train_mode(net_local, x, y, xt, yt, epochs, B, lr, mode="deep")

    ref_t = hist_ref[-1]["test_acc"]
    loc_t = hist_local[-1]["test_acc"]
    return {
        "seed": seed,
        "init_silence_hidden": hid0,
        "init_silence_output": out0,
        "ref": {"final_train_acc": hist_ref[-1]["train_acc"],
                "final_test_acc": ref_t,
                "final_hidden_silent_frac": hist_ref[-1]["hidden_silent_frac"],
                "history": hist_ref},
        "deep_local": {"final_train_acc": hist_local[-1]["train_acc"],
                       "final_test_acc": loc_t,
                       "final_hidden_silent_frac": hist_local[-1]["hidden_silent_frac"],
                       "history": hist_local},
        "pass": bool(loc_t >= ref_t - 0.10 and loc_t > 0.8),
    }


def e4_deep_diagnostics(epochs=30, B=64, lr=0.02):
    """Deep net (4 hidden layers): trains with 'deep' mode; per-layer cosine,
    scale stability, reference validity, depth utility (2606.21126 protocol)."""
    sizes = [10, 24, 24, 24, 24, 2]
    x, y, xt, yt = _make_class_task(seed=5)
    seed, hid0, out0 = _seed_with_silence(sizes, seed_lo=0, n_try=10)
    dev = device()
    t_in = torch.tensor((0.5 + 7.5 * (1.0 - x[:64].T)), dtype=torch.float64,
                        device=dev)
    yy = torch.tensor(y[:64], device=dev)

    net_deep = TTFSNetTorch(sizes, seed=seed, w_scale=0.4, bias_val=0.2,
                            grid_pts=1001)
    cosine = _per_layer_cosine(net_deep, t_in, yy)
    hist_deep = _train_mode(net_deep, x, y, xt, yt, epochs, B, lr, mode="deep")

    net_ref = TTFSNetTorch(sizes, seed=seed, w_scale=0.4, bias_val=0.2,
                           grid_pts=1001)
    hist_ref = _train_mode(net_ref, x, y, xt, yt, epochs, B, lr, mode="ref")

    net_frozen = TTFSNetTorch(sizes, seed=seed, w_scale=0.4, bias_val=0.2,
                              grid_pts=1001)
    hist_frozen = _train_mode(net_frozen, x, y, xt, yt, epochs, B, lr,
                              mode="deep", frozen_layers=(0, 1))

    fp64_eps = float(torch.finfo(torch.float64).eps)
    ref_norms = cosine["ref_grad_norm_per_layer"]
    depth_utility = hist_deep[-1]["test_acc"] - hist_frozen[-1]["test_acc"]
    return {
        "seed": seed,
        "sizes": sizes,
        "init_silence_hidden": hid0,
        "init_silence_output": out0,
        "cosine_per_layer": cosine,
        "reference_validity": {
            "fp64_eps": fp64_eps,
            "floor_threshold": 10.0 * fp64_eps,
            "min_ref_grad_norm": float(min(ref_norms)),
            "all_above_floor": bool(min(ref_norms) > 10.0 * fp64_eps),
        },
        "deep": {"final_test_acc": hist_deep[-1]["test_acc"],
                 "final_hidden_silent_frac": hist_deep[-1]["hidden_silent_frac"],
                 "history": hist_deep},
        "ref": {"final_test_acc": hist_ref[-1]["test_acc"],
                "final_hidden_silent_frac": hist_ref[-1]["hidden_silent_frac"],
                "history": hist_ref},
        "frozen_lower_blocks": {"frozen_layers": [0, 1],
                                "final_test_acc": hist_frozen[-1]["test_acc"],
                                "history": hist_frozen},
        "depth_utility_pp": float(depth_utility * 100.0),
        "depth_utility_pass": bool(depth_utility >= 0.02),
        "deep_trains": bool(hist_deep[-1]["test_acc"] > 0.8),
    }


def e5_ablations(epochs=30, B=64, lr=0.02):
    """Cost of stricter locality (Q4.1): 'fa' (random feedback, keeps global
    error) and 'contrastive' (forward-only, no backward/readouts) vs 'ref' and
    'deep' on the same deep net."""
    sizes = [10, 24, 24, 24, 24, 2]
    x, y, xt, yt = _make_class_task(seed=5)
    seed, hid0, out0 = _seed_with_silence(sizes, seed_lo=0, n_try=10)
    dev = device()
    t_in = torch.tensor((0.5 + 7.5 * (1.0 - x[:64].T)), dtype=torch.float64,
                        device=dev)
    yy = torch.tensor(y[:64], device=dev)

    results = {}
    for mode in ("fa", "contrastive"):
        net = TTFSNetTorch(sizes, seed=seed, w_scale=0.4, bias_val=0.2,
                           grid_pts=1001)
        cosine = _per_layer_cosine(net, t_in, yy)
        hist = _train_mode(net, x, y, xt, yt, epochs, B, lr, mode=mode)
        results[mode] = {
            "final_test_acc": hist[-1]["test_acc"],
            "final_hidden_silent_frac": hist[-1]["hidden_silent_frac"],
            "mean_cosine_vs_ref": cosine[mode]["mean_cosine"],
            "cosine_per_layer": cosine[mode]["cosine_per_layer"],
            "history": hist,
        }
    return {
        "seed": seed,
        "sizes": sizes,
        "init_silence_hidden": hid0,
        "init_silence_output": out0,
        "ablations": results,
        "note": "fa removes weight transport (keeps global error); contrastive "
                "removes the backward pass and trained readouts entirely.",
    }


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    t0_all = time.perf_counter()
    report = {"experiments": {},
              "meta": {"device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu"}}

    print("=== E1: gradcheck of the per-layer-loss objective (deep mode) ===")
    e1 = {"configs": []}
    for depth in (3, 4):
        for seed in range(2):
            s = _gradcheck_local(depth, seed)
            e1["configs"].append(s)
            d_, w_ = s["dot_rel_err"], s["per_weight_rel_err"]
            print(f"  depth={depth} seed={seed} dot={d_['mean']:.2e} "
                  f"w={w_['mean']:.2e} checked={s['n_weights_checked']} "
                  f"flips={s['skipped_flips']} pass={s['pass']}")
    try:
        net, t_in, y, stats = _build_mixed_net(3, 7)
        s = _gradcheck_local(3, 7, lam=LAM, net=net, t_in=t_in, y=y)
        s["config_type"] = "mixed fired/silent (existence active)"
        e1["configs"].append(s)
        d_, w_ = s["dot_rel_err"], s["per_weight_rel_err"]
        print(f"  mixed depth=3 dot={d_['mean']:.2e} w={w_['mean']:.2e} "
              f"fired={s['fired_frac']:.2f} pass={s['pass']}")
    except RuntimeError as exc:
        print(f"  mixed config skipped: {exc}")
    e1["pass"] = bool(all(c["pass"] for c in e1["configs"]))
    report["experiments"]["E1_local_gradcheck"] = e1

    print("=== E2: memory O(1) vs O(G) ===")
    e2 = e2_memory()
    for r in e2["rows"]:
        print(f"  G={r['grid_pts']:6d} retained={r['engine_retained_bytes']:8d}B "
              f"per_neuron={r['engine_retained_bytes_per_neuron']:.2f}B "
              f"peak_gpu={r['engine_peak_gpu_mb_forward']:.1f}MB "
              f"bptt_elems={r['bptt_over_grid_stored_elements']:d}")
    print(f"  retained O(1) in grid: {e2['retained_O1_in_grid']}")
    report["experiments"]["E2_memory"] = e2

    print("=== E3: no-regression (deep-local vs ref, both + existence) ===")
    e3 = e3_no_regression()
    print(f"  seed={e3['seed']} init_hidden_sil={e3['init_silence_hidden']:.2f}")
    print(f"  ref : final_test={e3['ref']['final_test_acc']:.3f} "
          f"hidden_sil={e3['ref']['final_hidden_silent_frac']:.3f}")
    print(f"  deep: final_test={e3['deep_local']['final_test_acc']:.3f} "
          f"hidden_sil={e3['deep_local']['final_hidden_silent_frac']:.3f}")
    print(f"  pass={e3['pass']}")
    report["experiments"]["E3_no_regression"] = e3

    print("=== E4: deep net (4 hidden) diagnostics ===")
    e4 = e4_deep_diagnostics()
    for m in ("deep", "fa", "contrastive"):
        c = e4["cosine_per_layer"][m]["cosine_per_layer"]
        print(f"  cosine({m:>11s}) per layer: " + " ".join(f"{v:.3f}" for v in c))
    print(f"  ref norm per layer: "
          + " ".join(f"{v:.2e}" for v in e4["cosine_per_layer"]["ref_grad_norm_per_layer"]))
    rv = e4["reference_validity"]
    print(f"  ref validity: min_norm={rv['min_ref_grad_norm']:.2e} "
          f"above_floor={rv['all_above_floor']}")
    print(f"  deep test={e4['deep']['final_test_acc']:.3f} "
          f"ref test={e4['ref']['final_test_acc']:.3f} "
          f"frozen test={e4['frozen_lower_blocks']['final_test_acc']:.3f}")
    print(f"  depth_utility={e4['depth_utility_pp']:.2f}pp pass={e4['depth_utility_pass']} "
          f"deep_trains={e4['deep_trains']}")
    report["experiments"]["E4_deep_diagnostics"] = e4

    print("=== E5: ablations (FA, contrastive) on the deep net ===")
    e5 = e5_ablations()
    for mode in ("fa", "contrastive"):
        a = e5["ablations"][mode]
        print(f"  {mode:>11s}: test={a['final_test_acc']:.3f} "
              f"mean_cos={a['mean_cosine_vs_ref']:.3f} "
              f"sil={a['final_hidden_silent_frac']:.3f}")
    report["experiments"]["E5_ablations"] = e5

    report["meta"]["wall_time_s"] = time.perf_counter() - t0_all
    path = os.path.join(RESULT_DIR, "sp04-results.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
