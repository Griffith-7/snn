"""Loss functions for exact TTFS training (SP-01)."""
import numpy as np


def latency_cross_entropy(t_out, y, t_max, beta=1.0):
    """Latency cross-entropy: p_k = softmax(-beta * t_out_k); L = -ln p_y.

    Silent outputs (t_out = inf) are placed at a large finite time so the softmax
    is well-defined; their gradient is routed through the placeholder ONLY if they
    were treated as fired, which backward_layer never does (silent => no dL/dt).
    Returns (loss, dL_dt_out) where dL_dt_out is shape (n_out, B), averaged over batch.
    """
    B = t_out.shape[1]
    t = np.where(np.isfinite(t_out), t_out, 2.0 * t_max + 10.0)
    logits = -beta * t
    logits -= np.max(logits, axis=0, keepdims=True)
    p = np.exp(logits)
    p /= np.sum(p, axis=0, keepdims=True)
    loss = -np.mean(np.log(p[y, np.arange(B)] + 1e-12))
    # dL/dt_k = beta * (onehot_k - p_k), since logit_k = -beta * t_k
    dL = -p.copy()
    dL[y, np.arange(B)] += 1.0
    dL *= beta / B
    return loss, dL
