"""Latency cross-entropy loss on torch tensors (SP-01)."""
import torch


def latency_cross_entropy(t_out, y, t_max, beta=1.0):
    """p_k = softmax(-beta * t_out_k); L = -ln p_y (averaged over batch).

    Silent outputs (t_out = inf) are placed at a large finite time for the
    softmax only; their gradient is NOT backpropagated (backward skips silent
    neurons), so the placeholder never leaks gradients.
    """
    B = t_out.shape[1]
    t = torch.where(torch.isfinite(t_out), t_out, 2.0 * t_max + 10.0)
    logits = -beta * t
    logits = logits - logits.max(dim=0, keepdim=True).values
    p = torch.exp(logits)
    p = p / p.sum(dim=0, keepdim=True)
    loss = -torch.log(p[y, torch.arange(B, device=t_out.device)] + 1e-12).mean()
    # dL/dt_k = beta * (onehot_k - p_k), since logit_k = -beta * t_k
    dL = -p.clone()
    dL[y, torch.arange(B, device=t_out.device)] += 1.0
    dL = dL * (beta / B)
    return float(loss.item()), dL
