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


def spike_count_cross_entropy(t_spikes, y, beta=1.0):
    """Cross-entropy on spike counts. t_spikes: (n_out, B, K) or (n_out, B).

    Counts finite spike times per output neuron, then applies CE on the counts.
    Silent outputs (zero spikes) are handled gracefully via softmax.
    Returns (loss, dL/d(count)) where dL/d(count) is (n_out, B).
    """
    if t_spikes.dim() == 3:
        counts = torch.isfinite(t_spikes).float().sum(dim=2)
    else:
        counts = (torch.isfinite(t_spikes)).float()
    counts = counts - counts.mean(dim=0, keepdim=True)
    logits = beta * counts
    logits = logits - logits.max(dim=0, keepdim=True).values
    p = torch.exp(logits)
    p = p / p.sum(dim=0, keepdim=True)
    B = t_spikes.shape[1]
    counts = torch.isfinite(t_spikes).float().sum(dim=2) if t_spikes.dim() == 3 else torch.isfinite(t_spikes).float()
    counts = counts - counts.mean(dim=0, keepdim=True)
    logits = beta * counts
    logits = logits - logits.max(dim=0, keepdim=True).values
    p = torch.exp(logits)
    p = p / p.sum(dim=0, keepdim=True)
    loss = -torch.log(p[y, torch.arange(B, device=p.device)] + 1e-12).mean()
    dL = -p.clone()
    dL[y, torch.arange(B, device=p.device)] += 1.0
    dL = dL * (beta / B)
    return float(loss.item()), dL


def rate_latency_loss(t_spikes, y, t_max, beta=1.0):
    """Combined rate-latency loss: CE on spike counts + latency CE on first spike.

    Combines spike_count_cross_entropy (rate) with latency_cross_entropy (TTFS).
    t_spikes: (n_out, B, K) multi-spike times.
    Returns (loss, dL/dt) where dL/dt is (n_out, B) for the backward pass.
    """
    B = t_spikes.shape[1]
    counts = torch.isfinite(t_spikes).float().sum(dim=2)
    t_first = t_spikes[:, :, 0]
    t_for_lat = torch.where(torch.isfinite(t_first), t_first,
                            torch.full_like(t_first, 2.0 * t_max + 10.0))
    logits_rate = beta * (counts - counts.mean(dim=0, keepdim=True))
    logits_lat = -beta * t_for_lat
    logits = logits_rate + logits_lat
    logits = logits - logits.max(dim=0, keepdim=True).values
    p = torch.exp(logits)
    p = p / p.sum(dim=0, keepdim=True)
    loss = -torch.log(p[y, torch.arange(B, device=p.device)] + 1e-12).mean()
    dL = -p.clone()
    dL[y, torch.arange(B, device=p.device)] += 1.0
    dL = dL * (beta / B)
    return float(loss.item()), dL
