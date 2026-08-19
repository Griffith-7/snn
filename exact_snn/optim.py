"""Adam optimizer for torch tensors (SP-01). In-place updates, no autograd."""
from __future__ import annotations

import math
from typing import Sequence

import torch


class AdamTorch:
    """Adam optimizer operating directly on torch tensors without autograd.

    Maintains first- and second-moment estimates (m, v) for each parameter
    tensor and applies in-place weight updates using bias-corrected
    learning rates.  Supports optional gradient clipping and decoupled
    weight decay.
    """

    def __init__(
        self,
        params: Sequence[torch.Tensor],
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        wd: float = 0.0,
        clip: float | None = None,
    ) -> None:
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.wd = float(wd)
        self.clip = float(clip) if clip is not None else None
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params: Sequence[torch.Tensor], grads: Sequence[torch.Tensor | None]) -> list[torch.Tensor]:
        """Perform one Adam update step on *params* using *grads*.

        Applies bias-corrected Adam updates with optional weight decay and
        gradient clipping.  Parameters whose corresponding gradient entry
        is ``None`` are skipped.  Returns the (mutated) list of parameters.
        """
        self.t += 1
        a_t = self.lr * (math.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t))
        for i, (W, g) in enumerate(zip(params, grads)):
            if g is None:
                continue
            g = g.to(W.dtype)
            if self.wd > 0.0:
                g = g + self.wd * W
            if self.clip is not None:
                g = torch.clamp(g, -self.clip, self.clip)
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)
            with torch.no_grad():
                W.sub_(a_t * self.m[i] / (torch.sqrt(self.v[i]) + self.eps))
        return params
