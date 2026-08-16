"""Adam optimizer for torch tensors (SP-01). In-place updates, no autograd."""
import math

import torch


class AdamTorch:
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0, clip=None):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.wd = float(wd)
        self.clip = float(clip) if clip is not None else None
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
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
            W.sub_(a_t * self.m[i] / (torch.sqrt(self.v[i]) + self.eps))
        return params
