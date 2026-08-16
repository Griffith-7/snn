"""Adam optimizer for exact-gradient TTFS training (SP-01)."""
import numpy as np


class Adam:
    def __init__(self, shapes, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.wd = float(wd)
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]
        self.t = 0

    def step(self, params, grads, clip=None):
        self.t += 1
        a_t = self.lr * (np.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t))
        for i, (W, g) in enumerate(zip(params, grads)):
            if g is None:
                continue
            g = np.asarray(g, dtype=float)
            if self.wd > 0.0:
                g = g + self.wd * W
            if clip is not None:
                g = np.clip(g, -clip, clip)
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)
            W -= a_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)
        return params
