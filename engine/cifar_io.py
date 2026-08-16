"""CIFAR-10 loading + TTFS preprocessing for Phase 5 (Gate E).

Loads the local `cifar-10-python/cifar-10-batches-py` (unpickled arrays), converts
to grayscale, downsamples to a chosen resolution, and provides the TTFS latency
encoding shared by the exact engine and the surrogate baseline (apples-to-apples).
"""
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

CIFAR_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cifar-10-python", "cifar-10-batches-py")


def _load_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    return d[b"data"], np.asarray(d[b"labels"])


def load_cifar10(root=None):
    """Returns (X_tr (50000, 3072) uint8, y_tr (50000,), X_te (10000, 3072), y_te)."""
    root = root or CIFAR_DIR
    X, y = [], []
    for i in range(1, 6):
        Xb, yb = _load_batch(os.path.join(root, f"data_batch_{i}"))
        X.append(Xb)
        y.append(yb)
    Xtr = np.concatenate(X, axis=0).astype(np.float64)
    ytr = np.concatenate(y, axis=0).astype(np.int64)
    Xte, yte = _load_batch(os.path.join(root, "test_batch"))
    return Xtr, ytr, Xte.astype(np.float64), yte.astype(np.int64)


def to_grayscale_resized(X, res=12):
    """X: (N, 3072) float64 in [0,255]. Returns (N, res, res) float64 in [0,1]."""
    X = X.reshape(-1, 3, 32, 32)
    g = 0.299 * X[:, 0] + 0.587 * X[:, 1] + 0.114 * X[:, 2]  # (N,32,32)
    g = g / 255.0
    if res == 32:
        return g
    gt = torch.from_numpy(g).unsqueeze(1)  # (N,1,32,32)
    gr = F.interpolate(gt, size=(res, res), mode="area").squeeze(1)
    return gr.numpy()


def encode_times(X, t_lo=0.5, t_hi=8.0):
    """(N, res, res) in [0,1] -> (N, res*res) input spike times in [t_lo, t_hi].
    Bright pixel -> early spike. Flat (t_hi) for exactly-0 pixels is allowed;
    the engine treats spikes at >= t_hi as late, never as the bias kink (bias is
    at t=0)."""
    x = X.reshape(X.shape[0], -1).clip(0.0, 1.0)
    return t_lo + (t_hi - t_lo) * (1.0 - x)


def subset(seed, X, y, n_train):
    """Deterministic stratified subset of n_train samples per class (balanced)."""
    n_cls = 10
    per = n_train // n_cls
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(n_cls):
        cand = np.where(y == c)[0]
        rng.shuffle(cand)
        idx.append(cand[:per])
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]
