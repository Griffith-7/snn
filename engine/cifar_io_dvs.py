"""CIFAR10-DVS loading + TTFS preprocessing for the Phase-5 Gate-E DVS benchmark.

Source data: NDA_SNN Google Drive mirror of the preprocessed CIFAR10-DVS
(per-sample torch .pt of binary spike frames, shape (2,128,128,10) = (pol, H, W,
frames), labels 0-9). Files are stored as zip archives under data/cifar10dvs/.

The engine is single-spike TTFS (one input spike per pixel), so each event stream
is reduced to one or two grayscale-like intensity frames before the usual
downsample + TTFS encode. This deliberately matches the CIFAR-10 Gate-E pipeline
(docs/results/SP-05-experiments.md): same n_in-scale, architecture, loss,
optimizer and encoding; ONLY the source dataset differs.

Encodings:
  'abs'    single frame, X = ON + OFF event counts per pixel (edge energy)
  'signed' single frame, X = ON - OFF event counts per pixel (motion direction)
  dual     two frames, ON and OFF kept as separate channels

A processed cache is written to data/cifar10dvs/frames_{res}_{mode}.npz so
experiments load fast without re-decompressing the 9000-sample train archive.
"""
import io
import os
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from cifar_io import encode_times  # noqa: E402

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "cifar10dvs")
TRAIN_ZIP = os.path.join(DATA_DIR, "train_file")
TEST_ZIP = os.path.join(DATA_DIR, "test_file")


def load_zip_polarities(zip_path, n, tblocks=1):
    """Iterate the first `n` .pt entries.

    tblocks=1 -> (ON (n,128,128), OFF (n,128,128), y)
    tblocks=k -> (ON (n,k,128,128), OFF (n,k,128,128), y): each block sums a
    contiguous slice of the 10 temporal frames.
    """
    z = zipfile.ZipFile(zip_path)
    entries = sorted(nm for nm in z.namelist() if nm.endswith(".pt"))
    on = np.zeros((n, tblocks, 128, 128), dtype=np.float64)
    off = np.zeros((n, tblocks, 128, 128), dtype=np.float64)
    y = np.zeros((n,), dtype=np.int64)
    for i, name in enumerate(entries[:n]):
        frames, label = torch.load(io.BytesIO(z.read(name)), weights_only=True)
        f_on = frames[0].cpu().numpy()   # (128,128,10)
        f_off = frames[1].cpu().numpy()
        if tblocks == 1:
            on[i, 0] = f_on.sum(-1)
            off[i, 0] = f_off.sum(-1)
        else:
            T = f_on.shape[-1]
            edges = np.linspace(0, T, tblocks + 1).round().astype(int)
            for b in range(tblocks):
                on[i, b] = f_on[..., edges[b]:edges[b + 1]].sum(-1)
                off[i, b] = f_off[..., edges[b]:edges[b + 1]].sum(-1)
        y[i] = int(label.item())
    return on, off, y


def load_dvs(n_train=9000, n_test=1000, res=12, mode="abs", rebuild=False,
             t_lo=0.5, t_hi=8.0,     dual=False, tblocks=1):
    """Return (ttr, ytr, tte, yte): TTFS input times + labels.

    dual=False -> single intensity frame, ttr (n, res*res)   [mode: abs|signed]
    dual=True  -> ON and OFF kept as separate channels, ttr (n, 2*res*res)
    tblocks    -> split the 10 temporal frames into k blocks, each encoded as its
                  own channel (ttr (n, k*res*res) for single-channel polarity).
    """
    n_ch = (2 if dual else 1) * tblocks
    cache = os.path.join(DATA_DIR, f"frames_{res}_{mode if not dual else 'dual'}_t{tblocks}.npz")
    if os.path.exists(cache) and not rebuild:
        d = np.load(cache)
        gtr, ytr, gte, yte = d["gtr"], d["ytr"], d["gte"], d["yte"]
    else:
        print(f"[dvs] building cache {cache} ...")
        on_tr, off_tr, ytr = load_zip_polarities(TRAIN_ZIP, n_train, tblocks)
        on_te, off_te, yte = load_zip_polarities(TEST_ZIP, n_test, tblocks)
        gtr = _reduce(on_tr, off_tr, res, mode, dual)
        gte = _reduce(on_te, off_te, res, mode, dual)
        np.savez(cache, gtr=gtr, ytr=ytr, gte=gte, yte=yte)
    ttr = encode_times(gtr, t_lo, t_hi).astype(np.float64)
    tte = encode_times(gte, t_lo, t_hi).astype(np.float64)
    return ttr, ytr, tte, yte


def _reduce(on, off, res, mode, dual):
    """(n, k, 128, 128) polarity counts -> (n, channels, res, res) in [0,1]."""
    n, k = on.shape[:2]
    if dual:
        combined = np.concatenate([on, off], axis=1)   # (n, 2k, 128, 128)
    elif mode == "signed":
        combined = on - off                             # (n, k, 128, 128)
    else:
        combined = on + off
    t = torch.from_numpy(combined.reshape(n, -1, 128, 128))  # (n, C, 128, 128)
    r = F.interpolate(t, size=(res, res), mode="area").numpy()  # (n, C, res, res)
    r = r.reshape(n, -1, res, res)
    mx = r.max(axis=(2, 3), keepdims=True)
    mx = np.where(mx > 0, mx, 1.0)
    return (r / mx).reshape(n, -1, res, res)
