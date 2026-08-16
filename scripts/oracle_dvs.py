import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "engine")
from cifar_io_dvs import load_dvs

torch.manual_seed(0)


def oracle(enc, epochs=20):
    ttr, ytr, tte, yte = load_dvs(**enc)
    n_in = ttr.shape[1]
    Xtr = torch.tensor(ttr, dtype=torch.float32)
    Xte = torch.tensor(tte, dtype=torch.float32)
    ytr_t = torch.tensor(ytr)
    yte_t = torch.tensor(yte)
    model = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, 10))
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    lossf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        idx = torch.randperm(Xtr.shape[0])[:4000]
        model.train()
        opt.zero_grad()
        out = model(Xtr[idx])
        loss = lossf(out, ytr_t[idx])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte_t).float().mean().item()
    return acc


for enc in [dict(res=12, mode="abs", tblocks=1), dict(res=12, mode="abs", tblocks=2),
            dict(res=12, mode="abs", tblocks=3), dict(res=12, mode="abs", tblocks=5)]:
    acc = oracle(enc)
    n_in = 144 * enc["tblocks"]
    print(f"tblocks={enc['tblocks']}: n_in={n_in:4d}  oracle test {acc:.3f}")
