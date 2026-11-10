"""REV1 (R3-2): label-free test-time adaptation as a lightweight remedy.

A practitioner who cannot label field images can still collect unlabelled
ones. Two standard label-free adaptations are evaluated on every model that
contains batch normalisation (EfficientNetV2-S and the EfficientNetV2-S branch
of both hybrids; Swin V2-T uses layer normalisation only):

  AdaBN  (Li et al. 2018) batch-norm running statistics are re-estimated on
         unlabelled field images; no weight changes.
  TENT   (Wang et al. 2021) AdaBN followed by one epoch of entropy minimisation
         that updates only the batch-norm affine parameters.

To keep adaptation and scoring disjoint, each field set is split in two halves
with a fixed seed; the model adapts on one half (labels unused) and is scored
on the other, and the roles are swapped, so every image is scored exactly once
by a model that never saw it during adaptation.
"""
from __future__ import annotations
import copy, json, os, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import LeafDataset, build_transforms, index_plantdoc_shared
from plantwild import index_plantwild_shared
from dump_features import load_model, R3RES

OUTD = Path(__file__).resolve().parents[1] / "results"


def ece(conf, correct, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def bn_layers(m):
    return [x for x in m.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]


@torch.no_grad()
def predict(model, items, c2i, S, device):
    model.eval()
    dl = DataLoader(LeafDataset(items, c2i, build_transforms(S, train=False)), batch_size=32, num_workers=6)
    P, Y = [], []
    for x, y in dl:
        P.append(model(x.to(device)).float().softmax(-1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def adabn(model, items, c2i, S, device):
    for b in bn_layers(model):
        b.reset_running_stats(); b.momentum = None  # cumulative average
    model.eval()
    for b in bn_layers(model):
        b.train()
    dl = DataLoader(LeafDataset(items, c2i, build_transforms(S, train=False)), batch_size=32,
                    shuffle=True, num_workers=6, drop_last=True)
    with torch.no_grad():
        for x, _ in dl:
            model(x.to(device))
    model.eval()
    return model


def tent(model, items, c2i, S, device, lr=1e-4, seed=0):
    model = adabn(model, items, c2i, S, device)
    params = []
    for p in model.parameters():
        p.requires_grad = False
    for b in bn_layers(model):
        b.weight.requires_grad = True; b.bias.requires_grad = True
        params += [b.weight, b.bias]
        b.momentum = 0.1
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(LeafDataset(items, c2i, build_transforms(S, train=False)), batch_size=16,
                    shuffle=True, num_workers=6, drop_last=True, generator=g)
    model.eval()
    for b in bn_layers(model):
        b.train()
    for x, _ in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            z = model(x.to(device))
        p = z.float().softmax(-1)
        loss = -(p * p.clamp_min(1e-12).log()).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def metrics(P, Y):
    c = P.argmax(1) == Y
    return {"acc": float(c.mean()), "ece": ece(P.max(1), c), "mean_conf": float(P.max(1).mean()),
            "max_class_share": float(np.bincount(P.argmax(1)).max() / len(Y))}


def main():
    tags = sys.argv[1:] or [f"{a}_s{s}" for a in ("effv2s", "hybrid_gated", "hybrid_tokenattn") for s in (42, 1337, 2024)]
    device = "cuda"
    dst = OUTD / "tta_rev1.json"
    out = json.load(open(dst)) if dst.exists() else {}
    for t in tags:
        if t in out:
            continue
        base, c2i, S = load_model(R3RES / f"best_{t}.pt", "cpu")
        sets = {"plantdoc": [(p, c) for p, c in index_plantdoc_shared(c2i) if c in c2i],
                "plantwild": index_plantwild_shared(c2i)}
        res = {}
        for ds, items in sets.items():
            rng = random.Random(7); idx = list(range(len(items))); rng.shuffle(idx)
            halves = [[items[i] for i in idx[: len(idx) // 2]], [items[i] for i in idx[len(idx) // 2:]]]
            block = {}
            for mode in ("none", "adabn", "tent"):
                Ps, Ys = [], []
                for a, b in ((0, 1), (1, 0)):
                    m = copy.deepcopy(base).to(device)
                    if mode == "adabn":
                        m = adabn(m, halves[a], c2i, S, device)
                    elif mode == "tent":
                        m = tent(m, halves[a], c2i, S, device)
                    P, Y = predict(m, halves[b], c2i, S, device)
                    Ps.append(P); Ys.append(Y)
                    del m; torch.cuda.empty_cache()
                block[mode] = metrics(np.concatenate(Ps), np.concatenate(Ys))
            res[ds] = block
            print(t, ds, {k: round(v["acc"], 4) for k, v in block.items()}, flush=True)
        out[t] = res
        json.dump(out, open(dst, "w"), indent=1)
        del base; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
