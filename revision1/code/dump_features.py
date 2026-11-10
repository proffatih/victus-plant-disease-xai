"""REV1 (2001013): dump logits and pre-head features for every trained model.

One forward pass per (model, benchmark) is stored so that every post-hoc
analysis requested in the interactive review -- alternative uncertainty scores
for selective prediction (R3), calibration baselines beyond temperature scaling
(R2-3), deep ensembles, head-level MC dropout and feature-space (Mahalanobis)
scores (R3) -- can be computed on the CPU without re-running the networks.

Sets: PlantVillage train subset (100 images/class, seeded; class statistics for
Mahalanobis), PlantVillage val (calibration fitting), PlantVillage test,
PlantDoc (28 shared classes, all images) and PlantWild (33 shared classes).
Outputs go to $RUN_OUT (outside git): feats_<tag>.npz.
"""
from __future__ import annotations
import argparse, os, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import LeafDataset, build_transforms, load_split_csv, index_plantdoc_shared
from plantwild import index_plantwild_shared
from model import HybridLeafClassifier, SingleBackbone

R3RES = Path(os.environ.get("R3_RESULTS", Path(__file__).resolve().parents[2] / "R3" / "results"))
OUT = Path(os.environ.get("RUN_OUT", Path.home() / "RUN_OUT" / "0011_rev1"))


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c2i = ck["class_to_idx"]
    name, fusion = ck.get("model_name", "hybrid"), ck.get("fusion", "gated_linear")
    if name == "hybrid":
        m = HybridLeafClassifier(len(c2i), img_size_tr=ck["img_size"], pretrained=False, fusion=fusion)
    elif name == "effv2s":
        m = SingleBackbone(len(c2i), "tf_efficientnetv2_s.in21k_ft_in1k", ck["img_size"], False)
    else:
        m = SingleBackbone(len(c2i), "swinv2_tiny_window8_256.ms_in1k", ck["img_size"], False)
    m.load_state_dict(ck["model_state"])
    return m.to(device).eval(), c2i, ck["img_size"]


def train_subset(per_class=100, seed=42):
    items = load_split_csv(R3RES / "split_train.csv")
    by = defaultdict(list)
    for p, c in items:
        by[c].append((p, c))
    rng = random.Random(seed)
    out = []
    for c in sorted(by):
        lst = sorted(by[c])
        rng.shuffle(lst)
        out += lst[:per_class]
    return out


@torch.no_grad()
def run(model, items, c2i, S, device, batch=32):
    feats = []
    h = model.head[0].register_forward_hook(lambda mod, inp, out: feats.append(inp[0].float().cpu()))
    dl = DataLoader(LeafDataset(items, c2i, build_transforms(S, train=False, preproc="aspect")),
                    batch_size=batch, shuffle=False, num_workers=6, pin_memory=True)
    L, Y = [], []
    for x, y in dl:
        L.append(model(x.to(device, non_blocking=True)).float().cpu())
        Y.append(y)
    h.remove()
    return torch.cat(L).numpy(), torch.cat(feats).numpy().astype(np.float32), torch.cat(Y).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    for tag in args.tags:
        dst = OUT / f"feats_{tag}.npz"
        if dst.exists():
            print(f"[skip] {dst}"); continue
        model, c2i, S = load_model(R3RES / f"best_{tag}.pt", device)
        sets = {
            "pv_trainsub": train_subset(),
            "pv_val": load_split_csv(R3RES / "split_val.csv"),
            "pv_test": load_split_csv(R3RES / "split_test.csv"),
            "plantdoc": [(p, c) for p, c in index_plantdoc_shared(c2i) if c in c2i],
            "plantwild": index_plantwild_shared(c2i),
        }
        blob = {}
        for k, items in sets.items():
            lg, ft, y = run(model, items, c2i, S, device)
            blob[f"{k}_logits"], blob[f"{k}_feats"], blob[f"{k}_y"] = lg, ft.astype(np.float16), y
            blob[f"{k}_paths"] = np.array([p for p, _ in items])
            print(f"[{tag}] {k}: n={len(y)} acc={(lg.argmax(1) == y).mean():.4f}", flush=True)
        # Head weights, for head-level MC dropout (LayerNorm -> Dropout -> Linear).
        ln, lin = model.head[0], model.head[2]
        blob["ln_w"], blob["ln_b"] = ln.weight.detach().cpu().numpy(), ln.bias.detach().cpu().numpy()
        blob["lin_w"], blob["lin_b"] = lin.weight.detach().cpu().numpy(), lin.bias.detach().cpu().numpy()
        np.savez_compressed(dst, **blob)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
