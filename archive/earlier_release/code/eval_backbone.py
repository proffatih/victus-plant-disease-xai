"""Evaluate any trained checkpoint (hybrid/effv2s/swinv2t) on the PlantVillage
test split and save per-image predictions + metrics, for the ablation table,
bootstrap CIs and McNemar tests.

Run:
  DATA_ROOT=... RESULTS_DIR=... python3 code/eval_backbone.py --tag effv2s
"""
from __future__ import annotations
import os, json, csv, sys
from pathlib import Path
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
from dataset import (
    LeafDataset, build_transforms, list_plantvillage_classes,
    index_plantvillage, stratified_split,
)
from model import HybridLeafClassifier, SingleBackbone

RESULTS = Path(os.environ.get("RESULTS_DIR", "/home/victus/Papers/Victus_Pardus_0011_Plant_Disease_XAI/Frontiers_Plant_Science/results"))


def build(name, num_classes, img):
    if name == "hybrid":
        return HybridLeafClassifier(num_classes=num_classes, img_size_tr=img, pretrained=False)
    if name == "effv2s":
        return SingleBackbone(num_classes, "tf_efficientnetv2_s.in21k_ft_in1k", img_size=img, pretrained=False)
    if name == "swinv2t":
        return SingleBackbone(num_classes, "swinv2_tiny_window8_256.ms_in1k", img_size=img, pretrained=False)
    raise ValueError(name)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classes = list_plantvillage_classes()
    c2i = {c: i for i, c in enumerate(classes)}
    # Reconstruct the exact test split deterministically (seed=42), avoiding any
    # CSV parsing ambiguity from comma-containing class/folder names.
    _tr, _va, te = stratified_split(index_plantvillage(), seed=42)
    dl = DataLoader(LeafDataset(te, c2i, build_transforms(args.img, train=False)),
                    batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    ck = torch.load(RESULTS / f"best_{args.tag}.pt", map_location=device)
    model = build(ck["model_name"], len(classes), args.img).to(device)
    model.load_state_dict(ck["model_state"]); model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            with autocast('cuda', dtype=torch.float16):
                logits = model(x)
            ps.append(logits.argmax(-1).cpu().numpy()); ys.append(y.numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    acc = float((y == p).mean())
    macro_f1 = float(f1_score(y, p, average="macro"))
    weighted_f1 = float(f1_score(y, p, average="weighted"))

    with open(RESULTS / f"{args.tag}_test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["true_idx", "pred_idx"])
        for a, b in zip(y, p): w.writerow([int(a), int(b)])
    with open(RESULTS / f"{args.tag}_test_metrics.json", "w") as f:
        json.dump({"tag": args.tag, "model": ck["model_name"], "acc": acc,
                   "macro_f1": macro_f1, "weighted_f1": weighted_f1, "n": int(len(y)),
                   "params_total": sum(pp.numel() for pp in model.parameters())}, f, indent=2)
    print(f"[{args.tag}] acc={acc:.4f} macro_f1={macro_f1:.4f} weighted_f1={weighted_f1:.4f} n={len(y)}")


if __name__ == "__main__":
    main()
