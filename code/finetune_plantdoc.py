"""Target-domain fine-tuning on PlantDoc (reviewer-requested experiment).

Takes the PlantVillage-trained hybrid checkpoint and fine-tunes it on the
PlantDoc *train* split (shared-label subset), then evaluates on the disjoint
PlantDoc *test* split. Reports zero-shot vs fine-tuned accuracy/macro-F1 on the
same held-out test split, so no fine-tuning image is ever scored.

Run:
  DATA_ROOT=/path/to/datasets RESULTS_DIR=results \
    python3 code/finetune_plantdoc.py --epochs 30 --lr 5e-5
"""
from __future__ import annotations
import os, json, csv, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
from dataset import (
    LeafDataset, build_transforms, list_plantvillage_classes,
    PD_TRAIN_ROOT, PD_TEST_ROOT, PLANTDOC_TO_PV,
)
from model import HybridLeafClassifier

RESULTS = Path(os.environ.get("RESULTS_DIR", "/home/victus/Papers/Victus_Pardus_0011_Plant_Disease_XAI/Frontiers_Plant_Science/results"))


def index_plantdoc_split(root: Path, pv_class_to_idx: dict):
    """Images under one PlantDoc split folder, mapped to shared PV labels."""
    items = []
    for pd_cls, pv_cls in PLANTDOC_TO_PV.items():
        if pv_cls not in pv_class_to_idx:
            continue
        d = root / pd_cls
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append((str(f), pv_cls))
    return items


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast('cuda', dtype=torch.float16):
            logits = model(x)
        ps.append(logits.argmax(-1).cpu().numpy())
        ys.append(y.numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    acc = float((y == p).mean())
    macro_f1 = float(f1_score(y, p, average="macro", labels=np.unique(y)))
    return acc, macro_f1, y, p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt", default=str(RESULTS / "best_hybrid.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = list_plantvillage_classes()
    c2i = {c: i for i, c in enumerate(classes)}

    tr_items = index_plantdoc_split(PD_TRAIN_ROOT, c2i)
    te_items = index_plantdoc_split(PD_TEST_ROOT, c2i)
    shared = sorted({c for _, c in te_items})
    print(f"[info] PlantDoc fine-tune fit={len(tr_items)} eval={len(te_items)} shared_classes={len(shared)}")

    tf_tr = build_transforms(args.img, train=True)
    tf_ev = build_transforms(args.img, train=False)
    dl_tr = DataLoader(LeafDataset(tr_items, c2i, tf_tr), batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_te = DataLoader(LeafDataset(te_items, c2i, tf_ev), batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    model = HybridLeafClassifier(num_classes=len(classes), img_size_tr=args.img, pretrained=False).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model_state"])
    print(f"[info] loaded checkpoint {args.ckpt} (val_acc={ck.get('val_acc')})")

    # Zero-shot baseline on the held-out PlantDoc test split
    zs_acc, zs_f1, _, _ = evaluate(model, dl_te, device)
    print(f"[zero-shot on PD test] acc={zs_acc:.4f} macro_f1={zs_f1:.4f}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
    total = len(dl_tr) * args.epochs
    warm = max(50, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = GradScaler('cuda')
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    log = [{"epoch": 0, "ft_acc": zs_acc, "ft_macro_f1": zs_f1}]
    best_acc, best_f1 = zs_acc, zs_f1
    step = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast('cuda', dtype=torch.float16):
                loss = ce(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
        acc, f1, y_true, y_pred = evaluate(model, dl_te, device)
        print(f"[ft epoch {ep}] PD-test acc={acc:.4f} macro_f1={f1:.4f}")
        log.append({"epoch": ep, "ft_acc": acc, "ft_macro_f1": f1})
        if acc > best_acc:
            best_acc, best_f1 = acc, f1
            with open(RESULTS / "plantdoc_finetuned_predictions.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["true_idx", "pred_idx"])
                for a, b in zip(y_true, y_pred): w.writerow([int(a), int(b)])

    with open(RESULTS / "plantdoc_finetune_metrics.json", "w") as f:
        json.dump({
            "zero_shot_pd_test_acc": zs_acc, "zero_shot_pd_test_macro_f1": zs_f1,
            "finetuned_pd_test_acc": best_acc, "finetuned_pd_test_macro_f1": best_f1,
            "n_fit": len(tr_items), "n_eval": len(te_items), "shared_classes": len(shared),
            "epochs": args.epochs, "lr": args.lr,
        }, f, indent=2)
    with open(RESULTS / "plantdoc_finetune_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "ft_acc", "ft_macro_f1"]); w.writeheader()
        for r in log: w.writerow(r)
    print(f"[done] zero-shot acc={zs_acc:.4f} -> fine-tuned acc={best_acc:.4f} (macro_f1 {zs_f1:.4f}->{best_f1:.4f})")


if __name__ == "__main__":
    main()
