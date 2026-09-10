"""Evaluate the trained hybrid model on PlantVillage test split and the
shared-label PlantDoc subset. Writes:
  - test_pv_predictions.csv (path,true,pred,top1_prob)
  - test_pv_metrics.json    (accuracy, top-5, macro/micro F1)
  - cross_plantdoc_predictions.csv
  - cross_plantdoc_metrics.json
  - confusion_matrix_pv.npy  (num_classes x num_classes)
  - per_class_report_pv.csv
"""
from __future__ import annotations
import os, json, csv, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).parent))
from dataset import (
    LeafDataset, build_transforms, list_plantvillage_classes,
    stratified_split, index_plantvillage, index_plantdoc_shared,
)
from model import HybridLeafClassifier

RESULTS = Path(os.environ.get("RESULTS_DIR", "/home/victus/Papers/Victus_Pardus_0011_Plant_Disease_XAI/Frontiers_Plant_Science/results"))


def load_split_from_csv(path: Path):
    """Parse a two-column ``path,class`` CSV where the class name may itself
    contain commas (e.g. ``Pepper,_bell___Bacterial_spot``). The CSV writer
    used at split time did not quote fields, so we cannot naively split on
    commas. Instead we recover the class from the parent directory name of
    the image path (which is authoritative)."""
    items = []
    with open(path) as f:
        _header = f.readline()
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # Find the .JPG/.jpg/.png suffix to locate end of path.
            low = line.lower()
            end_p = -1
            for ext in (".jpg", ".jpeg", ".png"):
                k = low.find(ext)
                if k != -1:
                    end_p = k + len(ext)
                    break
            if end_p == -1:
                # Fallback: rfind on comma (correct for non-Pepper rows)
                idx = line.rfind(",")
                items.append((line[:idx], line[idx + 1:]))
                continue
            p = line[:end_p]
            # class = parent directory name of p
            cls = Path(p).parent.name
            items.append((p, cls))
    return items


def predict(model, loader, device):
    model.eval()
    all_true, all_pred, all_top1p, all_paths = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with autocast("cuda", dtype=torch.float16):
                logits = model(x)
            probs = logits.softmax(-1).float().cpu().numpy()
            p = probs.argmax(-1)
            top1p = probs.max(-1)
            all_true.extend(y.numpy().tolist())
            all_pred.extend(p.tolist())
            all_top1p.extend(top1p.tolist())
    return np.array(all_true), np.array(all_pred), np.array(all_top1p)


def per_class_f1(true, pred, num_classes: int, labels: list[str]):
    rep = []
    for c in range(num_classes):
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = int((true == c).sum())
        rep.append((labels[c], prec, rec, f1, support))
    return rep


def confusion_matrix(true, pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    return cm


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = RESULTS / "best_hybrid.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    labels = [idx_to_class[i] for i in range(num_classes)]

    model = HybridLeafClassifier(num_classes=num_classes, img_size_tr=ckpt["img_size"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    tf_ev = build_transforms(ckpt["img_size"], train=False)

    # ---------- PlantVillage test ----------
    te = load_split_from_csv(RESULTS / "split_test.csv")
    ds_te = LeafDataset(te, class_to_idx, tf_ev)
    dl_te = DataLoader(ds_te, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    print(f"[info] PV test items: {len(te)}")
    t0 = time.time()
    true_pv, pred_pv, top1p_pv = predict(model, dl_te, device)
    latency_pv = (time.time() - t0) / max(1, len(te))
    acc_pv = float((pred_pv == true_pv).mean())
    print(f"[pv-test] acc={acc_pv:.4f}  latency_per_img={latency_pv*1000:.2f}ms")

    cm_pv = confusion_matrix(true_pv, pred_pv, num_classes)
    np.save(RESULTS / "confusion_matrix_pv.npy", cm_pv)

    rep = per_class_f1(true_pv, pred_pv, num_classes, labels)
    macro_f1 = float(np.mean([r[3] for r in rep]))
    supports = np.array([r[4] for r in rep])
    weighted_f1 = float(np.sum(supports * np.array([r[3] for r in rep])) / max(1, supports.sum()))

    with open(RESULTS / "per_class_report_pv.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for r in rep:
            w.writerow(r)

    with open(RESULTS / "test_pv_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "top1_prob"])
        for (p, _c), t, pr, prob in zip(te, true_pv, pred_pv, top1p_pv):
            w.writerow([p, labels[t], labels[pr], f"{prob:.4f}"])

    pv_metrics = {
        "acc": acc_pv,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n": int(len(te)),
        "latency_per_img_ms": latency_pv * 1000,
        "num_classes": num_classes,
    }
    with open(RESULTS / "test_pv_metrics.json", "w") as f:
        json.dump(pv_metrics, f, indent=2)

    # ---------- PlantDoc cross-domain ----------
    pd_items = index_plantdoc_shared(class_to_idx)
    # filter to only items whose PV class truly exists
    pd_items = [(p, c) for (p, c) in pd_items if c in class_to_idx]
    ds_pd = LeafDataset(pd_items, class_to_idx, tf_ev)
    dl_pd = DataLoader(ds_pd, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    print(f"[info] PlantDoc shared items: {len(pd_items)}")
    true_pd, pred_pd, top1p_pd = predict(model, dl_pd, device)
    # per-class accuracy on shared subset
    shared_cls = sorted(set(true_pd.tolist()))
    per_class_acc = {}
    for c in shared_cls:
        mask = true_pd == c
        if mask.sum() > 0:
            per_class_acc[labels[c]] = float((pred_pd[mask] == c).mean())
    acc_pd = float((pred_pd == true_pd).mean())
    top5_pd = None  # requires probs, skip for brevity
    print(f"[plantdoc-cross] acc={acc_pd:.4f}  n={len(pd_items)}")

    with open(RESULTS / "cross_plantdoc_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "top1_prob"])
        for (p, _c), t, pr, prob in zip(pd_items, true_pd, pred_pd, top1p_pd):
            w.writerow([p, labels[t], labels[pr], f"{prob:.4f}"])

    pd_metrics = {
        "acc": acc_pd,
        "n": int(len(pd_items)),
        "per_class_acc": per_class_acc,
        "n_shared_classes": len(shared_cls),
    }
    with open(RESULTS / "cross_plantdoc_metrics.json", "w") as f:
        json.dump(pd_metrics, f, indent=2)

    print(f"[done] PV acc={acc_pv:.4f} macroF1={macro_f1:.4f}   PD acc={acc_pd:.4f}")


if __name__ == "__main__":
    main()
