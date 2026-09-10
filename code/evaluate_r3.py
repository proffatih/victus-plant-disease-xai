"""R3 — one evaluation entry point for every benchmark and every repair.

Scores a checkpoint on the PlantVillage test split, the PlantDoc shared-label
subset and the PlantWild shared-label subset (E9), always in FP32 (E11c), and
for the field sets under both preprocessing schemes (E7) so the contribution of
the aspect-ratio distortion can be separated from the domain shift.

Writes per-image predictions with the full softmax confidence, plus a metrics
JSON containing accuracy, macro-F1, balanced accuracy, MCC, crop-species
accuracy, prediction-collapse statistics and calibration (ECE/MCE).
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import (LeafDataset, build_transforms, load_split_csv,
                     index_plantdoc_shared)
from plantwild import index_plantwild_shared
from model import HybridLeafClassifier, SingleBackbone

RES = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parents[1] / "results"))


def load_model(ckpt_path: Path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    c2i = ck["class_to_idx"]
    name = ck.get("model_name", "hybrid")
    fusion = ck.get("fusion", "gated_linear")
    if name == "hybrid":
        m = HybridLeafClassifier(len(c2i), img_size_tr=ck["img_size"], pretrained=False, fusion=fusion)
    elif name == "effv2s":
        m = SingleBackbone(len(c2i), "tf_efficientnetv2_s.in21k_ft_in1k", ck["img_size"], False)
    else:
        m = SingleBackbone(len(c2i), "swinv2_tiny_window8_256.ms_in1k", ck["img_size"], False)
    m.load_state_dict(ck["model_state"])
    return m.to(device).eval(), c2i, ck


@torch.no_grad()
def predict(model, loader, device):
    """FP32 forward pass; returns labels, predictions and top-1 confidence."""
    T, P, C = [], [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True)).float()
        pr = logits.softmax(-1)
        conf, pred = pr.max(-1)
        T.append(y.numpy()); P.append(pred.cpu().numpy()); C.append(conf.cpu().numpy())
    return np.concatenate(T), np.concatenate(P), np.concatenate(C)


def ece(conf, correct, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    e = m = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        gap = abs(correct[mask].mean() - conf[mask].mean())
        e += mask.mean() * gap
        m = max(m, gap)
    return float(e), float(m)


def metrics(true, pred, conf, labels, tag):
    K = len(labels)
    acc = float((pred == true).mean())
    f1s, recs = [], []
    for c in range(K):
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        sup = tp + fn
        if sup == 0 and tp + fp == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / sup if sup else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        if sup:
            recs.append(rec)
    C = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(true, pred):
        C[t, p] += 1
    t_k, p_k = C.sum(1).astype(float), C.sum(0).astype(float)
    s, c_ = float(C.sum()), float(np.trace(C))
    den = np.sqrt(max(s * s - float(p_k @ p_k), 0)) * np.sqrt(max(s * s - float(t_k @ t_k), 0))
    mcc = float((c_ * s - float(t_k @ p_k)) / den) if den > 0 else 0.0

    crop_t = np.array([labels[i].split("___")[0] for i in true])
    crop_p = np.array([labels[i].split("___")[0] for i in pred])
    uniq, cnt = np.unique(pred, return_counts=True)
    order = np.argsort(-cnt)
    correct = (pred == true)
    e, mx = ece(conf, correct)
    return {
        "tag": tag, "n": int(len(true)),
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "balanced_accuracy": float(np.mean(recs)) if recs else 0.0,
        "mcc": mcc,
        "n_true_classes": int(len(set(true.tolist()))),
        "n_distinct_predicted_classes": int(len(uniq)),
        "crop_species_accuracy": float((crop_t == crop_p).mean()),
        "top_predicted_classes": [[labels[uniq[i]], int(cnt[i]), float(cnt[i] / len(pred))]
                                  for i in order[:5]],
        "max_single_class_prediction_share": float(cnt.max() / len(pred)),
        "mean_confidence": float(conf.mean()),
        "mean_confidence_correct": float(conf[correct].mean()) if correct.any() else None,
        "mean_confidence_incorrect": float(conf[~correct].mean()) if (~correct).any() else None,
        "ece_15bin": e, "mce_15bin": mx,
        "overconfidence_gap": float(conf.mean() - acc),
    }


def dump(path, items, true, pred, conf, labels):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "top1_prob"])
        for (p, _), t, pr, c in zip(items, true, pred, conf):
            w.writerow([p, labels[t], labels[pr], f"{c:.6f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, c2i, ck = load_model(Path(args.ckpt), device)
    labels = [k for k, _ in sorted(c2i.items(), key=lambda kv: kv[1])]
    print(f"[info] {args.tag}: model={ck.get('model_name')} fusion={ck.get('fusion')} "
          f"epoch={ck.get('epoch')} val_acc={ck.get('val_acc'):.4f}")

    out = {"ckpt": str(args.ckpt), "tag": args.tag, "eval_precision": "fp32",
           "model_name": ck.get("model_name"), "fusion": ck.get("fusion"),
           "seed": ck.get("seed"), "benchmarks": {}}

    def run(name, items, preproc):
        tf = build_transforms(ck["img_size"], train=False, preproc=preproc)
        dl = DataLoader(LeafDataset(items, c2i, tf), batch_size=args.batch,
                        shuffle=False, num_workers=args.workers, pin_memory=True)
        t, p, c = predict(model, dl, device)
        m = metrics(t, p, c, labels, f"{name}/{preproc}")
        out["benchmarks"][f"{name}_{preproc}"] = m
        dump(RES / f"preds_{args.tag}_{name}_{preproc}.csv", items, t, p, c, labels)
        print(f"  [{name}/{preproc}] n={m['n']} acc={m['accuracy']:.4f} "
              f"macroF1={m['macro_f1']:.4f} conf={m['mean_confidence']:.3f} "
              f"ECE={m['ece_15bin']:.3f} collapse={m['max_single_class_prediction_share']:.3f}",
              flush=True)

    run("plantvillage_test", load_split_csv(RES / "split_test.csv"), "aspect")
    pd_items = [(p, c) for p, c in index_plantdoc_shared(c2i) if c in c2i]
    run("plantdoc", pd_items, "aspect")
    run("plantdoc", pd_items, "square")            # E7: quantify the distortion
    pw_items = index_plantwild_shared(c2i)
    run("plantwild", pw_items, "aspect")
    run("plantwild", pw_items, "square")

    with open(RES / f"eval_{args.tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote eval_{args.tag}.json")


if __name__ == "__main__":
    main()
