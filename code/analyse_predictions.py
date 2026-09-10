"""R3 / E6, E12, minor-1, minor-13.

Analyses that need only the released per-image predictions:
  * PlantDoc prediction collapse and crop-species-level accuracy
  * calibration (ECE, MCE, reliability bins) on PlantDoc and PlantVillage
  * PlantDoc train- vs test-partition difficulty
  * balanced accuracy and Matthews correlation coefficient on PlantVillage
  * image-count reconciliation
Writes R3/results/analysis_predictions.json and reliability bin tables.
"""
from __future__ import annotations
import json, csv
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[2]
OLD = BASE / "results"
OUT = BASE / "R3" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def read_preds(p: Path):
    rows = []
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def crop_of(cls: str) -> str:
    return cls.split("___")[0]


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e, m, table = 0.0, 0.0, []
    n = len(conf)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        k = int(mask.sum())
        if k == 0:
            table.append((lo, hi, 0, None, None, None))
            continue
        acc = float(correct[mask].mean())
        cf = float(conf[mask].mean())
        gap = abs(acc - cf)
        e += (k / n) * gap
        m = max(m, gap)
        table.append((lo, hi, k, acc, cf, gap))
    return e, m, table


def write_bins(path: Path, table):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_lo", "bin_hi", "n", "accuracy", "mean_confidence", "gap"])
        for lo, hi, k, a, c, g in table:
            w.writerow([f"{lo:.4f}", f"{hi:.4f}", k,
                        "" if a is None else f"{a:.6f}",
                        "" if c is None else f"{c:.6f}",
                        "" if g is None else f"{g:.6f}"])


res: dict = {}

# ------------------------------------------------------------------ PlantDoc
pd_rows = read_preds(OLD / "cross_plantdoc_predictions.csv")
true = np.array([r["true"] for r in pd_rows])
pred = np.array([r["pred"] for r in pd_rows])
conf = np.array([float(r["top1_prob"]) for r in pd_rows])
paths = [r["path"] for r in pd_rows]
correct = (true == pred)

n = len(pd_rows)
acc = float(correct.mean())

# (i) prediction collapse
uniq, counts = np.unique(pred, return_counts=True)
order = np.argsort(-counts)
top_pred = [(uniq[i], int(counts[i]), float(counts[i] / n)) for i in order[:8]]

# (ii) crop-species-level accuracy
crop_true = np.array([crop_of(t) for t in true])
crop_pred = np.array([crop_of(p) for p in pred])
crop_acc = float((crop_true == crop_pred).mean())

# (iii) calibration
pd_ece, pd_mce, pd_table = ece(conf, correct)
write_bins(OUT / "reliability_plantdoc.csv", pd_table)

# (iv) partition difficulty
is_test = np.array(["/test/" in p for p in paths])
is_train = np.array(["/train/" in p for p in paths])

res["plantdoc"] = {
    "n": n,
    "n_distinct_predicted_classes": int(len(uniq)),
    "accuracy": acc,
    "top_predicted_classes": top_pred,
    "crop_species_accuracy": crop_acc,
    "n_crop_species": int(len(set(crop_true.tolist()))),
    "mean_confidence": float(conf.mean()),
    "mean_confidence_correct": float(conf[correct].mean()),
    "mean_confidence_incorrect": float(conf[~correct].mean()),
    "ece_15bin": pd_ece,
    "mce_15bin": pd_mce,
    "overconfidence_gap": float(conf.mean() - acc),
    "partition_train": {"n": int(is_train.sum()), "acc": float(correct[is_train].mean())},
    "partition_test": {"n": int(is_test.sum()), "acc": float(correct[is_test].mean())},
}
res["plantdoc"]["partition_ratio_test_over_train"] = (
    res["plantdoc"]["partition_test"]["acc"] / res["plantdoc"]["partition_train"]["acc"]
)

# -------------------------------------------------------------- PlantVillage
pv_rows = read_preds(OLD / "test_pv_predictions.csv")
pv_true = np.array([r["true"] for r in pv_rows])
pv_pred = np.array([r["pred"] for r in pv_rows])
pv_conf = np.array([float(r["top1_prob"]) for r in pv_rows])
pv_correct = (pv_true == pv_pred)

classes = sorted(set(pv_true.tolist()) | set(pv_pred.tolist()))
c2i = {c: i for i, c in enumerate(classes)}
yt = np.array([c2i[c] for c in pv_true])
yp = np.array([c2i[c] for c in pv_pred])
K = len(classes)

# balanced accuracy = mean per-class recall
recalls = []
for c in range(K):
    m = yt == c
    if m.sum():
        recalls.append(float((yp[m] == c).mean()))
bal_acc = float(np.mean(recalls))

# multiclass MCC
C = np.zeros((K, K), dtype=np.int64)
for t, p in zip(yt, yp):
    C[t, p] += 1
t_k = C.sum(axis=1).astype(float)   # true totals
p_k = C.sum(axis=0).astype(float)   # predicted totals
c_ = float(np.trace(C))
s = float(C.sum())
num = c_ * s - float(t_k @ p_k)
den = np.sqrt(max(s * s - float(p_k @ p_k), 0.0)) * np.sqrt(max(s * s - float(t_k @ t_k), 0.0))
mcc = float(num / den) if den > 0 else 0.0

pv_ece, pv_mce, pv_table = ece(pv_conf, pv_correct)
write_bins(OUT / "reliability_plantvillage.csv", pv_table)

supports = t_k
res["plantvillage_test"] = {
    "n": int(s),
    "accuracy": float(pv_correct.mean()),
    "balanced_accuracy": bal_acc,
    "mcc": mcc,
    "mean_confidence": float(pv_conf.mean()),
    "ece_15bin": pv_ece,
    "mce_15bin": pv_mce,
    "class_imbalance_ratio": float(supports.max() / supports.min()),
    "min_class_support": int(supports.min()),
    "max_class_support": int(supports.max()),
}

# ------------------------------------------------------- minor-1 image counts
counts_split = {}
for name in ("train", "val", "test"):
    with open(OLD / f"split_{name}.csv") as f:
        counts_split[name] = sum(1 for _ in f) - 1
counts_split["total"] = sum(counts_split[k] for k in ("train", "val", "test"))
res["split_counts"] = counts_split
res["manuscript_claimed_total"] = 54304

with open(OUT / "analysis_predictions.json", "w") as f:
    json.dump(res, f, indent=2)

print(json.dumps(res, indent=2))
