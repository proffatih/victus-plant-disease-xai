"""Bootstrap confidence intervals and McNemar tests for the manuscript.

Reads whatever prediction files are present under RESULTS_DIR and writes
stats_summary.json. Safe to run repeatedly as more predictions appear.

- PlantVillage test CIs  <- results/test_pv_predictions.csv (path,true,pred,prob)
- PlantDoc zero-shot CI  <- results/cross_plantdoc_predictions.csv
- Ablation CIs + McNemar <- results_revision/{tag}_test_predictions.csv (true_idx,pred_idx)
"""
from __future__ import annotations
import os, json, csv
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

ORIG = Path(os.environ.get("RESULTS_DIR", "results"))
REV = Path(os.environ.get("RESULTS_DIR", "results"))
RNG = np.random.default_rng(1234)
B = 10000


def boot_ci(fn, *arrs, b=B):
    n = len(arrs[0])
    vals = np.empty(b)
    idx_all = np.arange(n)
    for i in range(b):
        idx = RNG.choice(idx_all, size=n, replace=True)
        vals[i] = fn(*[a[idx] for a in arrs])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def read_named(path):
    y, p = [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            y.append(row["true"]); p.append(row["pred"])
    labels = sorted(set(y))
    li = {c: i for i, c in enumerate(labels)}
    return np.array([li[c] for c in y]), np.array([li.get(c, -1) for c in p])


def read_idx(path):
    y, p = [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            y.append(int(row["true_idx"])); p.append(int(row["pred_idx"]))
    return np.array(y), np.array(p)


def acc(y, p): return float((y == p).mean())
def mf1(y, p): return float(f1_score(y, p, average="macro", labels=np.unique(y), zero_division=0))


def mcnemar(y, pa, pb):
    a_correct = (pa == y); b_correct = (pb == y)
    n01 = int(np.sum(a_correct & ~b_correct))  # a right, b wrong
    n10 = int(np.sum(~a_correct & b_correct))  # a wrong, b right
    n = n01 + n10
    if n == 0:
        return {"n01": n01, "n10": n10, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(n01 - n10) - 1) ** 2 / n  # continuity-corrected
    from math import erfc, sqrt
    p = erfc(sqrt(chi2 / 2))  # survival of chi2_1 = erfc(sqrt(x/2))
    return {"n01": n01, "n10": n10, "chi2": float(chi2), "p_value": float(p)}


out = {}

# --- PlantVillage in-domain (hybrid) ---
pv = ORIG / "test_pv_predictions.csv"
if pv.exists():
    y, p = read_named(pv)
    out["plantvillage_test_hybrid"] = {
        "n": int(len(y)), "acc": acc(y, p), "acc_ci95": boot_ci(acc, y, p),
        "macro_f1": mf1(y, p), "macro_f1_ci95": boot_ci(mf1, y, p),
    }

# --- PlantDoc zero-shot (hybrid, union subset as in manuscript) ---
pd = ORIG / "cross_plantdoc_predictions.csv"
if pd.exists():
    y, p = read_named(pd)
    out["plantdoc_zeroshot_hybrid"] = {
        "n": int(len(y)), "acc": acc(y, p), "acc_ci95": boot_ci(acc, y, p),
    }

# --- Ablation (revision) ---
preds = {}
for tag in ["hybrid", "effv2s", "swinv2t"]:
    f = REV / f"{tag}_test_predictions.csv"
    if f.exists():
        y, p = read_idx(f); preds[tag] = (y, p)
        out[f"ablation_{tag}"] = {
            "n": int(len(y)), "acc": acc(y, p), "acc_ci95": boot_ci(acc, y, p),
            "macro_f1": mf1(y, p), "macro_f1_ci95": boot_ci(mf1, y, p),
        }
# McNemar hybrid vs each backbone (same test split / indices)
if "hybrid" in preds:
    yh, ph = preds["hybrid"]
    for tag in ["effv2s", "swinv2t"]:
        if tag in preds:
            yb, pb = preds[tag]
            if np.array_equal(yh, yb):
                out[f"mcnemar_hybrid_vs_{tag}"] = mcnemar(yh, ph, pb)

# --- PlantDoc fine-tuning ---
ftm = REV / "plantdoc_finetune_metrics.json"
if ftm.exists():
    out["plantdoc_finetune"] = json.loads(ftm.read_text())

(REV if REV.exists() else ORIG).mkdir(parents=True, exist_ok=True)
Path(REV / "stats_summary.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
