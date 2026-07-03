"""Fill numeric placeholders in manuscript.tex and abstract_plain.txt from the
JSON metric files produced by evaluate.py and the training log."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

ROOT = Path("/home/victus/Papers/Victus_Pardus_0011_Plant_Disease_XAI/Frontiers_Plant_Science")
RESULTS = ROOT / "results"

def read_last_epoch():
    """Return dict with val_acc, val_top5, plateau_epoch."""
    log = RESULTS / "train_log_hybrid.csv"
    rows = list(csv.DictReader(open(log)))
    best_val = 0.0; best_ep = 0; best_top5 = 0.0
    for r in rows:
        va = float(r["val_acc"])
        if va > best_val:
            best_val = va; best_ep = int(r["epoch"]); best_top5 = float(r["val_top5"])
    # plateau = epoch where val_acc first came within 0.5% of best
    plateau = best_ep
    for r in rows:
        if float(r["val_acc"]) >= best_val - 0.005:
            plateau = int(r["epoch"]); break
    return {"val_acc": best_val, "val_top5": best_top5, "plateau_epoch": plateau}


def main():
    tr = read_last_epoch()
    pv = json.load(open(RESULTS / "test_pv_metrics.json"))
    pd = json.load(open(RESULTS / "cross_plantdoc_metrics.json"))

    subst = {
        "PLACEHOLDER_VAL_ACC":        f"{tr['val_acc']*100:.2f}\\%",
        "PLACEHOLDER_VAL_TOP5":       f"{tr['val_top5']*100:.2f}\\%",
        "PLACEHOLDER_PLATEAU_EPOCH":  str(tr["plateau_epoch"]),
        "PLACEHOLDER_TEST_ACC":       f"{pv['acc']*100:.2f}\\%",
        "PLACEHOLDER_MACRO_F1":       f"{pv['macro_f1']*100:.2f}\\%",
        "PLACEHOLDER_WEIGHTED_F1":    f"{pv['weighted_f1']*100:.2f}\\%",
        "PLACEHOLDER_LAT":            f"{pv['latency_per_img_ms']:.2f}",
        "PLACEHOLDER_PD_ACC":         f"{pd['acc']*100:.2f}\\%",
        "PLACEHOLDER_PD_N":           str(pd["n"]),
        "PLACEHOLDER_PD_SHARED":      str(pd["n_shared_classes"]),
        "PLACEHOLDER_GAP":            f"{(pv['acc']-pd['acc'])*100:.1f}",
    }

    # plain-text substitutions (no LaTeX escape for %)
    subst_plain = {k: v.replace("\\%", "%") for k, v in subst.items()}

    for target, sub in [(ROOT / "manuscript.tex", subst),
                        (ROOT / "submission" / "abstract_plain.txt", subst_plain)]:
        s = target.read_text()
        for k, v in sub.items():
            s = s.replace(k, v)
        target.write_text(s)
        print(f"[fill] {target.name} substitutions:", ", ".join(f"{k}={v}" for k, v in sub.items()))


if __name__ == "__main__":
    main()
