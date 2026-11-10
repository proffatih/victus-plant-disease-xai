"""REV1 (R1-1, R1-2): what do the field errors look like biologically?

From the stored predictions of the 12 models, per benchmark:
  * per-class recall (mean over models) with host, disease and agent type;
  * recall grouped by agent type (healthy, fungus, bacterium, oomycete, virus);
  * the error breakdown: wrong host vs right host but wrong disease vs
    healthy<->diseased confusion on the right host;
  * whether per-class recall tracks the number of PlantVillage training images
    (class imbalance) or the number of field images (Spearman correlation);
  * the classes that attract the collapsed predictions.
"""
from __future__ import annotations
import csv, json, os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

RUN = Path(os.environ.get("RUN_OUT", Path.home() / "RUN_OUT" / "0011_rev1"))
RES = Path(__file__).resolve().parents[1] / "results"
R3 = Path(__file__).resolve().parents[2] / "R3" / "results"
ARCHS = ["effv2s", "swinv2t", "hybrid_gated", "hybrid_tokenattn"]
SEEDS = [42, 1337, 2024]


def main():
    c2i = json.load(open(R3 / "class_to_idx.json"))
    i2c = {v: k for k, v in c2i.items()}
    tab = {r["class"]: r for r in csv.DictReader(open(RES / "class_table.csv"))}
    host = {c: tab[c]["host"] for c in tab}
    kind = {c: ("Healthy" if tab[c]["disease"] == "Healthy" else tab[c]["agent_type"]) for c in tab}
    tags = [f"{a}_s{s}" for a in ARCHS for s in SEEDS if (RUN / f"feats_{a}_s{s}.npz").exists()]
    out = {"n_models": len(tags)}
    for ds in ("plantdoc", "plantwild"):
        rec = defaultdict(list); brk = Counter(); n_err = 0; attract = Counter(); per_model_kind = defaultdict(list)
        for t in tags:
            B = np.load(RUN / f"feats_{t}.npz")
            y, p = B[f"{ds}_y"], B[f"{ds}_logits"].argmax(1)
            for c in np.unique(y):
                rec[i2c[c]].append(float((p[y == c] == c).mean()))
            kk = defaultdict(list)
            for yi, pi in zip(y, p):
                kk[kind[i2c[yi]]].append(yi == pi)
                if yi == pi:
                    continue
                n_err += 1
                tc, pc = i2c[yi], i2c[pi]
                if host[tc] != host[pc]:
                    brk["wrong_host"] += 1
                elif kind[tc] == "Healthy" or kind[pc] == "Healthy":
                    brk["right_host_healthy_vs_diseased"] += 1
                else:
                    brk["right_host_wrong_disease"] += 1
                attract[pc] += 1
            for k, v in kk.items():
                per_model_kind[k].append(float(np.mean(v)))
        classes = sorted(rec)
        mean_rec = {c: float(np.mean(rec[c])) for c in classes}
        pv_n = np.array([int(tab[c]["pv_train"]) for c in classes])
        fld_n = np.array([int(tab[c]["plantdoc_all" if ds == "plantdoc" else "plantwild"]) for c in classes])
        r = np.array([mean_rec[c] for c in classes])
        out[ds] = {
            "per_class_recall": {c: {"recall": mean_rec[c], "host": host[c], "kind": kind[c],
                                     "pv_train": int(tab[c]["pv_train"])} for c in classes},
            "recall_by_kind": {k: {"mean": float(np.mean(v)), "sd_over_models": float(np.std(v, ddof=1)),
                                   "n_classes": sum(kind[c] == k for c in classes)}
                               for k, v in per_model_kind.items()},
            "error_breakdown": {k: v / n_err for k, v in brk.items()},
            "spearman_recall_vs_pv_train": [float(x) for x in spearmanr(r, pv_n)],
            "spearman_recall_vs_field_n": [float(x) for x in spearmanr(r, fld_n)],
            "top_attractors": [(c, n / n_err) for c, n in attract.most_common(6)],
            "classes_recall_ge_0.5": [c for c in classes if mean_rec[c] >= 0.5],
            "classes_recall_le_0.05": [c for c in classes if mean_rec[c] <= 0.05],
        }
        o = out[ds]
        print(ds, "kinds", {k: round(v["mean"], 3) for k, v in o["recall_by_kind"].items()})
        print("  errors", {k: round(v, 3) for k, v in o["error_breakdown"].items()})
        print("  rho pv", o["spearman_recall_vs_pv_train"], "rho field", o["spearman_recall_vs_field_n"])
        print("  attractors", [(c, round(v, 3)) for c, v in o["top_attractors"]])
        print("  >=0.5", o["classes_recall_ge_0.5"])
    json.dump(out, open(RES / "bio_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
