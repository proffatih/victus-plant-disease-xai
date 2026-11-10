"""REV1 (R2-1, R3-1): architecture comparison under the controlled protocol.

Reads the logits dumped by ctrl_eval.py for the 12 controlled models (and, for the
same statistics on the reference models, the dumps in ~/RUN_OUT/0011_rev1) and writes
results/ctrl_stats.json:

  per architecture (mean, sd over three seeds): PlantVillage test accuracy, and on
    PlantDoc and PlantWild top-1 accuracy, mean confidence and ECE (15 bins);
  pairwise exact McNemar tests on the PlantVillage test split for every seed
    (Holm-adjusted within seed) with the minimum significant difference, and on the
    field sets for seed 42;
  reference vs controlled protocol for each architecture: change in field accuracy.
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).parent))
from stats_r3 import exact_mcnemar, min_significant_difference, holm  # noqa: E402

REF = Path.home() / "RUN_OUT" / "0011_rev1"
CTRL = REF / "ctrl"
OUTD = Path(__file__).resolve().parents[1] / "results"
ARCHS = ["effv2s", "swinv2t", "hybrid_gated", "hybrid_tokenattn"]
SEEDS = [42, 1337, 2024]
SETS = ["pv_test", "plantdoc", "plantwild"]


def ece(conf, correct, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def load(d, tag):
    B = np.load(d / f"feats_{tag}.npz", allow_pickle=False)
    out = {}
    for k in SETS:
        P = softmax(B[f"{k}_logits"].astype(np.float64), 1)
        y = B[f"{k}_y"]
        out[k] = {"correct": P.argmax(1) == y, "conf": P.max(1)}
    return out


def ms(v):
    v = np.asarray(v, float)
    return [float(v.mean()), float(v.std(ddof=1)), int(len(v))]


def per_arch(d):
    res = {}
    for a in ARCHS:
        M = [load(d, f"{a}_s{s}") for s in SEEDS]
        res[a] = {k: {"acc": ms([m[k]["correct"].mean() for m in M]),
                      "conf": ms([m[k]["conf"].mean() for m in M]),
                      "ece": ms([ece(m[k]["conf"], m[k]["correct"]) for m in M])}
                  for k in SETS}
    return res


def pairwise(seed, k):
    C = {a: load(CTRL, f"{a}_s{seed}")[k]["correct"] for a in ARCHS}
    n = len(next(iter(C.values())))
    out, raw = {}, {}
    for a, b in itertools.combinations(ARCHS, 2):
        ca, cb = C[a], C[b]
        n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        p, nd = exact_mcnemar(n10, n01)
        _, msd = min_significant_difference(nd, n)
        key = f"{a}_vs_{b}"
        raw[key] = p
        out[key] = {"diff_pp": float(100 * (ca.mean() - cb.mean())), "n_discordant": nd,
                    "p": p, "min_sig_diff_pp": msd}
    for key, adj in holm(raw).items():
        out[key]["holm_p"] = adj
    return out


def main():
    res = {"ctrl": per_arch(CTRL), "ref": per_arch(REF), "pairwise": {}}
    for s in SEEDS:
        res["pairwise"][f"pv_test_s{s}"] = pairwise(s, "pv_test")
    for k in ("plantdoc", "plantwild"):
        res["pairwise"][f"{k}_s42"] = pairwise(42, k)
    json.dump(res, open(OUTD / "ctrl_stats.json", "w"), indent=1)
    for a in ARCHS:
        c, r = res["ctrl"][a], res["ref"][a]
        print(f"{a:18s} PV {c['pv_test']['acc'][0]*100:.2f}±{c['pv_test']['acc'][1]*100:.2f} | "
              + " | ".join(f"{k} ctrl {c[k]['acc'][0]*100:.1f}±{c[k]['acc'][1]*100:.1f} "
                           f"ece {c[k]['ece'][0]:.3f} conf {c[k]['conf'][0]*100:.0f} "
                           f"(ref {r[k]['acc'][0]*100:.1f}, ece {r[k]['ece'][0]:.3f})"
                           for k in ("plantdoc", "plantwild")))
    for key, v in res["pairwise"].items():
        print(key, {p: (round(x["diff_pp"], 3), x["n_discordant"], round(x["holm_p"], 3),
                        None if x["min_sig_diff_pp"] is None else round(x["min_sig_diff_pp"], 3))
                    for p, x in v.items()})


if __name__ == "__main__":
    main()
