"""R3 / E12 — statistical reporting for the ablation.

Repairs the three problems Reviewer 5 raised:
  (b) multiplicity: the three pairwise comparisons are Holm-adjusted and the
      analysis is declared exploratory;
  (c) power: with roughly two dozen discordant pairs the test cannot detect a
      small difference, so the *minimum significant difference* is computed --
      the smallest accuracy gap that this discordance count could have declared
      significant -- and reported alongside every null result;
  (a) seed variance: accuracy is summarised as mean +/- SD across seeds.

The exact (binomial) McNemar test is used throughout in preference to the
chi-square approximation, as the reviewer recommends at this discordance count.
"""
from __future__ import annotations
import csv, json, os, sys, itertools, glob, re
from pathlib import Path
import numpy as np
from scipy import stats

RES = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parents[1] / "results"))
B_BOOT = 10000
RNG = np.random.default_rng(20260903)


def load_preds(path: Path):
    d = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d[r["path"]] = (r["true"], r["pred"])
    return d


def exact_mcnemar(b: int, c: int):
    """Two-sided exact binomial McNemar test on discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0, n
    p = float(stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    return p, n


def min_significant_difference(n_disc: int, n_total: int, alpha=0.05):
    """Smallest |b-c| that would be significant at this discordance count,
    expressed as an accuracy difference in percentage points."""
    if n_disc == 0:
        return None, None
    for k in range(n_disc // 2, -1, -1):
        if stats.binomtest(k, n_disc, 0.5, alternative="two-sided").pvalue < alpha:
            b, c = n_disc - k, k
            return abs(b - c), 100.0 * abs(b - c) / n_total
    return None, None


def holm(pvals: dict):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def boot_ci(correct: np.ndarray, B=B_BOOT):
    n = len(correct)
    idx = RNG.integers(0, n, size=(B, n))
    accs = correct[idx].mean(axis=1)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def main():
    # Collect every PlantVillage-test prediction file produced by evaluate_r3.
    files = sorted(glob.glob(str(RES / "preds_*_plantvillage_test_aspect.csv")))
    if not files:
        print("[warn] no prediction files yet; run evaluate_r3.py first")
        return
    models = {}
    for f in files:
        tag = re.search(r"preds_(.+)_plantvillage_test_aspect\.csv", Path(f).name).group(1)
        models[tag] = load_preds(Path(f))
    print(f"[info] {len(models)} runs: {', '.join(sorted(models))}")

    common = set.intersection(*(set(d) for d in models.values()))
    common = sorted(common)
    N = len(common)
    print(f"[info] {N} test images common to every run")

    out = {"n_test": N, "note": "exploratory analysis; Holm-adjusted across the "
                                "pairwise comparisons reported in the manuscript",
           "bootstrap_B": B_BOOT, "per_run": {}, "pairwise": {}, "by_config": {}}

    corr = {}
    for tag, d in models.items():
        c = np.array([d[p][0] == d[p][1] for p in common])
        corr[tag] = c
        lo, hi = boot_ci(c)
        out["per_run"][tag] = {"accuracy": float(c.mean()),
                               "ci95_bootstrap": [lo, hi], "n": N}

    # ---- seed aggregation --------------------------------------------------
    by_cfg = {}
    for tag in models:
        cfg = re.sub(r"_s\d+$", "", tag)
        by_cfg.setdefault(cfg, []).append(tag)
    for cfg, tags in sorted(by_cfg.items()):
        accs = np.array([corr[t].mean() for t in tags])
        out["by_config"][cfg] = {
            "n_seeds": len(tags), "seeds": sorted(tags),
            "accuracy_mean": float(accs.mean()),
            "accuracy_sd": float(accs.std(ddof=1)) if len(accs) > 1 else None,
            "accuracy_min": float(accs.min()), "accuracy_max": float(accs.max()),
        }
        sd = out["by_config"][cfg]["accuracy_sd"]
        print(f"  {cfg:28s} n={len(tags)} acc={accs.mean()*100:.3f}"
              + (f" +/- {sd*100:.3f}" if sd is not None else "") + " %")

    # ---- pairwise exact McNemar on the seed-42 runs ------------------------
    seed42 = {re.sub(r"_s42$", "", t): t for t in models if t.endswith("_s42")}
    raw_p = {}
    for a, b in itertools.combinations(sorted(seed42), 2):
        ta, tb = seed42[a], seed42[b]
        ca, cb = corr[ta], corr[tb]
        n01 = int((~ca & cb).sum())      # a wrong, b right
        n10 = int((ca & ~cb).sum())      # a right, b wrong
        p, nd = exact_mcnemar(n10, n01)
        msd_k, msd_pp = min_significant_difference(nd, N)
        key = f"{a}_vs_{b}"
        raw_p[key] = p
        out["pairwise"][key] = {
            "n01_a_wrong_b_right": n01, "n10_a_right_b_wrong": n10,
            "n_discordant": nd,
            "acc_a": float(ca.mean()), "acc_b": float(cb.mean()),
            "acc_diff_pp": float(100 * (ca.mean() - cb.mean())),
            "exact_mcnemar_p": p,
            "min_significant_abs_disagreement": msd_k,
            "min_significant_difference_pp": msd_pp,
        }
    for k, adj in holm(raw_p).items():
        out["pairwise"][k]["holm_adjusted_p"] = adj

    for k, v in out["pairwise"].items():
        print(f"  {k}: diff={v['acc_diff_pp']:+.3f} pp  n_disc={v['n_discordant']}  "
              f"p={v['exact_mcnemar_p']:.4f}  holm={v['holm_adjusted_p']:.4f}  "
              f"MSD={v['min_significant_difference_pp']:.3f} pp"
              if v["min_significant_difference_pp"] else f"  {k}: (no discordance)")

    with open(RES / "stats_r3.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {RES/'stats_r3.json'}")


if __name__ == "__main__":
    main()
