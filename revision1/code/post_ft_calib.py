"""Calibration and selective prediction on the held-out field test partitions after
fine-tuning (full and k-shot), for every seed: no scaling vs temperature fitted on the
target validation split. Scores: MSP, entropy, energy. -> results/post_ft_calib.json"""
import json, re
import numpy as np
from scipy.special import softmax, logsumexp
from posthoc_rev1 import calib_block, selective_summary, fit_temperature, RUN, OUTD

FT = RUN / "ft"
out = {}
for f in sorted(FT.glob("logits_*.npz")):
    tag = f.stem[len("logits_"):]
    z = np.load(f)
    zv, yv, zt, yt = z["val_logits"], z["val_y"], z["test_logits"], z["test_y"]
    T = fit_temperature(zv, yv)
    r = {"T_target": T}
    for name, t in (("none", 1.0), ("target_T", T)):
        p = softmax(zt / t, 1)
        corr = p.argmax(1) == yt
        r[name] = calib_block(p, yt)
        r[name]["sel_msp"] = selective_summary(p.max(1), corr)
    p = softmax(zt, 1); corr = p.argmax(1) == yt
    r["sel_entropy"] = selective_summary((p * np.log(np.clip(p, 1e-12, 1))).sum(1), corr)
    r["sel_energy"] = selective_summary(logsumexp(zt, 1), corr)
    out[tag] = r

# aggregate over seeds: key = tag without _s<seed>
agg = {}
for tag, r in out.items():
    key = re.sub(r"_s\d+", "", tag)
    agg.setdefault(key, []).append(r)
ms = lambda v: [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)]
summ = {}
for key, rs in agg.items():
    s = {"T_target": ms([r["T_target"] for r in rs])}
    for name in ("none", "target_T"):
        s[name] = {m: ms([r[name][m] for r in rs]) for m in ("acc", "mean_conf", "ece", "nll", "brier")}
        s[name]["sel_msp"] = {m: ms([r[name]["sel_msp"][m] for r in rs]) for m in rs[0][name]["sel_msp"]}
    for sc in ("sel_entropy", "sel_energy"):
        s[sc] = {m: ms([r[sc][m] for r in rs]) for m in rs[0][sc]}
    summ[key] = s
json.dump({"per_run": out, "summary": summ}, open(OUTD / "post_ft_calib.json", "w"), indent=1)
f = lambda x, d=3: f"{x[0]:.{d}f}±{x[1]:.{d}f}(n{x[2]})"
for k, s in sorted(summ.items()):
    print(k, "T", f(s["T_target"], 2), "| acc", f(s["none"]["acc"]), "conf", f(s["none"]["mean_conf"]),
          "ECE", f(s["none"]["ece"]), "->", f(s["target_T"]["ece"]), "| AURC", f(s["none"]["sel_msp"]["aurc"]),
          "cov20", f(s["none"]["sel_msp"]["cov_at_risk_0.20"]), "Eaurc msp/ent/en",
          f(s["none"]["sel_msp"]["e_aurc"]), f(s["sel_entropy"]["e_aurc"]), f(s["sel_energy"]["e_aurc"]))
