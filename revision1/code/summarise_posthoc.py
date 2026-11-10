"""Summarise posthoc_rev1.json: mean +- SD over the 12 models (and over the four
3-seed ensembles), per benchmark. Writes results/posthoc_summary.json and prints
compact tables."""
import json
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parents[1] / "results"
D = json.load(open(RES / "posthoc_rev1.json"))
M = D["members"]; E = D["ensembles"]
EVAL = ["pv_test", "plantdoc", "plantwild"]


def ms(vals):
    v = np.array([x for x in vals if x is not None], float)
    return [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, int(len(v))]


out = {"calibration": {}, "selective": {}, "ood_auroc": {}, "gate": {}, "ensembles": {}, "temperature": ms([m["temperature"] for m in M.values()])}
for k in EVAL:
    meths = next(iter(M.values()))["calibration"][k].keys()
    out["calibration"][k] = {me: {mt: ms([m["calibration"][k][me][mt] for m in M.values()])
                                  for mt in ("acc", "mean_conf", "ece", "nll", "brier")} for me in meths}
    scores = [s for s in next(iter(M.values()))["selective"][k] if not s.startswith("_")]
    out["selective"][k] = {s: {mt: ms([m["selective"][k][s][mt] for m in M.values()])
                               for mt in next(iter(M.values()))["selective"][k][s]} for s in scores}
    out["gate"][k] = {mt: ms([m["gate"][k][mt] for m in M.values()])
                      for mt in ("accepted_frac", "acc_accepted", "acc_deferred", "errors_caught_frac")}
    if k != "pv_test":
        out["ood_auroc"][k] = {s: ms([m["ood_auroc"][k][s] for m in M.values()]) for s in next(iter(M.values()))["ood_auroc"][k]}
    out["ensembles"][k] = {
        "calibration": {mt: ms([e["calibration"][k][mt] for e in E.values()]) for mt in ("acc", "mean_conf", "ece", "nll", "brier")},
        "selective": {s: {mt: ms([e["selective"][k][s][mt] for e in E.values()]) for mt in ("aurc", "e_aurc", "cov_at_risk_0.20", "cov_at_risk_0.50", "acc_at_cov_0.10", "acc_at_cov_0.25")}
                      for s in ("ens_msp", "neg_ens_mutual_info", "seed_agreement")},
        "per_arch_acc": {a: e["calibration"][k]["acc"] for a, e in E.items()}}
json.dump(out, open(RES / "posthoc_summary.json", "w"), indent=1)

f = lambda x, d=3: f"{x[0]:.{d}f}±{x[1]:.{d}f}"
for k in ("plantdoc", "plantwild"):
    print(f"\n== {k}: calibration (acc / conf / ECE / NLL)")
    for me, v in out["calibration"][k].items():
        print(f"  {me:32s} {f(v['acc'])} {f(v['mean_conf'])} {f(v['ece'])} {f(v['nll'],2)}")
    e = out["ensembles"][k]["calibration"]
    print(f"  {'deep_ensemble_3seeds (4 archs)':32s} {f(e['acc'])} {f(e['mean_conf'])} {f(e['ece'])} {f(e['nll'],2)}")
    print(f"== {k}: selective (AURC / E-AURC / cov@20 / cov@50 / acc@10%)")
    for s, v in out["selective"][k].items():
        print(f"  {s:22s} {f(v['aurc'])} {f(v['e_aurc'])} {f(v['cov_at_risk_0.20'])} {f(v['cov_at_risk_0.50'])} {f(v['acc_at_cov_0.10'])}")
    for s, v in out["ensembles"][k]["selective"].items():
        print(f"  ens:{s:18s} {f(v['aurc'])} {f(v['e_aurc'])} {f(v['cov_at_risk_0.20'])} {f(v['cov_at_risk_0.50'])} {f(v['acc_at_cov_0.10'])}")
    print("  OOD AUROC", {s: f(v) for s, v in out["ood_auroc"][k].items()})
    print("  gate", {s: f(v) for s, v in out["gate"][k].items()})
print("\npv_test gate", {s: f(v) for s, v in out["gate"]["pv_test"].items()})
print("pv_test calib", {me: f(v["ece"]) for me, v in out["calibration"]["pv_test"].items()})
print("T", out["temperature"])
