"""Single source of truth for every number quoted in the manuscript.

Reads the result files in R3/results and writes NEW_submission/numbers.json.
The manuscript must not contain a number that is not in this file.
"""
import json, glob, re, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
import os
RES = Path(os.environ.get("RESULTS_DIR") or
           (HERE / "results" if (HERE / "results").is_dir() else HERE.parent / "R3" / "results"))
OUT = HERE / "numbers.json"

def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None

def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": st.mean(vals), "sd": st.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals), "n": len(vals)}

ARCH = {"effv2s": "EfficientNetV2-S", "swinv2t": "Swin V2-T",
        "hybrid_gated": "Hybrid, gated linear fusion", "hybrid_tokenattn": "Hybrid, token cross-attention"}
BENCH = ["plantvillage_test_aspect", "plantdoc_aspect", "plantdoc_square", "plantwild_aspect", "plantwild_square"]
MET = ["accuracy", "macro_f1", "balanced_accuracy", "mcc", "mean_confidence", "mean_confidence_correct",
       "mean_confidence_incorrect", "ece_15bin", "mce_15bin", "crop_species_accuracy",
       "max_single_class_prediction_share", "n_distinct_predicted_classes", "overconfidence_gap"]

out = {"models": {}, "by_arch": {}, "all_models": {}}

evals = {}
for f in sorted(glob.glob(str(RES / "eval_*.json"))):
    j = json.loads(Path(f).read_text())
    if "benchmarks" not in j:          # eval_finetuned_*.json has a different schema
        continue
    tag = re.search(r"eval_(.+)\.json", Path(f).name).group(1)
    evals[tag] = j
    out["models"][tag] = {b: {m: evals[tag]["benchmarks"][b].get(m) for m in MET + ["n", "top_predicted_classes"]}
                          for b in BENCH if b in evals[tag]["benchmarks"]}

for a in ARCH:
    tags = [t for t in evals if re.sub(r"_s\d+$", "", t) == a]
    out["by_arch"][a] = {"label": ARCH[a], "seeds": sorted(tags),
                         **{b: {m: agg([evals[t]["benchmarks"][b].get(m) for t in tags]) for m in MET}
                            for b in BENCH}}
out["all_models"] = {b: {m: agg([evals[t]["benchmarks"][b].get(m) for t in evals]) for m in MET} for b in BENCH}
out["all_models"]["n_models"] = len(evals)

# abstention
ab = {}
for f in sorted(glob.glob(str(RES / "abstention_*.json"))):
    tag = re.search(r"abstention_(.+)\.json", Path(f).name).group(1)
    ab[tag] = json.loads(Path(f).read_text())
out["abstention"] = {"per_model": ab}
def abget(j, bench, cal, score, key):
    return j["benchmarks"][bench][cal][score].get(key)
summ = {}
for bench in ("plantvillage_test", "plantdoc", "plantwild"):
    summ[bench] = {}
    for cal in ("uncalibrated", "temp_scaled"):
        summ[bench][cal] = {}
        for key in ("aurc", "coverage_at_risk_0.10", "coverage_at_risk_0.20", "coverage_at_risk_0.50",
                    "selective_acc_at_coverage_0.10", "selective_acc_at_coverage_0.25",
                    "selective_acc_at_coverage_0.50", "ece_15bin", "mean_confidence"):
            summ[bench][cal][key] = agg([abget(j, bench, cal, "msp", key) for j in ab.values()])
summ["temperature"] = agg([j["temperature_from_indomain_val"] for j in ab.values()])
summ["ood_auroc_msp_uncal"] = {fld: agg([j["ood_detection_auroc_indomain_vs_field"]["uncalibrated"][fld]["msp"]
                                          for j in ab.values()]) for fld in ("plantdoc", "plantwild")}
summ["ood_auroc_best_score_uncal"] = {fld: {s: agg([j["ood_detection_auroc_indomain_vs_field"]["uncalibrated"][fld][s]
                                          for j in ab.values()]) for s in ("msp", "neg_entropy", "neg_energy")}
                                      for fld in ("plantdoc", "plantwild")}
summ["n_models"] = len(ab)
out["abstention"]["summary"] = summ

for k, f in [("efficiency", "efficiency_measured.json"), ("stats", "stats_r3.json"),
             ("background_bias", "background_bias.json"), ("duplicates", "duplicate_analysis.json"),
             ("r2_released_predictions", "analysis_predictions.json"), ("fusion_proof", "fusion_degeneracy_proof.json"),
             ("plantwild_mapping", "plantwild_mapping.json"),
             ("finetune_plantdoc", "ft_pd_gated_s42.json"), ("finetune_plantwild", "ft_pw_gated_s42.json"),
             ("gradcam_field", "gradcam_summary_hybrid_gated_s42_field.json"),
             ("gradcam_indomain", "gradcam_summary_hybrid_gated_s42_indomain.json")]:
    out[k] = load(f)
tf = {}
for f in glob.glob(str(RES / "test_final_*.json")):
    tag = re.search(r"test_final_(.+)\.json", Path(f).name).group(1)
    tf[tag] = json.loads(Path(f).read_text())
out["training"] = tf
out["finetune_calibration"] = {re.search(r"eval_finetuned_(.+)\.json", Path(f).name).group(1):
                              json.loads(Path(f).read_text())
                              for f in sorted(glob.glob(str(RES / "eval_finetuned_*.json")))}

OUT.write_text(json.dumps(out, indent=2))
print(f"wrote {OUT}  (models={len(evals)}, abstention={len(ab)})")
