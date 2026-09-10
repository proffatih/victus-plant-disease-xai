"""Does target-domain fine-tuning -- or target-domain recalibration alone -- repair calibration?

A reviewer of a calibration-under-shift result will ask two questions this
script answers on the held-out target TEST partition only (never an image used
for fine-tuning, model selection or temperature fitting):

  1. After fine-tuning on field imagery, is the classifier calibrated, and can
     it abstain safely?
  2. Without fine-tuning, does fitting a single temperature on a small labelled
     target validation split fix calibration, where the in-domain temperature
     made it worse?

For the zero-shot and the fine-tuned model it reports, on the target test
partition: accuracy, mean confidence, ECE, AURC and coverage at 20% / 50% risk,
each (a) uncalibrated, (b) with the temperature fitted on the in-domain
PlantVillage validation split, and (c) with the temperature fitted on the target
validation split.

Partitions are rebuilt with the same seeded stratification used for
fine-tuning, so they are identical to those in finetune_r3.py.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import load_split_csv, index_plantdoc_shared
from plantwild import index_plantwild_shared
from finetune_r3 import stratify
from abstention import logits_for, fit_temperature, risk_coverage, coverage_at_risk, ece
from model import HybridLeafClassifier, SingleBackbone

RES = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parents[1] / "results"))


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    c2i = ck["class_to_idx"]
    name, fusion = ck.get("model_name", "hybrid"), ck.get("fusion", "gated_linear")
    if name == "hybrid":
        m = HybridLeafClassifier(len(c2i), img_size_tr=ck["img_size"], pretrained=False, fusion=fusion)
    elif name == "effv2s":
        m = SingleBackbone(len(c2i), "tf_efficientnetv2_s.in21k_ft_in1k", ck["img_size"], False)
    else:
        m = SingleBackbone(len(c2i), "swinv2_tiny_window8_256.ms_in1k", ck["img_size"], False)
    m.load_state_dict(ck["model_state"])
    return m.to(device).eval(), c2i, ck["img_size"]


def target_partitions(dataset, c2i, seed=42, val_frac=0.15):
    if dataset == "plantdoc":
        tr_all = [(p, c) for p, c in index_plantdoc_shared(c2i, splits=("train",)) if c in c2i]
        test = [(p, c) for p, c in index_plantdoc_shared(c2i, splits=("test",)) if c in c2i]
        _, val = stratify(tr_all, [1 - val_frac, val_frac], seed)
    else:
        _, val, test = stratify(index_plantwild_shared(c2i), [0.70, 0.10, 0.20], seed)
    return val, test


def summarise(logits, y, T):
    p = (logits / T).softmax(-1)
    conf, pred = p.max(-1)
    conf, correct = conf.numpy(), (pred == y).numpy()
    cov, risk, aurc = risk_coverage(conf, correct)
    return {
        "temperature": float(T), "n": int(len(y)),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(conf.mean()),
        "mean_confidence_correct": float(conf[correct].mean()) if correct.any() else None,
        "mean_confidence_incorrect": float(conf[~correct].mean()) if (~correct).any() else None,
        "ece_15bin": ece(conf, correct),
        "aurc": aurc,
        "coverage_at_risk_0.20": coverage_at_risk(cov, risk, 0.20),
        "coverage_at_risk_0.50": coverage_at_risk(cov, risk, 0.50),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--ft_ckpt", required=True)
    ap.add_argument("--dataset", choices=["plantdoc", "plantwild"], required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out = {"dataset": args.dataset, "tag": args.tag, "models": {}}
    pv_val = load_split_csv(RES / "split_val.csv")
    for name, ck in (("zero_shot", args.base_ckpt), ("fine_tuned", args.ft_ckpt)):
        model, c2i, S = load_model(ck, device)
        val, test = target_partitions(args.dataset, c2i)
        print(f"[{name}] target val={len(val)} test={len(test)}", flush=True)
        lg_pv, y_pv = logits_for(model, pv_val, c2i, S, device)
        lg_val, y_val = logits_for(model, val, c2i, S, device)
        lg_te, y_te = logits_for(model, test, c2i, S, device)
        T_in, T_tgt = fit_temperature(lg_pv, y_pv), fit_temperature(lg_val, y_val)
        out["models"][name] = {
            "ckpt": str(ck),
            "uncalibrated": summarise(lg_te, y_te, 1.0),
            "T_indomain_val": summarise(lg_te, y_te, T_in),
            "T_target_val": summarise(lg_te, y_te, T_tgt),
        }
        for cal, r in out["models"][name].items():
            if cal == "ckpt":
                continue
            print(f"  {cal:15s} T={r['temperature']:.3f} acc={r['accuracy']:.4f} conf={r['mean_confidence']:.3f} "
                  f"ECE={r['ece_15bin']:.3f} AURC={r['aurc']:.3f} cov@20%={r['coverage_at_risk_0.20']:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    with open(RES / f"eval_finetuned_{args.tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote eval_finetuned_{args.tag}.json")


if __name__ == "__main__":
    main()
