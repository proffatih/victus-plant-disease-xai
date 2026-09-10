"""Selective prediction and abstention under domain shift.

Reviewer 5's closing observation is that a field classifier which is 76%
confident and 12% correct is a safety problem rather than an accuracy problem,
and that the deployment-relevant question is whether the model can be made to
abstain. This script answers that question quantitatively.

For each benchmark it computes three confidence scores from the same forward
pass -- maximum softmax probability, predictive entropy, and the free energy
-logsumexp(z) -- and then reports:

  * risk-coverage curves and their area (AURC): the error rate among retained
    predictions as a function of how many are retained;
  * coverage attainable at a target risk, i.e. how much of the field data the
    model could handle at a given error budget while deferring the rest;
  * selective accuracy at fixed coverage levels;
  * out-of-distribution separability (AUROC) of in-domain PlantVillage against
    each field set, which says whether a single global threshold could route
    field images to a human at all;
  * the effect of temperature scaling fitted on the in-domain validation split,
    which tests the common assumption that in-domain calibration transfers.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import LeafDataset, build_transforms, load_split_csv, index_plantdoc_shared
from plantwild import index_plantwild_shared
from model import HybridLeafClassifier, SingleBackbone

RES = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parents[1] / "results"))


@torch.no_grad()
def logits_for(model, items, c2i, img_size, device, batch=32, workers=4):
    tf = build_transforms(img_size, train=False, preproc="aspect")
    dl = DataLoader(LeafDataset(items, c2i, tf), batch_size=batch, shuffle=False,
                    num_workers=workers, pin_memory=True)
    L, Y = [], []
    for x, y in dl:
        L.append(model(x.to(device, non_blocking=True)).float().cpu())
        Y.append(y)
    return torch.cat(L), torch.cat(Y)


def scores_from(logits: torch.Tensor, T: float = 1.0):
    z = logits / T
    p = z.softmax(-1)
    msp = p.max(-1).values
    ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)
    energy = -torch.logsumexp(z, dim=-1)
    return {"msp": msp.numpy(), "neg_entropy": (-ent).numpy(), "neg_energy": (-energy).numpy()}


def risk_coverage(score: np.ndarray, correct: np.ndarray):
    """Sort by confidence descending; risk = error rate among the retained."""
    order = np.argsort(-score)
    c = correct[order].astype(float)
    n = len(c)
    cum_correct = np.cumsum(c)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = 1.0 - cum_correct / k
    aurc = float(np.trapezoid(risk, coverage))
    return coverage, risk, aurc


def coverage_at_risk(coverage, risk, target):
    """Largest coverage whose retained-set risk is at or below the target."""
    ok = np.where(risk <= target)[0]
    return float(coverage[ok[-1]]) if len(ok) else 0.0


def acc_at_coverage(coverage, risk, target_cov):
    i = int(np.searchsorted(coverage, target_cov))
    i = min(i, len(risk) - 1)
    return float(1.0 - risk[i])


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC via the rank-sum identity; `pos` should score higher if separable."""
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort().astype(float) + 1
    # average ranks for ties
    order = np.argsort(s)
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
        i = j + 1
    n1, n2 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """One-parameter temperature scaling by LBFGS on in-domain validation NLL."""
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / logT.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp().item())


def ece(conf, correct, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    c2i = ck["class_to_idx"]
    name, fusion = ck.get("model_name", "hybrid"), ck.get("fusion", "gated_linear")
    if name == "hybrid":
        model = HybridLeafClassifier(len(c2i), img_size_tr=ck["img_size"], pretrained=False, fusion=fusion)
    elif name == "effv2s":
        model = SingleBackbone(len(c2i), "tf_efficientnetv2_s.in21k_ft_in1k", ck["img_size"], False)
    else:
        model = SingleBackbone(len(c2i), "swinv2_tiny_window8_256.ms_in1k", ck["img_size"], False)
    model.load_state_dict(ck["model_state"])
    model = model.to(device).eval()
    S = ck["img_size"]

    sets = {
        "plantvillage_val": load_split_csv(RES / "split_val.csv"),
        "plantvillage_test": load_split_csv(RES / "split_test.csv"),
        "plantdoc": [(p, c) for p, c in index_plantdoc_shared(c2i) if c in c2i],
        "plantwild": index_plantwild_shared(c2i),
    }
    raw = {}
    for k, items in sets.items():
        print(f"[info] scoring {k} ({len(items)} images)", flush=True)
        raw[k] = logits_for(model, items, c2i, S, device)

    # Temperature fitted on the in-domain validation split only.
    T = fit_temperature(*raw["plantvillage_val"])
    print(f"[info] temperature fitted on PlantVillage val: T={T:.4f}")

    out = {"tag": args.tag, "ckpt": str(args.ckpt), "temperature_from_indomain_val": T,
           "benchmarks": {}}

    for k in ("plantvillage_test", "plantdoc", "plantwild"):
        lg, y = raw[k]
        correct = (lg.argmax(-1) == y).numpy()
        rec = {"n": int(len(y)), "accuracy": float(correct.mean())}
        for temp_name, temp in (("uncalibrated", 1.0), ("temp_scaled", T)):
            sc = scores_from(lg, temp)
            block = {}
            for sname, s in sc.items():
                cov, risk, aurc = risk_coverage(s, correct)
                block[sname] = {
                    "aurc": aurc,
                    "coverage_at_risk_0.10": coverage_at_risk(cov, risk, 0.10),
                    "coverage_at_risk_0.20": coverage_at_risk(cov, risk, 0.20),
                    "coverage_at_risk_0.50": coverage_at_risk(cov, risk, 0.50),
                    "selective_acc_at_coverage_0.10": acc_at_coverage(cov, risk, 0.10),
                    "selective_acc_at_coverage_0.25": acc_at_coverage(cov, risk, 0.25),
                    "selective_acc_at_coverage_0.50": acc_at_coverage(cov, risk, 0.50),
                }
                if sname == "msp":
                    block[sname]["mean_confidence"] = float(s.mean())
                    block[sname]["ece_15bin"] = ece(s, correct)
                    np.save(RES / f"riskcov_{args.tag}_{k}_{temp_name}.npy",
                            np.stack([cov, risk]))
            rec[temp_name] = block
        out["benchmarks"][k] = rec
        u = rec["uncalibrated"]["msp"]
        print(f"  [{k}] acc={rec['accuracy']:.4f} ECE={u['ece_15bin']:.3f} "
              f"AURC={u['aurc']:.4f} cov@risk20%={u['coverage_at_risk_0.20']:.3f}", flush=True)

    # Can a single global threshold route field images to a human?
    ood = {}
    for temp_name, temp in (("uncalibrated", 1.0), ("temp_scaled", T)):
        s_in = scores_from(raw["plantvillage_test"][0], temp)
        blk = {}
        for field in ("plantdoc", "plantwild"):
            s_out = scores_from(raw[field][0], temp)
            blk[field] = {sn: auroc(s_in[sn], s_out[sn]) for sn in s_in}
        ood[temp_name] = blk
    out["ood_detection_auroc_indomain_vs_field"] = ood
    print("[ood] AUROC (in-domain vs field), uncalibrated:",
          json.dumps(ood["uncalibrated"], indent=2))

    with open(RES / f"abstention_{args.tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote abstention_{args.tag}.json")


if __name__ == "__main__":
    main()
