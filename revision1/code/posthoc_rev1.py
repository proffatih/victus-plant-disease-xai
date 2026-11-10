"""REV1 (2001013): post-hoc calibration baselines, uncertainty scores for
selective prediction, feature-space deferral -- all on the CPU from the
logits/features stored by dump_features.py.

Answers
  R2-3  calibration baselines beyond temperature scaling: vector scaling,
        Dirichlet calibration (ODIR), top-label isotonic regression, head-level
        MC dropout, and deep ensembles of the three seeds;
  R3-3  selective prediction ranked by scores other than MSP: predictive
        entropy, negative free energy, MC-dropout mutual information, ensemble
        disagreement, Mahalanobis distance and k-NN feature distance;
  R3-2  a label-free deferral rule: a feature-space (Mahalanobis) gate whose
        threshold is set on in-domain validation data only.

Every calibrator and threshold is fitted on PlantVillage data only (the
practitioner without field labels). An oracle temperature fitted on half of
each field set (scored on the other half) is reported as an upper reference.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax, logsumexp, log_softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.covariance import LedoitWolf

RUN = Path(os.environ.get("RUN_OUT", Path.home() / "RUN_OUT" / "0011_rev1"))
OUTD = Path(__file__).resolve().parents[1] / "results"
ARCHS = ["effv2s", "swinv2t", "hybrid_gated", "hybrid_tokenattn"]
SEEDS = [42, 1337, 2024]
FIELD = ["plantdoc", "plantwild"]
EVAL = ["pv_test"] + FIELD
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------- metrics
def ece(conf, correct, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def brier(p, y):
    oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1
    return float(((p - oh) ** 2).sum(1).mean())


def risk_coverage(score, correct):
    order = np.argsort(-score, kind="stable")
    c = correct[order].astype(float)
    k = np.arange(1, len(c) + 1)
    cov, risk = k / len(c), 1 - np.cumsum(c) / k
    return cov, risk


def selective_summary(score, correct):
    cov, risk = risk_coverage(score, correct)
    aurc = float(np.trapezoid(risk, cov))
    # oracle AURC: all correct ranked first
    cov_o, risk_o = risk_coverage(correct.astype(float), correct)
    eaurc = aurc - float(np.trapezoid(risk_o, cov_o))
    def cov_at(r):
        ok = np.where(risk <= r)[0]
        return float(cov[ok[-1]]) if len(ok) else 0.0
    def acc_at(c):
        i = min(int(np.searchsorted(cov, c)), len(risk) - 1)
        return float(1 - risk[i])
    return {"aurc": aurc, "e_aurc": eaurc, "cov_at_risk_0.20": cov_at(0.20),
            "cov_at_risk_0.50": cov_at(0.50), "acc_at_cov_0.10": acc_at(0.10),
            "acc_at_cov_0.25": acc_at(0.25), "acc_at_cov_0.50": acc_at(0.50)}


def auroc(pos, neg):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def calib_block(p, y):
    conf, pred = p.max(1), p.argmax(1)
    corr = (pred == y)
    return {"acc": float(corr.mean()), "mean_conf": float(conf.mean()),
            "ece": ece(conf, corr), "nll": nll(p, y), "brier": brier(p, y)}


# -------------------------------------------------------------- calibrators
def fit_temperature(z, y):
    zt, yt = torch.tensor(z, dtype=torch.float64), torch.tensor(y)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=300)
    def cl():
        opt.zero_grad(); l = torch.nn.functional.cross_entropy(zt / logT.exp(), yt); l.backward(); return l
    opt.step(cl)
    return float(logT.exp())


def fit_affine(x, y, full: bool, lam=1e-2, iters=500):
    """Vector scaling (full=False: per-class scale and bias on logits) or
    Dirichlet calibration with ODIR regularisation (full=True: matrix on
    log-probabilities, off-diagonal and intercept penalties; Kull et al. 2019)."""
    K = x.shape[1]
    xt, yt = torch.tensor(x, dtype=torch.float64), torch.tensor(y)
    if full:
        W = torch.eye(K, dtype=torch.float64).requires_grad_(True)
    else:
        W = torch.ones(K, dtype=torch.float64).requires_grad_(True)
    b = torch.zeros(K, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")
    eye = torch.eye(K, dtype=torch.float64)
    def cl():
        opt.zero_grad()
        out = xt @ W.T + b if full else xt * W + b
        l = torch.nn.functional.cross_entropy(out, yt)
        if full:
            l = l + lam * (((W * (1 - eye)) ** 2).sum() / (K * (K - 1)) + (b ** 2).mean())
        l.backward(); return l
    opt.step(cl)
    W, b = W.detach().numpy(), b.detach().numpy()
    return (lambda z: z @ W.T + b) if full else (lambda z: z * W + b)


def fit_isotonic(p, y):
    conf, corr = p.max(1), (p.argmax(1) == y).astype(float)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(conf, corr)
    def apply(q):
        c = q.max(1); c2 = np.clip(iso.predict(c), 1e-6, 1 - 1e-6)
        # top-label recalibration; spread the remaining mass proportionally
        r = q.copy(); top = q.argmax(1)
        rest = 1 - c
        scale = np.where(rest > 0, (1 - c2) / np.maximum(rest, 1e-12), 0)
        r *= scale[:, None]; r[np.arange(len(q)), top] = c2
        return r
    return apply


def mc_dropout_probs(feats, blob, T=30, p=0.15):
    """Head-level MC dropout: LayerNorm -> Dropout(p) -> Linear, sampled T times."""
    f = feats.astype(np.float32)
    mu, var = f.mean(1, keepdims=True), f.var(1, keepdims=True)
    h = (f - mu) / np.sqrt(var + 1e-5) * blob["ln_w"] + blob["ln_b"]
    W, b = blob["lin_w"], blob["lin_b"]
    rng = np.random.default_rng(1)
    P = np.zeros((len(f), W.shape[0]), np.float64); ent_each = np.zeros(len(f))
    for _ in range(T):
        m = (rng.random(h.shape) > p) / (1 - p)
        pr = softmax((h * m) @ W.T + b, axis=1)
        P += pr; ent_each += -(pr * np.log(np.clip(pr, 1e-12, 1))).sum(1)
    P /= T; ent_each /= T
    H = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)
    return P, H - ent_each  # mean probs, mutual information


# ------------------------------------------------------------ feature scores
class Maha:
    def __init__(self, F, y):
        F = F.astype(np.float64)
        self.mu = np.stack([F[y == c].mean(0) for c in range(y.max() + 1)])
        lw = LedoitWolf().fit(F - self.mu[y])
        self.L = np.linalg.cholesky(lw.precision_)   # P = L L^T
        self.muw = self.mu @ self.L
    def score(self, F):  # higher = more in-distribution
        Fw = F.astype(np.float64) @ self.L
        d = (Fw ** 2).sum(1)[:, None] - 2 * Fw @ self.muw.T + (self.muw ** 2).sum(1)[None]
        return -d.min(1)


def knn_score(Ftr, F, k=10):
    a = Ftr / np.linalg.norm(Ftr, axis=1, keepdims=True)
    b = F / np.linalg.norm(F, axis=1, keepdims=True)
    s = b.astype(np.float32) @ a.T.astype(np.float32)
    return np.sort(s, 1)[:, -k]  # k-th largest cosine similarity


# ------------------------------------------------------------------- driver
def load(tag):
    return dict(np.load(RUN / f"feats_{tag}.npz", allow_pickle=False))


def analyse_member(tag):
    B = load(tag)
    y = {k: B[f"{k}_y"] for k in ["pv_trainsub", "pv_val"] + EVAL}
    z = {k: B[f"{k}_logits"].astype(np.float64) for k in ["pv_val"] + EVAL}
    F = {k: B[f"{k}_feats"].astype(np.float32) for k in ["pv_trainsub", "pv_val"] + EVAL}
    out = {"tag": tag, "calibration": {}, "selective": {}, "ood_auroc": {}, "gate": {}}

    T = fit_temperature(z["pv_val"], y["pv_val"])
    vs = fit_affine(z["pv_val"], y["pv_val"], full=False)
    lp_val = log_softmax(z["pv_val"], 1)
    dr = fit_affine(lp_val, y["pv_val"], full=True)
    iso = fit_isotonic(softmax(z["pv_val"], 1), y["pv_val"])
    out["temperature"] = T

    mc = {k: mc_dropout_probs(F[k], B) for k in EVAL}
    maha = Maha(F["pv_trainsub"], y["pv_trainsub"])
    sm = {k: maha.score(F[k]) for k in ["pv_val"] + EVAL}
    kn = {k: knn_score(F["pv_trainsub"], F[k]) for k in ["pv_val"] + EVAL}

    for k in EVAL:
        p0 = softmax(z[k], 1)
        cal = {"uncalibrated": p0, "temperature": softmax(z[k] / T, 1),
               "vector_scaling": softmax(vs(z[k]), 1),
               "dirichlet_odir": softmax(dr(log_softmax(z[k], 1)), 1),
               "isotonic_toplabel": iso(p0), "mc_dropout_head": mc[k][0]}
        out["calibration"][k] = {n: calib_block(p, y[k]) for n, p in cal.items()}
        if k in FIELD:  # oracle: temperature fitted on a random half of the field set
            idx = RNG.permutation(len(y[k])); a, b_ = idx[: len(idx) // 2], idx[len(idx) // 2:]
            To = fit_temperature(z[k][a], y[k][a])
            out["calibration"][k]["oracle_temperature_field_half"] = {
                **calib_block(softmax(z[k][b_] / To, 1), y[k][b_]), "T": To}

        corr = (p0.argmax(1) == y[k])
        ent = -(p0 * np.log(np.clip(p0, 1e-12, 1))).sum(1)
        scores = {"msp": p0.max(1), "neg_entropy": -ent, "neg_energy": logsumexp(z[k], 1),
                  "mc_dropout_msp": mc[k][0].max(1), "neg_mc_mutual_info": -mc[k][1],
                  "mahalanobis": sm[k], "knn_cosine": kn[k]}
        out["selective"][k] = {n: selective_summary(s, corr) for n, s in scores.items()}
        out["selective"][k]["_base_acc"] = float(corr.mean())
        if k in FIELD:
            s_in = {"msp": softmax(z["pv_test"], 1).max(1), "neg_entropy": None}
            p_in = softmax(z["pv_test"], 1)
            ins = {"msp": p_in.max(1),
                   "neg_entropy": (p_in * np.log(np.clip(p_in, 1e-12, 1))).sum(1),
                   "neg_energy": logsumexp(z["pv_test"], 1),
                   "mahalanobis": sm["pv_test"], "knn_cosine": kn["pv_test"]}
            outs = {"msp": p0.max(1), "neg_entropy": -ent, "neg_energy": logsumexp(z[k], 1),
                    "mahalanobis": sm[k], "knn_cosine": kn[k]}
            out["ood_auroc"][k] = {n: auroc(ins[n], outs[n]) for n in ins}

    # Label-free gate: accept an image only if its Mahalanobis score exceeds the
    # 5th percentile of PlantVillage validation scores (95% in-domain retention).
    thr = np.percentile(sm["pv_val"], 5)
    for k in EVAL:
        acc_mask = sm[k] >= thr
        corr = (z[k].argmax(1) == y[k])
        out["gate"][k] = {"threshold_pct_val": 5, "accepted_frac": float(acc_mask.mean()),
                          "acc_accepted": float(corr[acc_mask].mean()) if acc_mask.any() else None,
                          "acc_deferred": float(corr[~acc_mask].mean()) if (~acc_mask).any() else None,
                          "errors_caught_frac": float((~acc_mask & ~corr).sum() / max(1, (~corr).sum()))}
    return out


def analyse_ensemble(arch):
    Bs = [load(f"{arch}_s{s}") for s in SEEDS]
    out = {"arch": arch, "calibration": {}, "selective": {}}
    for k in EVAL:
        y = Bs[0][f"{k}_y"]
        P = np.stack([softmax(B[f"{k}_logits"].astype(np.float64), 1) for B in Bs])
        pm = P.mean(0)
        out["calibration"][k] = calib_block(pm, y)
        corr = pm.argmax(1) == y
        Hm = -(pm * np.log(np.clip(pm, 1e-12, 1))).sum(1)
        He = -(P * np.log(np.clip(P, 1e-12, 1))).sum(2).mean(0)
        votes = np.stack([p.argmax(1) for p in P])
        agree = (votes == votes[0]).all(0).astype(float) + pm.max(1) * 1e-3
        out["selective"][k] = {"ens_msp": selective_summary(pm.max(1), corr),
                               "neg_ens_mutual_info": selective_summary(-(Hm - He), corr),
                               "seed_agreement": selective_summary(agree, corr),
                               "_base_acc": float(corr.mean())}
    return out


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    tags = [f"{a}_s{s}" for a in ARCHS for s in SEEDS]
    tags = [t for t in tags if (RUN / f"feats_{t}.npz").exists()]
    res = {"members": {}, "ensembles": {}}
    for t in tags:
        print("[member]", t, flush=True)
        res["members"][t] = analyse_member(t)
    for a in ARCHS:
        if all((RUN / f"feats_{a}_s{s}.npz").exists() for s in SEEDS):
            print("[ensemble]", a, flush=True)
            res["ensembles"][a] = analyse_ensemble(a)
    outp = OUTD / os.environ.get("POSTHOC_OUT", "posthoc_rev1.json")
    with open(outp, "w") as f:
        json.dump(res, f, indent=1)
    print("[done]", outp)


if __name__ == "__main__":
    main()
