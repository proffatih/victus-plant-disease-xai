"""REV1 (2001013), R3 comment 3: test-time-augmentation (TTA) uncertainty for abstention.

For each reference model, every PlantVillage test / PlantDoc / PlantWild image is scored
under 8 label-preserving views of the evaluation crop (identity, horizontal flip,
vertical flip, rotations by 90 and 270 degrees, and three 87.5% zoom crops resized back:
centre, top-left, bottom-right). Prediction = argmax of the view-averaged probability.
Abstention scores: TTA probability (largest averaged probability), negative standard
deviation over views of the predicted class's probability, and view agreement (share of
views whose argmax equals the TTA prediction). Selective summaries use the TTA
prediction's correctness; OOD AUROC separates each field set from PV test.
-> results/tta_scores.json (per model) ; per-view probabilities are not stored."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import LeafDataset, build_transforms, load_split_csv, index_plantdoc_shared
from plantwild import index_plantwild_shared
from dump_features import load_model, R3RES
from posthoc_rev1 import selective_summary, auroc, calib_block

OUTD = Path(__file__).resolve().parents[1] / "results"


def views(x):
    S = x.shape[-1]; c = int(round(S * 0.875))
    out = [x, x.flip(-1), x.flip(-2), torch.rot90(x, 1, (-2, -1)), torch.rot90(x, 3, (-2, -1))]
    o = (S - c) // 2
    for (i, j) in ((o, o), (0, 0), (S - c, S - c)):
        out.append(F.interpolate(x[..., i:i + c, j:j + c], size=(S, S), mode="bilinear", align_corners=False))
    return out


@torch.no_grad()
def score(model, items, c2i, S, device):
    dl = DataLoader(LeafDataset(items, c2i, build_transforms(S, train=False, preproc="aspect")),
                    batch_size=32, shuffle=False, num_workers=6, pin_memory=True)
    P, Y = [], []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            P.append(torch.stack([model(v).float().softmax(1) for v in views(x)], 1).cpu())  # B,V,C
        Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def main():
    tags = sys.argv[1:] or [f"{a}_s{s}" for a in ("effv2s", "swinv2t", "hybrid_gated", "hybrid_tokenattn") for s in (42, 1337, 2024)]
    dst = OUTD / "tta_scores.json"
    res = json.load(open(dst)) if dst.exists() else {}
    for tag in tags:
        if tag in res:
            print("[skip]", tag); continue
        model, c2i, S = load_model(R3RES / f"best_{tag}.pt", "cuda")
        sets = {"pv_test": load_split_csv(R3RES / "split_test.csv"),
                "plantdoc": [(p, c) for p, c in index_plantdoc_shared(c2i) if c in c2i],
                "plantwild": index_plantwild_shared(c2i)}
        r, sc = {}, {}
        for k, items in sets.items():
            Pv, y = score(model, items, c2i, S, "cuda")
            pm = Pv.mean(1); pred = pm.argmax(1); corr = pred == y
            p0 = Pv[:, 0]; corr0 = p0.argmax(1) == y
            s = {"tta_prob": pm.max(1),
                 "neg_tta_std": -Pv[np.arange(len(y)), :, pred].std(1),
                 "tta_agreement": (Pv.argmax(2) == pred[:, None]).mean(1) + 1e-3 * pm.max(1)}
            sc[k] = s
            r[k] = {"acc_single_view": float(corr0.mean()), "acc_tta": float(corr.mean()),
                    "calib_tta": calib_block(pm, y),
                    "selective": {n: selective_summary(v, corr) for n, v in s.items()},
                    "selective_msp_single_view": selective_summary(p0.max(1), corr0)}
            print(f"[{tag}] {k} acc1={corr0.mean():.4f} accTTA={corr.mean():.4f} "
                  f"E-AURC msp1={r[k]['selective_msp_single_view']['e_aurc']:.3f} "
                  + " ".join(f"{n}={v['e_aurc']:.3f}" for n, v in r[k]['selective'].items()), flush=True)
        r["ood_auroc"] = {k: {n: auroc(sc["pv_test"][n], sc[k][n]) for n in sc[k]} for k in ("plantdoc", "plantwild")}
        res[tag] = r
        json.dump(res, open(dst, "w"), indent=1)
        del model; torch.cuda.empty_cache()
    print("[done]", dst)


if __name__ == "__main__":
    main()
