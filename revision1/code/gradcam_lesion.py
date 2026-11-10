"""REV1 (R3 minor 1; R1-2): Grad-CAM against expert lesion masks.

PlantSeg (Wei et al. 2025, Scientific Data; CC BY-NC 4.0) provides pixel-level
lesion annotations for in-the-wild images. Its disease names follow the
PlantWild taxonomy, so the same conservative mapping onto PlantVillage is used;
21 diseases map. For every mapped image we compute Grad-CAM for the true class
and for the predicted class, apply the evaluation geometry (short side 272,
centre crop 256) to the mask as well, and measure

  energy_in_lesion   share of total CAM mass inside the lesion mask; a map
                     placed at random has an expected value equal to the
                     lesion's area fraction, which is reported alongside;
  pointing_game      whether the CAM maximum falls on a lesion pixel;
  lesion_auroc       AUROC of CAM values for separating lesion from
                     non-lesion pixels (0.5 = no localisation).

Outputs results/gradcam_lesion_<tag>.json plus per-image records, and saves
example overlays with lesion outlines for the figure.
"""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from gradcam_r3 import GradCAM
from dump_features import load_model, R3RES

PS = Path(os.environ.get("PLANTSEG_ROOT", Path.home() / "datasets" / "plantseg" / "extracted" / "plantseg"))
OUTD = Path(__file__).resolve().parents[1] / "results" / "gradcam_lesion"
EXTRA = {"bell pepper bacterial spot": "Pepper,_bell___Bacterial_spot"}


def items(c2i):
    mp = json.load(open(R3RES / "plantwild_mapping.json"))["mapping"]
    mp = {**mp, **EXTRA}
    out = []
    for r in csv.DictReader(open(PS / "Metadata.csv", encoding="utf-8-sig")):
        d = r["Disease"]
        if d not in mp or mp[d] not in c2i:
            continue
        split = {"Training": "train", "Validation": "val", "Test": "test"}.get(r["Split"], r["Split"].lower())
        img = PS / "images" / split / r["Name"]
        msk = PS / "annotations" / split / r["Label file"]
        if img.exists() and msk.exists():
            out.append((img, msk, mp[d], d))
    return out


def main():
    tag = sys.argv[1]
    OUTD.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    model, c2i, S = load_model(R3RES / f"best_{tag}.pt", device)
    for p in model.parameters():
        p.requires_grad_(True)
    cam = GradCAM(model, device)
    geo = transforms.Compose([transforms.Resize(S + 16), transforms.CenterCrop((S, S))])
    geo_m = transforms.Compose([transforms.Resize(S + 16, interpolation=transforms.InterpolationMode.NEAREST),
                                transforms.CenterCrop((S, S))])
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    its = items(c2i)
    recs = []
    ex_dir = OUTD / f"examples_{tag}"; ex_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(11)
    ex_pick = set(rng.choice(len(its), size=min(24, len(its)), replace=False).tolist())
    for i, (ip, mp_, cls, dname) in enumerate(its):
        im = geo(Image.open(ip).convert("RGB"))
        m = np.array(geo_m(Image.open(mp_)))
        m = (m > 0) if m.ndim == 2 else (m[..., 0] > 0)
        if m.sum() == 0 or m.all():
            continue
        x = norm(im).unsqueeze(0).to(device)
        y = c2i[cls]
        c_true, logits = cam(x, [y])
        pred = int(logits.argmax(-1))
        c_pred = c_true if pred == y else cam(x, [pred])[0]
        rec = {"image": ip.name, "disease": dname, "class": cls, "pred": pred, "correct": pred == y,
               "lesion_frac": float(m.mean())}
        for nm, c in (("true", c_true), ("pred", c_pred)):
            c = c[0].cpu().numpy()
            tot = c.sum() + 1e-12
            rec[f"energy_in_lesion_{nm}"] = float(c[m].sum() / tot)
            rec[f"pointing_{nm}"] = bool(m.flat[int(c.argmax())])
            rec[f"lesion_auroc_{nm}"] = float(roc_auc_score(m.ravel(), c.ravel()))
        recs.append(rec)
        if i in ex_pick:
            np.savez_compressed(ex_dir / f"{ip.stem}.npz", img=np.array(im), mask=m,
                                cam=c_true[0].cpu().numpy(), pred=pred, y=y)
    cam.close()
    with open(OUTD / f"records_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    arr = lambda k, sel=None: np.array([r[k] for r in recs if sel is None or r["correct"] == sel], float)
    summ = {"tag": tag, "n_images": len(recs), "n_diseases": len({r["disease"] for r in recs}),
            "accuracy": float(arr("correct").mean()), "mean_lesion_frac": float(arr("lesion_frac").mean())}
    for nm in ("true", "pred"):
        for k in ("energy_in_lesion", "pointing", "lesion_auroc"):
            summ[f"{k}_{nm}"] = float(arr(f"{k}_{nm}").mean())
            for sel, lab in ((True, "correct"), (False, "wrong")):
                v = arr(f"{k}_{nm}", sel)
                summ[f"{k}_{nm}_{lab}"] = float(v.mean()) if len(v) else None
    # random-map reference for the pointing game is the lesion area fraction
    summ["pointing_random_reference"] = summ["mean_lesion_frac"]
    json.dump(summ, open(OUTD / f"summary_{tag}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
