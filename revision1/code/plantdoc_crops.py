"""REV1 (R1-1): why do published PlantVillage->PlantDoc numbers differ so much?

Salman et al. (2025) report 0.74 for a PlantVillage-trained model on PlantDoc,
after splitting composite images "by symptom" and removing unrelated images;
neither the evaluated subset nor the class list is given. We cannot reproduce
their subset, but we can measure how much each of the protocol choices that
such curation implies moves the number for our own models:

  A  whole images, 38-way prediction (our zero-shot protocol)
  B  whole images, prediction restricted to the classes present in the test
     set (a common implicit choice when the label space is matched first)
  C  single-leaf crops from the PlantDoc object-detection annotations (the
     closest public analogue of splitting composite images by symptom)
  D  C + restricted label space
  E  C, crops whose shorter side is >= 100 px only (removes tiny leaves)

The detection release (Singh et al. 2020, github.com/pratikkayal/
PlantDoc-Object-Detection-Dataset) uses the same folder/class names as the
classification release, so the same PlantDoc->PlantVillage mapping applies.
"""
from __future__ import annotations
import csv, json, os, sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from dataset import PLANTDOC_TO_PV, build_transforms
from dump_features import load_model, R3RES

DET = Path(os.environ.get("PD_DET_ROOT", Path.home() / "datasets" / "plantdoc_det"))
OUTD = Path(__file__).resolve().parents[1] / "results"


class Items(Dataset):
    def __init__(self, recs, c2i, tf):
        self.r, self.c2i, self.tf = recs, c2i, tf
    def __len__(self):
        return len(self.r)
    def __getitem__(self, i):
        r = self.r[i]
        im = Image.open(r["file"]).convert("RGB")
        if r["box"] is not None:
            x0, y0, x1, y1 = r["box"]
            W, H = im.size
            # annotation sizes can differ from the stored file; rescale
            sx, sy = (W / r["w"], H / r["h"]) if r["w"] > 0 and r["h"] > 0 else (1.0, 1.0)
            im = im.crop((max(0, x0 * sx), max(0, y0 * sy), min(W, x1 * sx), min(H, y1 * sy)))
        return self.tf(im), self.c2i[r["cls"]]


def records():
    whole, crops = {}, []
    for split, folder in (("test", "TEST"), ("train", "TRAIN")):
        for r in csv.DictReader(open(DET / f"{split}_labels.csv")):
            if r["class"] not in PLANTDOC_TO_PV:
                continue
            f = DET / folder / r["filename"]
            if not f.exists():
                continue
            cls = PLANTDOC_TO_PV[r["class"]]
            w, h = int(r["width"]), int(r["height"])
            box = tuple(int(float(r[k])) for k in ("xmin", "ymin", "xmax", "ymax"))
            side = min(box[2] - box[0], box[3] - box[1])
            if side < 16:
                continue
            crops.append({"file": f, "cls": cls, "box": box, "w": w, "h": h, "side": side, "split": split})
            whole.setdefault(str(f), set()).add(cls)
    # whole images: keep those whose boxes all carry one label
    wh = [{"file": Path(f), "cls": next(iter(c)), "box": None, "w": 1, "h": 1, "split": "na"}
          for f, c in sorted(whole.items()) if len(c) == 1]
    return wh, crops


@torch.no_grad()
def logits(model, recs, c2i, S, device):
    dl = DataLoader(Items(recs, c2i, build_transforms(S, train=False, preproc="aspect")),
                    batch_size=32, num_workers=6)
    L, Y = [], []
    for x, y in dl:
        L.append(model(x.to(device)).float().cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(L), np.concatenate(Y)


def acc(L, Y, restrict=None):
    if restrict is not None:
        L = L.copy(); m = np.full(L.shape[1], -np.inf); m[restrict] = 0; L = L + m
    return float((L.argmax(1) == Y).mean())


def main():
    tags = sys.argv[1:] or [f"{a}_s{s}" for a in ("effv2s", "swinv2t", "hybrid_gated", "hybrid_tokenattn")
                            for s in (42, 1337, 2024)]
    wh, cr = records()
    big = [r for r in cr if r["side"] >= 100]
    print(f"[info] whole single-label images={len(wh)} crops={len(cr)} crops>=100px={len(big)}")
    out = {"n_whole": len(wh), "n_crops": len(cr), "n_crops_ge100": len(big),
           "crop_classes": Counter(r["cls"] for r in cr), "models": {}}
    device = "cuda"
    for t in tags:
        model, c2i, S = load_model(R3RES / f"best_{t}.pt", device)
        Lw, Yw = logits(model, wh, c2i, S, device)
        Lc, Yc = logits(model, cr, c2i, S, device)
        sides = np.array([r["side"] for r in cr])
        present = sorted(set(Yc.tolist()) | set(Yw.tolist()))
        res = {"A_whole_38way": acc(Lw, Yw), "B_whole_restricted": acc(Lw, Yw, present),
               "C_crops_38way": acc(Lc, Yc), "D_crops_restricted": acc(Lc, Yc, present),
               "E_crops_ge100_38way": acc(Lc[sides >= 100], Yc[sides >= 100]),
               "F_crops_ge100_restricted": acc(Lc[sides >= 100], Yc[sides >= 100], present),
               "n_restricted_classes": len(present)}
        out["models"][t] = res
        print(t, {k: round(v, 4) for k, v in res.items()}, flush=True)
        del model; torch.cuda.empty_cache()
    keys = [k for k in next(iter(out["models"].values())) if k != "n_restricted_classes"]
    out["mean_sd_over_models"] = {k: [float(np.mean([m[k] for m in out["models"].values()])),
                                      float(np.std([m[k] for m in out["models"].values()], ddof=1)) if len(out["models"]) > 1 else 0.0]
                                  for k in keys}
    json.dump(out, open(OUTD / "plantdoc_protocols.json", "w"), indent=1)
    print(json.dumps(out["mean_sd_over_models"], indent=1))


if __name__ == "__main__":
    main()
