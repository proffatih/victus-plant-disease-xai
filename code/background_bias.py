"""R3 / E8(a) — PlantVillage capture-bias control on our own split.

Noyan (2022, arXiv:2206.04374) trained a classifier on eight background pixels
alone and reached 49.0% on 38 classes against a 2.6% chance baseline, and showed
that removing the background does not remove the bias because capture bias
contaminates the leaf foreground too. This script replicates that control on the
exact split used here, so the claim is measured on our data rather than cited.

Three probes, all trained on the training split and scored on the test split:
  1. eight-corner-pixel probe    - Noyan's protocol: 8 pixels x 3 channels = 24 features
  2. border-ring probe           - mean colour of a 4-px border ring (12 features)
  3. whole-image colour-statistics probe - per-channel mean/std/percentiles of the
     *foreground*, i.e. after the uniform background is masked out, which tests
     Noyan's stronger claim that the leaf pixels themselves carry capture bias.

A logistic-regression head is used throughout, so any accuracy far above chance
comes from low-level capture signal rather than from model capacity.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).parent))
from dataset import load_split_csv

RES = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parents[1] / "results"))
S = 64          # images are downscaled to 64x64 before feature extraction


def features_one(path: str):
    try:
        with Image.open(path) as im:
            a = np.asarray(im.convert("RGB").resize((S, S), Image.BILINEAR), dtype=np.float32) / 255.0
    except Exception:
        return None
    # 1) eight pixels: four corners + four edge midpoints (Noyan's protocol)
    pts = [(0, 0), (0, S - 1), (S - 1, 0), (S - 1, S - 1),
           (0, S // 2), (S - 1, S // 2), (S // 2, 0), (S // 2, S - 1)]
    f_corner = np.concatenate([a[y, x] for y, x in pts])                 # 24
    # 2) 4-px border ring statistics
    ring = np.concatenate([a[:4].reshape(-1, 3), a[-4:].reshape(-1, 3),
                           a[:, :4].reshape(-1, 3), a[:, -4:].reshape(-1, 3)])
    f_ring = np.concatenate([ring.mean(0), ring.std(0), np.median(ring, 0), ring.max(0)])   # 12
    # 3) foreground colour statistics: mask out pixels close to the border colour
    bg = ring.mean(0)
    d = np.linalg.norm(a - bg, axis=-1)
    fg = a[d > 0.15]
    if fg.shape[0] < 32:
        fg = a.reshape(-1, 3)
    f_fg = np.concatenate([fg.mean(0), fg.std(0),
                           np.percentile(fg, 10, axis=0), np.percentile(fg, 90, axis=0),
                           [fg.shape[0] / (S * S)]])                     # 13
    return np.concatenate([f_corner, f_ring, f_fg]).astype(np.float32)


def build(items, workers=6):
    X, y, keep = [], [], []
    paths = [p for p, _ in items]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, f in enumerate(ex.map(features_one, paths, chunksize=256)):
            if f is not None:
                X.append(f); keep.append(i)
            if (i + 1) % 10000 == 0:
                print(f"  featurised {i+1}/{len(paths)}", flush=True)
    y = [items[i][1] for i in keep]
    return np.stack(X), np.array(y)


def probe(Xtr, ytr, Xte, yte, cols, name, n_classes):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, n_jobs=-1))
    clf.fit(Xtr[:, cols], ytr)
    acc = float(clf.score(Xte[:, cols], yte))
    print(f"  [{name}] n_features={len(cols)} test_acc={acc:.4f} "
          f"({acc/(1/n_classes):.1f}x chance)", flush=True)
    return {"n_features": len(cols), "test_accuracy": acc,
            "chance": 1.0 / n_classes, "times_chance": acc * n_classes}


def main():
    tr = load_split_csv(RES / "split_train.csv")
    te = load_split_csv(RES / "split_test.csv")
    print(f"[info] featurising train={len(tr)} test={len(te)}", flush=True)
    Xtr, ytr = build(tr)
    Xte, yte = build(te)
    n_classes = len(set(ytr.tolist()))
    print(f"[info] features={Xtr.shape[1]} classes={n_classes}", flush=True)

    idx_corner = list(range(0, 24))
    idx_ring = list(range(24, 36))
    idx_fg = list(range(36, 49))
    out = {
        "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
        "n_classes": n_classes, "classifier": "multinomial logistic regression",
        "reference": "Noyan 2022 (arXiv:2206.04374) reports 49.0% from 8 background pixels vs 2.6% chance",
        "probes": {
            "eight_pixels_corners_and_edges": probe(Xtr, ytr, Xte, yte, idx_corner, "8-pixel", n_classes),
            "border_ring_statistics": probe(Xtr, ytr, Xte, yte, idx_ring, "border-ring", n_classes),
            "foreground_colour_statistics": probe(Xtr, ytr, Xte, yte, idx_fg, "foreground-only", n_classes),
            "all_low_level_features": probe(Xtr, ytr, Xte, yte, list(range(Xtr.shape[1])), "all", n_classes),
        },
    }
    with open(RES / "background_bias.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["probes"], indent=2))


if __name__ == "__main__":
    main()
