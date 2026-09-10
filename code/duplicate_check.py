"""R3 / E8(b) — near-duplicate analysis of the PlantVillage split.

PlantVillage contains multiple photographs of the same physical leaf, so a
random image-level split can leak near-duplicates between train and test and
inflate the in-domain accuracy. This script quantifies that with perceptual
hashing (fast, over all 54,305 images) and reports:

  * exact-duplicate groups (identical pHash) and how they straddle the splits;
  * for every test image, the minimum Hamming distance to any training image
    of the same class, and the fraction below a near-duplicate threshold;
  * accuracy on the test images that have a near-duplicate in train versus
    those that do not - the quantity that says whether leakage inflated 99.63%.
"""
from __future__ import annotations
import csv, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import imagehash
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).parent))
from dataset import load_split_csv

R3 = Path(__file__).resolve().parents[1]
RES = R3 / "results"
OLD = R3.parent / "results"
HASH_SIZE = 8            # 64-bit pHash
NEAR_T = 5               # Hamming distance <= 5 counts as a near-duplicate


def phash_one(path: str):
    try:
        with Image.open(path) as im:
            h = imagehash.phash(im.convert("RGB"), hash_size=HASH_SIZE)
        return path, str(h)
    except Exception:
        return path, None


def hashes_for(items, workers=6):
    out = {}
    paths = [p for p, _ in items]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (p, h) in enumerate(ex.map(phash_one, paths, chunksize=256)):
            if h is not None:
                out[p] = h
            if (i + 1) % 5000 == 0:
                print(f"  hashed {i+1}/{len(paths)}", flush=True)
    return out


def to_bits(hexstr: str) -> np.ndarray:
    v = int(hexstr, 16)
    return np.array([(v >> i) & 1 for i in range(64)], dtype=np.uint8)


def main():
    tr = load_split_csv(RES / "split_train.csv")
    va = load_split_csv(RES / "split_val.csv")
    te = load_split_csv(RES / "split_test.csv")
    print(f"[info] hashing {len(tr)+len(va)+len(te)} images", flush=True)

    h_tr = hashes_for(tr)
    h_va = hashes_for(va)
    h_te = hashes_for(te)
    all_h = {**h_tr, **h_va, **h_te}
    print(f"[info] hashed {len(all_h)}", flush=True)

    split_of = {}
    for p, _ in tr: split_of[p] = "train"
    for p, _ in va: split_of[p] = "val"
    for p, _ in te: split_of[p] = "test"
    class_of = dict(tr + va + te)

    # ---- exact pHash collisions -------------------------------------------
    groups = defaultdict(list)
    for p, h in all_h.items():
        groups[h].append(p)
    dup_groups = {h: ps for h, ps in groups.items() if len(ps) > 1}
    straddling = {h: ps for h, ps in dup_groups.items()
                  if len({split_of[p] for p in ps}) > 1}
    n_test_in_straddle = sum(1 for ps in straddling.values()
                             for p in ps if split_of[p] == "test")

    # ---- nearest train neighbour for every test image (within class) ------
    tr_bits_by_class, tr_paths_by_class = {}, {}
    for p, c in tr:
        if p in h_tr:
            tr_bits_by_class.setdefault(c, []).append(to_bits(h_tr[p]))
            tr_paths_by_class.setdefault(c, []).append(p)
    for c in tr_bits_by_class:
        tr_bits_by_class[c] = np.stack(tr_bits_by_class[c])

    rows, min_d = [], []
    for p, c in te:
        if p not in h_te or c not in tr_bits_by_class:
            continue
        b = to_bits(h_te[p])
        d = (tr_bits_by_class[c] ^ b).sum(axis=1)
        j = int(d.argmin())
        rows.append((p, c, int(d[j]), tr_paths_by_class[c][j]))
        min_d.append(int(d[j]))
    min_d = np.array(min_d)

    with open(RES / "duplicate_test_nn.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_path", "class", "min_hamming_to_train_same_class", "nearest_train_path"])
        w.writerows(rows)

    # ---- accuracy split by near-duplicate status --------------------------
    pred = {}
    pred_file = RES / "preds_hybrid_gated_s42_plantvillage_test_aspect.csv"
    if not pred_file.exists():          # before the R3 runs exist, fall back
        pred_file = OLD / "test_pv_predictions.csv"
    print(f"[info] leakage-conditioned accuracy from {pred_file.name}")
    with open(pred_file, newline="") as f:
        for r in csv.DictReader(f):
            pred[r["path"]] = (r["true"], r["pred"])
    near, far, near_ok, far_ok = 0, 0, 0, 0
    for p, c, d, _ in rows:
        if p not in pred:
            continue
        ok = pred[p][0] == pred[p][1]
        if d <= NEAR_T:
            near += 1; near_ok += ok
        else:
            far += 1; far_ok += ok

    out = {
        "n_images_hashed": len(all_h),
        "hash": f"pHash-{HASH_SIZE*HASH_SIZE}bit",
        "near_duplicate_threshold_hamming": NEAR_T,
        "exact_phash_collision_groups": len(dup_groups),
        "exact_collision_groups_straddling_splits": len(straddling),
        "test_images_in_straddling_exact_groups": n_test_in_straddle,
        "test_n_with_nn": int(len(min_d)),
        "min_hamming_mean": float(min_d.mean()),
        "min_hamming_median": float(np.median(min_d)),
        "frac_test_with_near_duplicate_in_train": float((min_d <= NEAR_T).mean()),
        "frac_test_hamming_0": float((min_d == 0).mean()),
        "frac_test_hamming_le_2": float((min_d <= 2).mean()),
        "accuracy_test_with_near_duplicate": (near_ok / near) if near else None,
        "n_test_with_near_duplicate": near,
        "accuracy_test_without_near_duplicate": (far_ok / far) if far else None,
        "n_test_without_near_duplicate": far,
    }
    if near and far:
        out["leakage_accuracy_gap_pp"] = 100.0 * (near_ok / near - far_ok / far)
    with open(RES / "duplicate_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
