"""Dataset preparation and loaders for PlantVillage (train) and PlantDoc (cross-domain test).

R3 repairs (Reviewer 5):
  * E11(a) directory listings are sorted, so the seeded split is reproducible
    across filesystems.
  * E11(d) split manifests are written with the csv module, so class names that
    contain commas (``Pepper,_bell___*``) are properly quoted.
  * E11(e) no hard-coded absolute paths; roots come from DATA_ROOT / RESULTS_DIR.
  * E7 evaluation preprocessing is aspect-preserving (short side -> 272,
    centre crop 256), matching Table 1. The previous non-aspect-preserving
    square resize is retained behind ``preproc="square"`` so the effect of the
    distortion on field imagery can be quantified.
  * Minor-2 ``Tomato two spotted spider mites leaf`` and ``Potato leaf`` are
    added to the PlantDoc -> PlantVillage mapping.
"""
from __future__ import annotations
import os, csv, json, random
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

_DR = os.environ.get("DATA_ROOT", str(Path.home() / "datasets"))
PV_ROOT = Path(_DR) / "plantvillage/PV_root/Plant_leave_diseases_dataset_without_augmentation"
PD_TRAIN_ROOT = Path(_DR) / "plantdoc/PlantDoc-Dataset/train"
PD_TEST_ROOT = Path(_DR) / "plantdoc/PlantDoc-Dataset/test"

IMG_EXT = {".jpg", ".jpeg", ".png"}
PV_SKIP = {"Background_without_leaves"}

# Mapping from PlantDoc folder name -> PlantVillage folder name.
# R3: the two entries flagged in Reviewer 5 minor point 2 are now included,
# taking the mapping from 27 to 29 shared classes.
PLANTDOC_TO_PV = {
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry___healthy",
    "Corn Gray leaf spot": "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn___Common_rust",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
    "Peach leaf": "Peach___healthy",
    "Potato leaf": "Potato___healthy",                                   # R3 minor-2
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato two spotted spider mites leaf":                              # R3 minor-2
        "Tomato___Spider_mites Two-spotted_spider_mite",
}


def _listdir_sorted(d: Path) -> list[Path]:
    """Deterministic, filesystem-independent listing (Reviewer 5, E11a)."""
    return sorted((f for f in d.iterdir() if f.suffix.lower() in IMG_EXT),
                  key=lambda p: p.name)


def list_plantvillage_classes() -> list[str]:
    return sorted(d.name for d in PV_ROOT.iterdir()
                  if d.is_dir() and d.name not in PV_SKIP)


def index_plantvillage() -> list[tuple[str, str]]:
    """Return a deterministically ordered list of (image_path, class_name)."""
    items = []
    for cls in list_plantvillage_classes():
        for f in _listdir_sorted(PV_ROOT / cls):
            items.append((str(f), cls))
    return items


def stratified_split(items, val=0.1, test=0.1, seed=42):
    rng = random.Random(seed)
    by_class: dict[str, list] = {}
    for p, c in items:
        by_class.setdefault(c, []).append(p)
    train, va, te = [], [], []
    for c in sorted(by_class):                      # deterministic class order
        paths = sorted(by_class[c])                 # deterministic before shuffle
        rng.shuffle(paths)
        n = len(paths)
        n_test = max(1, int(round(n * test)))
        n_val = max(1, int(round(n * val)))
        te.extend([(p, c) for p in paths[:n_test]])
        va.extend([(p, c) for p in paths[n_test:n_test + n_val]])
        train.extend([(p, c) for p in paths[n_test + n_val:]])
    return train, va, te


class LeafDataset(Dataset):
    def __init__(self, items: Sequence[tuple[str, str]], class_to_idx: dict, transform=None):
        self.items = list(items)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, c = self.items[idx]
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.class_to_idx[c]


def build_transforms(img_size: int = 256, train: bool = True, preproc: str = "aspect"):
    """Build the image pipeline.

    preproc="aspect" (default, R3): short side resized to ``img_size + 16`` with
        the aspect ratio preserved, then a centre crop (eval) or random crop
        (train) to ``img_size``. This is what Table 1 has always described.
    preproc="square" (R2 behaviour): non-aspect-preserving resize to a square.
        Retained only so the effect of the distortion can be measured (E7).
    """
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    resize_to = img_size + 16
    if preproc == "aspect":
        pre_train = [transforms.Resize(resize_to), transforms.RandomCrop((img_size, img_size), pad_if_needed=True)]
        pre_eval = [transforms.Resize(resize_to), transforms.CenterCrop((img_size, img_size))]
    elif preproc == "square":
        pre_train = [transforms.Resize((resize_to, resize_to)), transforms.RandomCrop((img_size, img_size))]
        pre_eval = [transforms.Resize((img_size, img_size))]
    else:
        raise ValueError(f"unknown preproc {preproc!r}")

    if train:
        return transforms.Compose(pre_train + [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.2),
            transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose(pre_eval + [transforms.ToTensor(), transforms.Normalize(mean, std)])


def index_plantdoc_shared(pv_class_to_idx: dict, splits=("test", "train")) -> list[tuple[str, str]]:
    """PlantDoc images relabelled with their shared PlantVillage class name."""
    roots = {"test": PD_TEST_ROOT, "train": PD_TRAIN_ROOT}
    items = []
    for pd_cls in sorted(PLANTDOC_TO_PV):
        pv_cls = PLANTDOC_TO_PV[pd_cls]
        if pv_cls not in pv_class_to_idx:
            continue
        for sp in splits:
            d = roots[sp] / pd_cls
            if not d.is_dir():
                continue
            for f in _listdir_sorted(d):
                items.append((str(f), pv_cls))
    return items


def save_split_manifests(out_dir: Path, train, val, test, class_to_idx):
    """Write properly quoted CSV manifests (Reviewer 5, E11d)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, items in [("train", train), ("val", val), ("test", test)]:
        with open(out_dir / f"split_{name}.csv", "w", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            w.writerow(["path", "class"])
            w.writerows(items)
    with open(out_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)


def load_split_csv(path: Path) -> list[tuple[str, str]]:
    """Read a quoted manifest. Relative image paths are resolved against DATA_ROOT,
    so the released manifests work on any machine."""
    with open(path, newline="") as f:
        rows = [(r["path"], r["class"]) for r in csv.DictReader(f)]
    return [(p if os.path.isabs(p) else str(Path(_DR) / p), c) for p, c in rows]


if __name__ == "__main__":
    results = Path(os.environ.get("RESULTS_DIR",
                   Path(__file__).resolve().parents[1] / "results"))
    items = index_plantvillage()
    classes = list_plantvillage_classes()
    print(f"PlantVillage total: {len(items)} images across {len(classes)} classes")
    tr, va, te = stratified_split(items)
    print(f"train={len(tr)} val={len(va)} test={len(te)}")
    c2i = {c: i for i, c in enumerate(classes)}
    save_split_manifests(results, tr, va, te, c2i)
    pd_items = index_plantdoc_shared(c2i)
    n_shared = len({c for _, c in pd_items})
    print(f"PlantDoc shared-label images: {len(pd_items)} across {n_shared} classes")
    with open(results / "plantdoc_shared.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "class"])
        w.writerows(pd_items)
