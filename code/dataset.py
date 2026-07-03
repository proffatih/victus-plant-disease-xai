"""Dataset preparation and loaders for PlantVillage (train) and PlantDoc (cross-domain test).

Also provides a class-name mapping between the two so a cross-dataset
evaluation restricted to the shared label subset can be run.
"""
from __future__ import annotations
import os, json, random
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

PV_ROOT = Path("/home/victus/datasets/plantvillage/PV_root/Plant_leave_diseases_dataset_without_augmentation")
PD_TRAIN_ROOT = Path("/home/victus/datasets/plantdoc/PlantDoc-Dataset/train")
PD_TEST_ROOT = Path("/home/victus/datasets/plantdoc/PlantDoc-Dataset/test")

# Classes to skip (not real plant disease classes)
PV_SKIP = {"Background_without_leaves"}

# Mapping from PlantDoc folder name -> PlantVillage folder name.
# Only classes with a semantically compatible label in PlantVillage are kept.
# Ambiguous or PlantDoc-only classes are omitted.
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
}


def list_plantvillage_classes() -> list[str]:
    classes = sorted([d.name for d in PV_ROOT.iterdir() if d.is_dir() and d.name not in PV_SKIP])
    return classes


def index_plantvillage() -> list[tuple[str, str]]:
    """Return list of (image_path, class_name)."""
    items = []
    for cls in list_plantvillage_classes():
        for f in (PV_ROOT / cls).iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append((str(f), cls))
    return items


def stratified_split(items, val=0.1, test=0.1, seed=42) -> tuple[list, list, list]:
    rng = random.Random(seed)
    by_class: dict[str, list] = {}
    for p, c in items:
        by_class.setdefault(c, []).append(p)
    train, va, te = [], [], []
    for c, paths in by_class.items():
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


def build_transforms(img_size: int = 224, train: bool = True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 16, img_size + 16)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.2),
            transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def index_plantdoc_shared(pv_class_to_idx: dict) -> list[tuple[str, str]]:
    """Return PlantDoc test images labeled with the shared PV class name."""
    items = []
    for pd_cls, pv_cls in PLANTDOC_TO_PV.items():
        if pv_cls not in pv_class_to_idx:
            continue
        for split_root in [PD_TEST_ROOT, PD_TRAIN_ROOT]:
            d = split_root / pd_cls
            if not d.is_dir():
                continue
            for f in d.iterdir():
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    items.append((str(f), pv_cls))
    return items


def save_split_manifests(out_dir: Path, train, val, test, class_to_idx):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, items in [("train", train), ("val", val), ("test", test)]:
        with open(out_dir / f"split_{name}.csv", "w") as f:
            f.write("path,class\n")
            for p, c in items:
                f.write(f"{p},{c}\n")
    with open(out_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)


if __name__ == "__main__":
    items = index_plantvillage()
    print(f"PlantVillage total: {len(items)} images across {len(list_plantvillage_classes())} classes")
    tr, va, te = stratified_split(items)
    print(f"train={len(tr)} val={len(va)} test={len(te)}")
    classes = list_plantvillage_classes()
    c2i = {c: i for i, c in enumerate(classes)}
    save_split_manifests(Path("/home/victus/Papers/Victus_Pardus_0011_Plant_Disease_XAI/Frontiers_Plant_Science/results"), tr, va, te, c2i)
    pd_items = index_plantdoc_shared(c2i)
    print(f"PlantDoc shared-label images: {len(pd_items)}")
    with open("/home/victus/Papers/Victus_Pardus_0011_Plant_Disease_XAI/Frontiers_Plant_Science/results/plantdoc_shared.csv", "w") as f:
        f.write("path,class\n")
        for p, c in pd_items:
            f.write(f"{p},{c}\n")
