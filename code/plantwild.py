"""R3 / E9 — PlantWild as a second, larger in-the-wild benchmark.

PlantWild (Wei et al., ACM MM 2024; doi:10.1145/3664647.3680599) contains
18,542 expert-verified in-the-wild images across 89 classes. This module maps
its classes onto the 38 PlantVillage labels the model was trained on and
exposes the shared-label subset.

The mapping is deliberately conservative: only PlantWild classes whose disease
*and* host match a PlantVillage class unambiguously are kept. Ambiguous pairings
are listed in ``AMBIGUOUS`` with the reason and are excluded from the headline
numbers; PlantWild classes describing hosts or diseases absent from
PlantVillage (banana, basil, coffee, rice, ...) are necessarily dropped.
"""
from __future__ import annotations
import os
from pathlib import Path

_DR = os.environ.get("DATA_ROOT", str(Path.home() / "datasets"))
PW_ROOT = Path(_DR) / "plantwild/extracted/plantwild"
PW_IMAGES = PW_ROOT / "images"
IMG_EXT = {".jpg", ".jpeg", ".png"}

# PlantWild folder name -> PlantVillage class name (33 unambiguous pairs).
PLANTWILD_TO_PV = {
    "apple black rot": "Apple___Black_rot",
    "apple leaf": "Apple___healthy",
    "apple rust": "Apple___Cedar_apple_rust",
    "apple scab": "Apple___Apple_scab",
    "bell pepper leaf": "Pepper,_bell___healthy",
    "bell pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "blueberry leaf": "Blueberry___healthy",
    "cherry leaf": "Cherry___healthy",
    "cherry powdery mildew": "Cherry___Powdery_mildew",
    "citrus greening disease": "Orange___Haunglongbing_(Citrus_greening)",
    "corn gray leaf spot": "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "corn leaf": "Corn___healthy",
    "corn northern leaf blight": "Corn___Northern_Leaf_Blight",
    "corn rust": "Corn___Common_rust",
    "grape black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",
    "peach leaf": "Peach___healthy",
    "potato early blight": "Potato___Early_blight",
    "potato late blight": "Potato___Late_blight",
    "potato leaf": "Potato___healthy",
    "raspberry leaf": "Raspberry___healthy",
    "soybean leaf": "Soybean___healthy",
    "squash powdery mildew": "Squash___Powdery_mildew",
    "strawberry leaf": "Strawberry___healthy",
    "strawberry leaf scorch": "Strawberry___Leaf_scorch",
    "tomato bacterial leaf spot": "Tomato___Bacterial_spot",
    "tomato early blight": "Tomato___Early_blight",
    "tomato late blight": "Tomato___Late_blight",
    "tomato leaf": "Tomato___healthy",
    "tomato leaf mold": "Tomato___Leaf_Mold",
    "tomato mosaic virus": "Tomato___Tomato_mosaic_virus",
    "tomato septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "tomato yellow leaf curl virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}

# Excluded deliberately, with the reason, so the choice is auditable.
AMBIGUOUS = {
    "grape leaf spot": ("Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                        "'grape leaf spot' is used for several grape foliar diseases; "
                        "PlantWild also carries 'grape downy mildew' separately, so the "
                        "intended pathogen is not determinable from the label alone"),
    "apple mosaic virus": (None, "no apple mosaic class exists in PlantVillage"),
    "cherry leaf spot": (None, "PlantVillage has no cherry leaf-spot class"),
    "plum leaf": (None, "PlantVillage has no plum class"),
    "maple leaf": (None, "PlantVillage has no maple class"),
}


def index_plantwild_shared(pv_class_to_idx: dict) -> list[tuple[str, str]]:
    """PlantWild images relabelled with their shared PlantVillage class name."""
    items = []
    for pw_cls in sorted(PLANTWILD_TO_PV):
        pv_cls = PLANTWILD_TO_PV[pw_cls]
        if pv_cls not in pv_class_to_idx:
            continue
        d = PW_IMAGES / pw_cls
        if not d.is_dir():
            continue
        for f in sorted((f for f in d.iterdir() if f.suffix.lower() in IMG_EXT),
                        key=lambda p: p.name):
            items.append((str(f), pv_cls))
    return items


if __name__ == "__main__":
    import json, sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset import list_plantvillage_classes
    classes = list_plantvillage_classes()
    c2i = {c: i for i, c in enumerate(classes)}
    items = index_plantwild_shared(c2i)
    shared = sorted({c for _, c in items})
    all_folders = sorted(d.name for d in PW_IMAGES.iterdir() if d.is_dir())
    total = sum(1 for d in PW_IMAGES.iterdir() if d.is_dir()
                for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
    print(f"PlantWild total          : {total} images / {len(all_folders)} classes")
    print(f"Mapped to PlantVillage   : {len(items)} images / {len(shared)} classes")
    print(f"PlantWild classes dropped: {len(all_folders) - len(PLANTWILD_TO_PV)}")
    per = {}
    for _, c in items:
        per[c] = per.get(c, 0) + 1
    print("\nper-class counts:")
    for c in sorted(per):
        print(f"  {per[c]:5d}  {c}")
    out = Path(__file__).resolve().parents[1] / "results" / "plantwild_mapping.json"
    json.dump({"mapping": PLANTWILD_TO_PV, "ambiguous_excluded":
               {k: {"candidate": v[0], "reason": v[1]} for k, v in AMBIGUOUS.items()},
               "n_images": len(items), "n_classes": len(shared),
               "plantwild_total_images": total, "plantwild_total_classes": len(all_folders),
               "per_class_counts": per}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
