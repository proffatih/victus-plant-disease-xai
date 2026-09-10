"""Generate Grad-CAM overlays for a small, diverse set of PlantVillage samples.

The overlay is computed on the EfficientNet-V2-S branch of the hybrid model,
which retains a 2D spatial feature map. The classifier target is the predicted
class from the full hybrid forward pass, so the CAM attributes the full
model's decision back to the CNN branch's spatial map.
"""
from __future__ import annotations
import sys, os, csv, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_transforms, LeafDataset
from model import HybridLeafClassifier

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

RESULTS = Path("/home/victus/Papers/Victus_Pardus_0011_Plant_Disease_XAI/Frontiers_Plant_Science/results")
FIGDIR = Path("/home/victus/Papers/Victus_Pardus_0011_Plant_Disease_XAI/Frontiers_Plant_Science/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)


class HybridCAMWrapper(torch.nn.Module):
    """Wrap the hybrid model so pytorch-grad-cam's forward hook attaches to the
    CNN branch's last conv block. Grad-CAM expects a model whose forward returns
    class logits — we forward through the full hybrid.
    """
    def __init__(self, hybrid: HybridLeafClassifier):
        super().__init__()
        self.hybrid = hybrid

    def forward(self, x):
        return self.hybrid(x)


def load_model(device):
    ckpt = torch.load(RESULTS / "best_hybrid.pt", map_location=device, weights_only=False)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    model = HybridLeafClassifier(num_classes=num_classes, img_size_tr=ckpt["img_size"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, class_to_idx, idx_to_class, ckpt["img_size"]


def pick_samples(class_to_idx, n_per_crop: int = 2, seed: int = 7):
    """Pick 2 samples per crop across a spread of diseases."""
    # group PV classes by crop prefix
    crops = {}
    for c in class_to_idx:
        crop = c.split("___")[0]
        crops.setdefault(crop, []).append(c)
    rng = random.Random(seed)
    picked = []  # list of (crop, disease_class, image_path)
    for crop, cls_list in sorted(crops.items()):
        cls_list = sorted(cls_list)
        # prefer 1 healthy + 1 diseased when available
        healthy = [c for c in cls_list if "healthy" in c.lower()]
        diseased = [c for c in cls_list if c not in healthy]
        chosen_cls = []
        if healthy:
            chosen_cls.append(rng.choice(healthy))
        if diseased:
            chosen_cls.append(rng.choice(diseased))
        chosen_cls = chosen_cls[:n_per_crop]
        for c in chosen_cls:
            from dataset import PV_ROOT
            imgs = [p for p in (PV_ROOT / c).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            if not imgs:
                continue
            img_path = rng.choice(imgs)
            picked.append((crop, c, img_path))
    return picked


def prep_image(path: Path, img_size: int):
    img = Image.open(path).convert("RGB")
    img_resized = img.resize((img_size, img_size))
    rgb_np = np.array(img_resized).astype(np.float32) / 255.0
    tf = build_transforms(img_size, train=False)
    x = tf(img).unsqueeze(0)  # apply to *original* PIL — Resize inside will match
    return rgb_np, x


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_to_idx, idx_to_class, img_size = load_model(device)
    model.eval()
    wrapper = HybridCAMWrapper(model)

    # target layer: last conv block of EfficientNetV2-S (before global pool)
    target_layers = [model.cnn.conv_head] if hasattr(model.cnn, "conv_head") else [model.cnn.blocks[-1]]
    cam = GradCAM(model=wrapper, target_layers=target_layers)

    samples = pick_samples(class_to_idx, n_per_crop=1, seed=7)
    # Cap to 16
    samples = samples[:16]
    print(f"[info] generating Grad-CAM for {len(samples)} samples")

    # We build one summary figure and also save individual PNGs.
    n = len(samples)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 2 * 2.2, nrows * 2.4))
    axes = np.atleast_2d(axes)

    records = []
    for i, (crop, cls, img_path) in enumerate(samples):
        rgb_np, x = prep_image(img_path, img_size)
        x = x.to(device)
        with torch.no_grad():
            logits = model(x)
            pred_idx = int(logits.argmax(-1).item())
        pred_name = idx_to_class[pred_idx]
        true_idx = class_to_idx[cls]

        # Grad-CAM w.r.t. predicted class
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        cam_map = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred_idx)])[0]
        overlay = show_cam_on_image(rgb_np, cam_map, use_rgb=True)

        r, c_ = i // ncols, i % ncols
        axes[r, c_ * 2].imshow(rgb_np); axes[r, c_ * 2].axis("off")
        short_true = cls.replace("___", "\n").replace("_", " ")
        short_pred = pred_name.replace("___", " ")
        color = "green" if pred_idx == true_idx else "red"
        axes[r, c_ * 2].set_title(short_true, fontsize=7)
        axes[r, c_ * 2 + 1].imshow(overlay); axes[r, c_ * 2 + 1].axis("off")
        mark = "OK" if pred_idx == true_idx else "MISS"
        axes[r, c_ * 2 + 1].set_title(f"CAM ({mark})", fontsize=7, color=color)
        records.append({
            "crop": crop, "true_class": cls, "pred_class": pred_name,
            "correct": pred_idx == true_idx, "image": str(img_path),
        })

    # blank leftover cells
    for j in range(len(samples), nrows * ncols):
        r, c_ = j // ncols, j % ncols
        axes[r, c_ * 2].axis("off"); axes[r, c_ * 2 + 1].axis("off")

    plt.tight_layout()
    out_pdf = FIGDIR / "fig5_gradcam_samples.pdf"
    out_png = FIGDIR / "fig5_gradcam_samples.png"
    plt.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] saved {out_pdf} and {out_png}")

    with open(RESULTS / "gradcam_records.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["crop", "true_class", "pred_class", "correct", "image"])
        w.writeheader()
        w.writerows(records)


if __name__ == "__main__":
    main()
