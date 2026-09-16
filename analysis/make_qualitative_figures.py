"""Two qualitative figures: what the three benchmarks look like, and what
Grad-CAM highlights on correct and on misclassified images.

The saliency panel is deliberately disease-stratified and includes field images
the model gets wrong, which is what a reader needs in order to judge the
explanation claims.
"""
import csv, os, sys, textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
CODE = HERE / "code" if (HERE / "code" / "dataset.py").is_file() else HERE.parent / "R3" / "code"
sys.path.insert(0, str(CODE))
from dataset import build_transforms, load_split_csv           # noqa: E402
from model import HybridLeafClassifier                          # noqa: E402
from gradcam_r3 import GradCAM                                  # noqa: E402

RES = Path(os.environ.get("RESULTS_DIR") or
           (HERE / "results" if (HERE / "results").is_dir() else HERE.parent / "R3" / "results"))
FIG = HERE / "figures"; DATA = FIG / "data"
FIG.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
MM = 1 / 25.4
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#ffffff"
BCOL = {"plantvillage": "#2a78d6", "plantdoc": "#eb6834", "plantwild": "#1baf7a"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                     "font.size": 7, "text.color": INK, "figure.facecolor": SURF,
                     "savefig.facecolor": SURF, "pdf.fonttype": 42})


def label(c):
    host, _, dis = c.partition("___")
    host = host.replace("Pepper,_bell", "Pepper (bell)").replace("_", " ")
    dis = (dis.replace("Haunglongbing_(Citrus_greening)", "Huanglongbing")
              .replace("Cercospora_leaf_spot Gray_leaf_spot", "gray leaf spot")
              .replace("Spider_mites Two-spotted_spider_mite", "spider mite")
              .replace("Tomato_Yellow_Leaf_Curl_Virus", "yellow leaf curl virus")
              .replace("Tomato_mosaic_virus", "mosaic virus").replace("_", " ").lower())
    return f"{host}: {dis}" if dis else host


def rel(p):
    """DATA_ROOT-relative, so released figure data carries no machine path."""
    dr = os.environ.get("DATA_ROOT", str(Path.home() / "datasets"))
    return os.path.relpath(p, dr) if p.startswith(dr) else p


def preds(tag, bench):
    with open(RES / f"preds_{tag}_{bench}_aspect.csv", newline="") as f:
        return list(csv.DictReader(f))


def show(ax, img, title=None, color=INK2):
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=5.8, color=color, pad=2, loc="center", wrap=True)


def load_disp(path, size=256):
    tf = build_transforms(size, train=False, preproc="aspect")
    x = tf(Image.open(path).convert("RGB"))
    return x, (x * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


# --------------------------------------------------- Fig: benchmark examples
def fig_examples(seed=3):
    rng = np.random.default_rng(seed)
    rows = [("plantvillage", "plantvillage_test", "PlantVillage (studio, training domain)"),
            ("plantdoc", "plantdoc", "PlantDoc (field)"),
            ("plantwild", "plantwild", "PlantWild (field)")]
    want = ["Tomato___Early_blight", "Tomato___healthy", "Corn___Common_rust",
            "Apple___Apple_scab", "Squash___Powdery_mildew"]
    fig, axes = plt.subplots(3, len(want), figsize=(180 * MM, 118 * MM))
    fig.subplots_adjust(left=0.10, right=0.995, top=0.94, bottom=0.01, hspace=0.12, wspace=0.04)
    out = []
    for r, (b, bfile, rlab) in enumerate(rows):
        rec = preds("hybrid_gated_s42", bfile)
        for c, cls in enumerate(want):
            pool = [x for x in rec if x["true"] == cls]
            ax = axes[r, c]
            if not pool:
                ax.axis("off"); continue
            p = pool[rng.integers(len(pool))]
            _, disp = load_disp(p["path"])
            show(ax, disp, label(cls) if r == 0 else None)
            out.append([b, cls, rel(p["path"]), p["pred"], p["top1_prob"]])
        axes[r, 0].set_ylabel(rlab, fontsize=7, color=INK, rotation=90, labelpad=6)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for s in axes[r, 0].spines.values():
            s.set_visible(False)
    fig.savefig(FIG / "fig_examples.pdf"); fig.savefig(FIG / "fig_examples.png", dpi=300)
    plt.close(fig)
    with open(DATA / "fig_examples.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["benchmark", "true_class", "path", "predicted", "confidence"]); w.writerows(out)
    print(f"  fig_examples: {len(out)} goruntu")


# --------------------------------------------------------- Fig: Grad-CAM panel
def fig_gradcam(n_each=4, seed=11):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(RES / "best_hybrid_gated_s42.pt", map_location=device, weights_only=False)
    c2i = ck["class_to_idx"]
    m = HybridLeafClassifier(len(c2i), img_size_tr=ck["img_size"], pretrained=False, fusion=ck["fusion"])
    m.load_state_dict(ck["model_state"]); m = m.to(device).eval()
    cam_engine = GradCAM(m, device)
    rng = np.random.default_rng(seed)

    pv = [r for r in preds("hybrid_gated_s42", "plantvillage_test")
          if r["true"] == r["pred"] and not r["true"].endswith("healthy")]
    pd_ = [r for r in preds("hybrid_gated_s42", "plantdoc") if r["true"] != r["pred"]]
    pick = lambda pool, k: [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]
    blocks = [("PlantVillage — correctly classified", "plantvillage", pick(pv, n_each)),
              ("PlantDoc — misclassified", "plantdoc", pick(pd_, n_each))]

    fig = plt.figure(figsize=(180 * MM, 200 * MM))
    spans = [(0.930, 0.510), (0.425, 0.005)]          # (top, bottom) of each block
    rows_out = []
    for (head, b, sel), (top, bot) in zip(blocks, spans):
        gs = fig.add_gridspec(2, n_each, left=0.045, right=0.995, top=top, bottom=bot,
                              hspace=0.06, wspace=0.04)
        fig.text(0.045, top + 0.050, head, ha="left", va="bottom", fontsize=7.5,
                 color=BCOL[b], weight="semibold")
        fig.text(0.006, top - (top - bot) * 0.27, "image", rotation=90, fontsize=6.5,
                 color=MUTED, va="center", ha="left")
        fig.text(0.006, bot + (top - bot) * 0.23, "Grad-CAM", rotation=90, fontsize=6.5,
                 color=MUTED, va="center", ha="left")
        for c, r in enumerate(sel):
            x, disp = load_disp(r["path"])
            cam, _ = cam_engine(x.unsqueeze(0).to(device),
                                torch.tensor([c2i[r["pred"]]], device=device))
            cam = cam[0].cpu().numpy()
            ok = r["true"] == r["pred"]
            cap = textwrap.fill(label(r["true"]), 24)
            if not ok:
                cap += "\n" + textwrap.fill(f"→ {label(r['pred'])}", 22) + \
                       f" ({float(r['top1_prob']):.2f})"
            show(fig.add_subplot(gs[0, c]), disp, cap, INK2 if ok else BCOL[b])
            ax = fig.add_subplot(gs[1, c])
            ax.imshow(disp); ax.imshow(cam, cmap="Blues", alpha=0.55, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            rows_out.append([head, r["true"], r["pred"], r["top1_prob"], rel(r["path"])])
    cam_engine.close()
    fig.savefig(FIG / "fig_gradcam_panel.pdf"); fig.savefig(FIG / "fig_gradcam_panel.png", dpi=300)
    plt.close(fig)
    with open(DATA / "fig_gradcam_panel.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["group", "true", "pred", "confidence", "path"]); w.writerows(rows_out)
    print(f"  fig_gradcam_panel: {len(rows_out)} goruntu ({n_each} dogru + {n_each} hatali)")


if __name__ == "__main__":
    fig_examples(); fig_gradcam()
