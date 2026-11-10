"""Revision figures for the Grad-CAM analysis (three seeds, two architectures).

fig_faithfulness: deletion and insertion curves. Solid lines are the correctly
  classified PlantVillage images of the in-domain evidence set; dashed lines are the
  PlantDoc images of the field evidence set. Line = mean over the three seeds of the
  per-seed mean curve; band = range of the three per-seed means.
fig_gradcam_panel: Grad-CAM (true class, EfficientNetV2-S seed 42) on PlantSeg field
  images with the expert lesion outline; four correct and four misclassified images,
  chosen deterministically (alphabetical order of file name within each group).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
GC = ROOT / "results" / "gradcam"
GL = ROOT / "results" / "gradcam_lesion"
FIG = ROOT / "manuscript" / "figures"
SEEDS = (42, 1337, 2024)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "savefig.bbox": "tight",
})
COL = {"effv2s": "#0072B2", "hybrid_gated": "#D55E00"}
LAB = {"effv2s": "EfficientNetV2-S", "hybrid_gated": "Gated hybrid"}


def curves(arch, dom, kind, src):
    per_seed = []
    for s in SEEDS:
        r = pd.read_csv(GC / f"gradcam_records_{arch}_s{s}_{dom}.csv")
        c = np.load(GC / f"gradcam_{kind}_curve_{arch}_s{s}_{dom}.npy")
        pv = r.path.str.contains("plantvillage").values
        m = (pv & (r.correct.values == 1)) if src == "PV" else ~pv
        per_seed.append(c[m].mean(0))
    a = np.array(per_seed)
    return a.mean(0), a.min(0), a.max(0)


fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
for ax, kind, title in zip(axs, ("deletion", "insertion"),
                           ("(a) Deletion: most salient pixels removed first",
                            "(b) Insertion: most salient pixels revealed first")):
    for arch in ("effv2s", "hybrid_gated"):
        for dom, src, ls, lab in (("indomain", "PV", "-", "PlantVillage"),
                                  ("field", "PD", "--", "PlantDoc")):
            m, lo, hi = curves(arch, dom, kind, src)
            x = np.linspace(0, 1, len(m))
            ax.plot(x, m, ls, color=COL[arch], label=f"{LAB[arch]}, {lab}")
            ax.fill_between(x, lo, hi, color=COL[arch], alpha=0.15, lw=0)
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_xlabel("Fraction of pixels")
    ax.set_ylim(0, 1)
axs[0].set_ylabel("Probability of the true class")
axs[1].legend(frameon=False, fontsize=7.5, loc="center right")
fig.tight_layout()
fig.savefig(FIG / "fig_faithfulness.pdf")
fig.savefig(FIG / "fig_faithfulness.png", dpi=200)
plt.close(fig)
print("[fig] fig_faithfulness")

# ---------------------------------------------------------------- example panel
i2c = {v: k for k, v in json.load(open(GC / "class_to_idx.json")).items()}


def short(c):
    host, dis = c.split("___")
    host = host.replace("_", " ").replace(",", "").replace("(including sour)", "").strip()
    dis = dis.replace("_", " ").strip()
    return f"{host}: {dis}"


files = sorted((GL / "examples_effv2s_s42").glob("*.npz"))
ex = [(f, np.load(f)) for f in files]
right = [e for e in ex if int(e[1]["pred"]) == int(e[1]["y"])][:4]
wrong = [e for e in ex if int(e[1]["pred"]) != int(e[1]["y"])][:4]
sel = right + wrong
fig, axs = plt.subplots(2, 4, figsize=(7.2, 4.3))
for ax, (f, d) in zip(axs.ravel(), sel):
    img = d["img"].astype(float) / 255
    cam = d["cam"]
    cam = (cam - cam.min()) / (np.ptp(cam) + 1e-12)
    ax.imshow(img)
    ax.imshow(cam, cmap="jet", alpha=0.40, vmin=0, vmax=1)
    ax.contour(d["mask"].astype(float), levels=[0.5], colors="white", linewidths=1.0)
    ok = int(d["pred"]) == int(d["y"])
    t = f"True: {short(i2c[int(d['y'])])}"
    if not ok:
        t += f"\nPred.: {short(i2c[int(d['pred'])])}"
    ax.set_title(t, fontsize=6.3, loc="left")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#2E7D32" if ok else "#C62828"); sp.set_linewidth(1.8)
fig.tight_layout(h_pad=0.6, w_pad=0.3)
fig.savefig(FIG / "fig_gradcam_panel.pdf")
fig.savefig(FIG / "fig_gradcam_panel.png", dpi=200)
print("[fig] fig_gradcam_panel", [f.stem for f, _ in sel])
