"""fig_kshot: EfficientNetV2-S fine-tuned with k labelled field images per class.

Reads results/post_ft_calib.json (k-shot and full fine-tuning, three seeds) and
results/ft_summary.json (zero-shot accuracy on the same test partition).
Panels: test accuracy, ECE without recalibration, coverage at 20% risk (MSP).
Colours follow the Wong palette used by the other figures of the paper.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "manuscript" / "figures"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "savefig.bbox": "tight",
})
COL = {"plantdoc": "#0072B2", "plantwild": "#D55E00"}
NAME = {"plantdoc": "PlantDoc", "plantwild": "PlantWild"}

calib = json.load(open(RES / "post_ft_calib.json"))["summary"]
ft = json.load(open(RES / "ft_summary.json"))


def zs_acc(d):
    z = ft[f"{d}_effv2s"]["zero_shot_test_acc"]
    return z["mean"], z["sd"]


xs = [0, 1, 2, 3, 4]
xt = ["0", "5", "10", "25", "all"]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
for d in ("plantdoc", "plantwild"):
    tags = [f"{d}_effv2s_k5", f"{d}_effv2s_k10", f"{d}_effv2s_k25", f"{d}_effv2s"]
    acc = [calib[t]["none"]["acc"] for t in tags]
    ece = [calib[t]["none"]["ece"] for t in tags]
    cov = [calib[t]["none"]["sel_msp"]["cov_at_risk_0.20"] for t in tags]
    z = zs_acc(d)
    m = np.array([z[0]] + [a[0] for a in acc]) * 100
    s = np.array([z[1]] + [a[1] for a in acc]) * 100
    axs[0].errorbar(xs, m, yerr=s, color=COL[d], marker="o", ms=4, capsize=2, label=NAME[d])
    for ax, vals, scale in ((axs[1], ece, 1), (axs[2], cov, 100)):
        mm = np.array([v[0] for v in vals]) * scale
        ss = np.array([v[1] for v in vals]) * scale
        ax.errorbar(xs[1:], mm, yerr=ss, color=COL[d], marker="o", ms=4, capsize=2)
for ax, yl, t in zip(axs, ["Test accuracy (%)", "ECE (no recalibration)",
                            "Answered at 20% risk (%)"], "abc"):
    ax.set_xticks(xs)
    ax.set_xticklabels(xt)
    ax.set_xlim(-0.4, 4.4)
    ax.set_ylabel(yl)
    ax.set_title(f"({t})", loc="left", fontsize=9)
axs[0].set_ylim(0, 80)
axs[1].set_ylim(0, 0.25)
axs[2].set_ylim(0, 85)
axs[0].legend(frameon=False, loc="upper left")
fig.supxlabel("Labelled field images per class", fontsize=9)
fig.tight_layout()
FIG.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG / "fig_kshot.pdf")
fig.savefig(FIG / "fig_kshot.png", dpi=200)
print("[fig] fig_kshot")
