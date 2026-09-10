"""Figures for the new submission.

Every plotted value is read from R3/results or NEW_submission/numbers.json; nothing
is typed in by hand except one literature constant (Noyan 2022: 49.0%), which is
labelled as such. Each figure also writes its plotted values to figures/data/*.csv
as a table-view twin.

Colour follows entity across all figures: PlantVillage = slot 1 (blue),
PlantDoc = slot 2 (orange), PlantWild = slot 3 (aqua) of the reference palette
(validated all-pairs on the white print surface; aqua is below 3:1 contrast, so
every aqua mark is value-labelled or tabulated). Architecture figures use the
emphasis form (EfficientNetV2-S highlighted, rest grey).
"""
import csv, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, MaxNLocator

HERE = Path(__file__).resolve().parents[1]
import os
RES = Path(os.environ.get("RESULTS_DIR") or
           (HERE / "results" if (HERE / "results").is_dir() else HERE.parent / "R3" / "results"))
FIG = HERE / "figures"; DATA = FIG / "data"
FIG.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
NUM = json.loads((HERE / "numbers.json").read_text())

SURF, INK, INK2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BCOL = {"plantvillage": "#2a78d6", "plantdoc": "#eb6834", "plantwild": "#1baf7a"}
BLAB = {"plantvillage": "PlantVillage (in-domain test)", "plantdoc": "PlantDoc (field)", "plantwild": "PlantWild (field)"}
BKEY = {"plantvillage": "plantvillage_test_aspect", "plantdoc": "plantdoc_aspect", "plantwild": "plantwild_aspect"}
BFILE = {"plantvillage": "plantvillage_test", "plantdoc": "plantdoc", "plantwild": "plantwild"}
EMPH, DEEMPH = "#2a78d6", MUTED
ARCHS = [("effv2s", "EfficientNetV2-S"), ("swinv2t", "Swin V2-T"),
         ("hybrid_gated", "Hybrid (gated linear)"), ("hybrid_tokenattn", "Hybrid (token cross-attn.)")]
SEEDS = [42, 1337, 2024]
MARK = {"effv2s": "o", "swinv2t": "s", "hybrid_gated": "^", "hybrid_tokenattn": "D"}
MIN_BIN = 30          # reliability bins with fewer predictions are drawn faded
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "legend.frameon": False, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ------------------------------------------------------------------ helpers
def save(fig, name, rows, header):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    with open(DATA / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"  wrote {name}.pdf/.png + data/{name}.csv ({len(rows)} rows)")


def preds(tag, bench):
    t, p, c = [], [], []
    with open(RES / f"preds_{tag}_{BFILE[bench]}_aspect.csv", newline="") as f:
        for r in csv.DictReader(f):
            t.append(r["true"]); p.append(r["pred"]); c.append(float(r["top1_prob"]))
    t, p = np.array(t), np.array(p)
    return np.array(c), t == p, p, t


def px(ax):
    """Data units per pixel in x and y (layout must already be final)."""
    ax.figure.canvas.draw()
    bb = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return abs(x1 - x0) / bb.width, abs(y1 - y0) / bb.height


_M, _L, _Q, _C = MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CLOSEPOLY


def vbar(ax, xc, h, w, color, rpx=3, z=2, alpha=1.0):
    """Column as a single path: rounded data-end, square baseline (no seams)."""
    if h <= 0:
        return
    ux, uy = px(ax)
    rx, ry = min(rpx * ux, w / 2), min(rpx * uy, h)
    x0, x1, y1 = xc - w / 2, xc + w / 2, h
    v = [(x0, 0), (x1, 0), (x1, y1 - ry), (x1, y1), (x1 - rx, y1), (x0 + rx, y1), (x0, y1), (x0, y1 - ry), (x0, 0)]
    ax.add_patch(PathPatch(MPath(v, [_M, _L, _L, _Q, _Q, _L, _Q, _Q, _C]),
                           fc=color, ec="none", lw=0, zorder=z, alpha=alpha))


def hbar(ax, yc, val, hh, color, rpx=3, z=2):
    """Horizontal bar as a single path: rounded data-end, square baseline."""
    if val <= 0:
        return
    ux, uy = px(ax)
    rx, ry = min(rpx * ux, val), min(rpx * uy, hh / 2)
    y0, y1 = yc - hh / 2, yc + hh / 2
    v = [(0, y0), (val - rx, y0), (val, y0), (val, y0 + ry), (val, y1 - ry), (val, y1), (val - rx, y1), (0, y1), (0, y0)]
    ax.add_patch(PathPatch(MPath(v, [_M, _L, _Q, _Q, _L, _Q, _Q, _L, _C]),
                           fc=color, ec="none", lw=0, zorder=z))


def ece_bins(conf, correct, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    out, e = [], 0.0
    for i in range(n_bins):
        m = idx == i
        k = int(m.sum())
        acc = float(correct[m].mean()) if k else float("nan")
        cf = float(conf[m].mean()) if k else float("nan")
        if k:
            e += k / len(conf) * abs(acc - cf)
        out.append((edges[i], edges[i + 1], k, k / len(conf), acc, cf))
    return out, e


PROPER = ("Cercospora", "Huanglongbing", "Isariopsis")


def label_class(c):
    host, _, dis = c.partition("___")
    host = host.replace("Pepper,_bell", "Pepper (bell)").replace("_", " ")
    for a, b in (("Haunglongbing_(Citrus_greening)", "Huanglongbing (citrus greening)"),
                 ("Cercospora_leaf_spot Gray_leaf_spot", "Cercospora / gray leaf spot"),
                 ("Spider_mites Two-spotted_spider_mite", "two-spotted spider mite"),
                 ("Tomato_Yellow_Leaf_Curl_Virus", "yellow leaf curl virus"),
                 ("Tomato_mosaic_virus", "mosaic virus"), ("Esca_(Black_Measles)", "esca (black measles)"),
                 ("Leaf_blight_(Isariopsis_Leaf_Spot)", "leaf blight (Isariopsis)")):
        dis = dis.replace(a, b)
    dis = dis.replace("_", " ").lower()
    for w in PROPER:
        dis = dis.replace(w.lower(), w)
    return f"{host}: {dis}"


def tags():
    return [(a, s, f"{a}_s{s}") for a, _ in ARCHS for s in SEEDS]


# ------------------------------------------------ Fig: overconfidence scatter
def fig_overconfidence():
    fig, ax = plt.subplots(figsize=(85 * MM, 80 * MM))
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.13, top=0.96)
    ax.set_xlim(0, 102); ax.set_ylim(0, 102)
    ax.set_xticks(range(0, 101, 20)); ax.set_yticks(range(0, 101, 20))
    ax.grid(True); ax.set_axisbelow(True)
    ax.plot([0, 100], [0, 100], color=MUTED, lw=0.6, zorder=1)
    rows = []
    for b in BCOL:
        for a, s, t in tags():
            r = NUM["models"][t][BKEY[b]]
            x, y = 100 * r["mean_confidence"], 100 * r["accuracy"]
            ax.scatter(x, y, s=24, marker=MARK[a], c=BCOL[b], edgecolors=SURF, linewidths=0.8, zorder=3)
            rows.append([t, a, s, b, f"{x:.3f}", f"{y:.3f}", f"{r['ece_15bin']:.4f}"])
    ax.set_xlabel("Mean confidence (max softmax, %)")
    ax.set_ylabel("Top-1 accuracy (%)")
    fig.canvas.draw()
    p0, p1 = ax.transData.transform((0, 0)), ax.transData.transform((100, 100))
    ang = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.text(56, 58, "perfect calibration", rotation=ang, rotation_mode="anchor",
            color=MUTED, fontsize=6, ha="center", va="bottom")
    am = NUM["all_models"]
    pv, pd_, pw = am["plantvillage_test_aspect"], am["plantdoc_aspect"], am["plantwild_aspect"]
    ax.annotate(f"In domain: {100*pv['mean_confidence']['mean']:.0f}% confident,\n"
                f"{100*pv['accuracy']['mean']:.1f}% correct",
                xy=(100 * pv["mean_confidence"]["mean"], 100 * pv["accuracy"]["mean"]),
                xytext=(99, 74), color=INK2, fontsize=6, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=2, shrinkB=5))
    fx = 100 * (pd_["mean_confidence"]["mean"] + pw["mean_confidence"]["mean"]) / 2
    fy = 100 * (pd_["accuracy"]["mean"] + pw["accuracy"]["mean"]) / 2
    ax.annotate(f"Field: {fx:.0f}% confident,\n"
                f"{100*pw['accuracy']['mean']:.0f}–{100*pd_['accuracy']['mean']:.0f}% correct",
                xy=(fx, fy), xytext=(28, 36), color=INK2, fontsize=6, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=2, shrinkB=6))
    h1 = [Line2D([], [], marker="o", ls="", mfc=BCOL[b], mec=SURF, ms=5, label=BLAB[b]) for b in BCOL]
    h2 = [Line2D([], [], marker=MARK[a], ls="", mfc=MUTED, mec=SURF, ms=4.5, label=l) for a, l in ARCHS]
    leg1 = ax.legend(handles=h1, loc="upper left", bbox_to_anchor=(0.0, 1.0), handletextpad=0.3, labelcolor=INK2)
    ax.add_artist(leg1)
    ax.legend(handles=h2, loc="upper left", bbox_to_anchor=(0.0, 0.80), handletextpad=0.3, labelcolor=INK2)
    save(fig, "fig_overconfidence", rows,
         ["tag", "architecture", "seed", "benchmark", "mean_confidence_pct", "accuracy_pct", "ece"])


# ------------------------------------------------ Fig: reliability diagrams
def fig_reliability():
    fig = plt.figure(figsize=(180 * MM, 74 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.10, wspace=0.24,
                          left=0.065, right=0.99, bottom=0.14, top=0.91)
    rows = []
    for j, b in enumerate(BCOL):
        conf, corr = [], []
        for _, _, t in tags():
            c, k, _, _ = preds(t, b); conf.append(c); corr.append(k)
        conf, corr = np.concatenate(conf), np.concatenate(corr)
        bins, e = ece_bins(conf, corr)
        top = fig.add_subplot(gs[0, j]); bot = fig.add_subplot(gs[1, j], sharex=top)
        for ax in (top, bot):
            ax.set_xlim(0, 1); ax.grid(True, axis="y"); ax.set_axisbelow(True)
        top.set_ylim(0, 1)
        smax = max(x[3] for x in bins)
        bot.set_ylim(0, smax * 1.18)
        top.plot([0, 1], [0, 1], color=MUTED, lw=0.6, zorder=1)
        w = (1 / 15) * 0.72
        for lo, hi, k, sh, acc, cf in bins:
            if k:
                vbar(top, (lo + hi) / 2, acc, w, BCOL[b], alpha=1.0 if k >= MIN_BIN else 0.35)
                vbar(bot, (lo + hi) / 2, sh, w, MUTED, rpx=2)
            rows.append([b, f"{lo:.4f}", f"{hi:.4f}", k, f"{sh:.5f}",
                         "" if k == 0 else f"{acc:.5f}", "" if k == 0 else f"{cf:.5f}"])
        top.set_title(BLAB[b], loc="left", pad=4)
        top.text(0.03, 0.96, f"ECE {e:.3f}\nn = {len(conf):,}", transform=top.transAxes,
                 ha="left", va="top", fontsize=6.5, color=INK2)
        top.tick_params(labelbottom=False)
        bot.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        bot.yaxis.set_major_locator(MaxNLocator(2))
        bot.set_xlabel("Confidence (max softmax)")
        if j == 0:
            top.set_ylabel("Accuracy"); bot.set_ylabel("Share")
    save(fig, "fig_reliability", rows, ["benchmark", "bin_lo", "bin_hi", "n", "share", "accuracy", "mean_confidence"])


# ------------------------------------------------ Fig: capture-bias probes
def fig_capture_bias():
    bb = json.loads((RES / "background_bias.json").read_text())["probes"]
    order = [("eight_pixels_corners_and_edges", "Eight pixels\n(corners, edge midpoints)"),
             ("border_ring_statistics", "Border-ring colour\nstatistics"),
             ("foreground_colour_statistics", "Foreground colour statistics\n(background masked out)"),
             ("all_low_level_features", "All 49 low-level\ncolour features")]
    chance = 100 * bb["eight_pixels_corners_and_edges"]["chance"]
    eff = 100 * NUM["by_arch"]["effv2s"]["plantvillage_test_aspect"]["accuracy"]["mean"]
    NOYAN = 49.0  # literature constant: Noyan (2022), eight background pixels, 38 classes
    fig, ax = plt.subplots(figsize=(120 * MM, 58 * MM))
    fig.subplots_adjust(left=0.30, right=0.97, bottom=0.18, top=0.84)
    ax.set_xlim(0, 105); ax.set_ylim(len(order) - 0.4, -0.6)
    ax.grid(True, axis="x"); ax.set_axisbelow(True)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([l for _, l in order])
    ax.tick_params(axis="y", length=0)
    rows = []
    for i, (k, lab) in enumerate(order):
        v = 100 * bb[k]["test_accuracy"]
        hbar(ax, i, v, 0.46, EMPH)
        ax.text(v + 1.2, i, f"{v:.1f}%", va="center", ha="left", fontsize=6.5, color=INK2)
        rows.append([lab.replace("\n", " "), bb[k]["n_features"], f"{v:.3f}", f"{bb[k]['times_chance']:.2f}"])
    for x, txt, ha in ((chance, f"chance\n{chance:.1f}%", "left"),
                       (NOYAN, f"Noyan (2022)\n{NOYAN:.1f}%", "center"),
                       (eff, f"EfficientNetV2-S\n{eff:.1f}%", "right")):
        ax.axvline(x, color=MUTED, lw=0.6, zorder=1)
        ax.text(x, -0.75, txt, ha=ha, va="bottom", fontsize=6, color=INK2)
    ax.set_xlabel("PlantVillage test accuracy of a linear probe (%)")
    save(fig, "fig_capture_bias", rows, ["probe", "n_features", "test_accuracy_pct", "times_chance"])


# ------------------------------------------------ Fig: architecture null + cost
def fig_architecture():
    eff = json.loads((RES / "efficiency_measured.json").read_text())["models"]
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 60 * MM), sharey=True)
    fig.subplots_adjust(left=0.175, right=0.985, bottom=0.2, top=0.87, wspace=0.16)
    rows = []
    for ax in axes:
        ax.set_ylim(len(ARCHS) - 0.45, -0.55)
        ax.tick_params(axis="y", length=0)
        ax.grid(True, axis="x"); ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(ARCHS))); axes[0].set_yticklabels([l for _, l in ARCHS])
    for ax, bkey, title in ((axes[0], "plantvillage_test_aspect", "a  PlantVillage test"),
                            (axes[1], "plantwild_aspect", "b  PlantWild, zero-shot")):
        vals = []
        for i, (a, _) in enumerate(ARCHS):
            col = EMPH if a == "effv2s" else DEEMPH
            xs = [100 * NUM["models"][f"{a}_s{s}"][bkey]["accuracy"] for s in SEEDS]
            vals += xs
            ax.scatter(xs, [i - 0.13, i, i + 0.13], s=22, c=col, edgecolors=SURF, linewidths=0.8, zorder=3)
            m = float(np.mean(xs))
            ax.plot([m, m], [i - 0.3, i + 0.3], color=INK, lw=1.2, zorder=4)
            for s, x in zip(SEEDS, xs):
                rows.append([bkey, a, s, f"{x:.4f}"])
        lo, hi = min(vals), max(vals); pad = (hi - lo) * 0.25
        ax.set_xlim(lo - pad, hi + pad)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_title(title, loc="left", pad=4); ax.set_xlabel("Top-1 accuracy (%)")
    ax = axes[2]
    keys = [a for a, _ in ARCHS]
    lat = [eff[k]["timing_batch1"]["per_image_ms_mean"] for k in keys]
    ax.set_xlim(0, max(lat) * 1.75)
    for i, (k, v) in enumerate(zip(keys, lat)):
        hbar(ax, i, v, 0.46, EMPH if k == "effv2s" else DEEMPH)
        ax.text(v + max(lat) * 0.04, i, f"{v:.1f} ms · {eff[k]['params_total_M']:.1f} M",
                ha="left", va="center", fontsize=6.3, color=INK2)
        rows.append(["latency_batch1_ms", k, "", f"{v:.4f}"])
        rows.append(["params_M", k, "", f"{eff[k]['params_total_M']:.4f}"])
    ax.set_title("c  Latency, batch 1", loc="left", pad=4); ax.set_xlabel("ms per image (FP32)")
    save(fig, "fig_architecture", rows, ["quantity", "architecture", "seed", "value"])


# ------------------------------------------------ Fig: prediction collapse
def fig_collapse():
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 72 * MM))
    fig.subplots_adjust(left=0.235, right=0.975, bottom=0.14, top=0.86, wspace=1.2)
    rows = []
    for ax, b in zip(axes, ("plantdoc", "plantwild")):
        per_model, n_true = [], None
        for _, _, t in tags():
            _, _, p, tr = preds(t, b)
            n_true = len(set(tr.tolist()))
            u, c = np.unique(p, return_counts=True)
            per_model.append(dict(zip(u.tolist(), (c / len(p)).tolist())))
        allc = sorted(set().union(*per_model))
        mean = {c: float(np.mean([m.get(c, 0.0) for m in per_model])) for c in allc}
        sd = {c: float(np.std([m.get(c, 0.0) for m in per_model], ddof=1)) for c in allc}
        top = sorted(allc, key=lambda c: -mean[c])[:10]
        xmax = 100 * mean[top[0]] * 1.25
        ax.set_xlim(0, xmax); ax.set_ylim(len(top) - 0.4, -0.6)
        ax.grid(True, axis="x"); ax.set_axisbelow(True)
        ax.set_yticks(range(len(top))); ax.set_yticklabels([label_class(c) for c in top])
        ax.tick_params(axis="y", length=0)
        uni = 100 / n_true
        ax.axvline(uni, color=MUTED, lw=0.6, zorder=1)
        ax.text(uni, -0.75, f"uniform over {n_true} true classes", ha="left", va="bottom", fontsize=6, color=INK2)
        for i, c in enumerate(top):
            v = 100 * mean[c]
            hbar(ax, i, v, 0.56, BCOL[b])
            ax.text(v + xmax * 0.015, i, f"{v:.1f}%", va="center", ha="left", fontsize=6.3, color=INK2)
        for c in allc:
            rows.append([b, c, f"{100*mean[c]:.4f}", f"{100*sd[c]:.4f}"])
        ax.set_title(BLAB[b], loc="left", pad=14)
        ax.set_xlabel("Share of predictions (%)")
    save(fig, "fig_collapse", rows, ["benchmark", "predicted_class", "mean_share_pct", "sd_share_pct"])


# ------------------------------------------------ Fig: Grad-CAM faithfulness
def fig_faithfulness():
    tag = "hybrid_gated_s42_field"
    recs = list(csv.DictReader(open(RES / f"gradcam_records_{tag}.csv", newline="")))
    D = np.load(RES / f"gradcam_deletion_curve_{tag}.npy")
    I = np.load(RES / f"gradcam_insertion_curve_{tag}.npy")
    assert len(recs) == len(D) == len(I), "records and curves are misaligned"
    is_pd = np.array(["plantdoc/" in r["path"].lower() for r in recs])  # absolute or DATA_ROOT-relative
    groups = [("plantvillage", ~is_pd, "PlantVillage test images"), ("plantdoc", is_pd, "PlantDoc images")]
    x = np.linspace(0, 1, D.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 62 * MM), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.84, bottom=0.17, top=0.88, wspace=0.12)
    rows = []
    for ax, C, lab, key in ((axes[0], D, "a  Deletion: most-salient pixels removed first", "deletion"),
                            (axes[1], I, "b  Insertion: most-salient pixels revealed first", "insertion")):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(True, axis="y"); ax.set_axisbelow(True)
        for b, m, name in groups:
            cur = C[m]; mu = cur.mean(0); se = cur.std(0, ddof=1) / np.sqrt(len(cur))
            ax.fill_between(x, mu - se, mu + se, color=BCOL[b], alpha=0.10, lw=0)
            ax.plot(x, mu, color=BCOL[b], lw=1.5)
            if key == "insertion":
                ax.text(1.02, mu[-1], f"{name}\n(n = {int(m.sum())})", transform=ax.get_yaxis_transform(),
                        ha="left", va="center", fontsize=6.3, color=INK2, clip_on=False)
            auc = float(np.mean([float(r[f"{key}_auc"]) for r, k in zip(recs, m) if k]))
            rows.append([key, name, int(m.sum()), f"{auc:.4f}"] + [f"{v:.5f}" for v in mu])
        ax.set_title(lab, loc="left", pad=4)
        ax.set_xlabel("Fraction of pixels " + ("removed" if key == "deletion" else "revealed"))
    axes[0].set_ylabel("Target-class probability")
    h = [Line2D([], [], color=BCOL[b], lw=1.5, label=n) for b, _, n in groups]
    axes[0].legend(handles=h, loc="center left", bbox_to_anchor=(0.02, 0.55), labelcolor=INK2)
    save(fig, "fig_faithfulness", rows,
         ["curve", "group", "n", "mean_auc"] + [f"p{int(round(100*v))}" for v in x])


# ------------------------------------------------ Fig: risk-coverage
def fig_riskcoverage():
    """Selective risk against coverage for all 12 models, ranked by max softmax.
    Temperature scaling keeps each prediction's class but can reorder predictions
    by confidence; on these data it changes AURC by at most 0.005, so only the
    uncalibrated curves are drawn."""
    grid = np.linspace(0.005, 1.0, 200)
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 62 * MM), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.18, top=0.88, wspace=0.12)
    rows = []
    for ax, b in zip(axes, ("plantvillage", "plantdoc", "plantwild")):
        C = []
        for _, _, t in tags():
            cov, risk = np.load(RES / f"riskcov_{t}_{BFILE[b]}_uncalibrated.npy")
            C.append(np.interp(grid, cov, risk))
        C = 100 * np.array(C); mu = C.mean(0)
        for c in C:
            ax.plot(grid, c, color=BCOL[b], lw=0.6, alpha=0.3, zorder=2)
        ax.plot(grid, mu, color=BCOL[b], lw=1.5, zorder=3)
        err = 100 * (1 - np.mean([NUM["models"][t][BKEY[b]]["accuracy"] for _, _, t in tags()]))
        ax.set_xlim(0, 1); ax.set_ylim(0, 100); ax.grid(True, axis="y"); ax.set_axisbelow(True)
        ax.axhline(20, color=INK2, lw=0.6, zorder=1)
        ax.text(0.99, 21.5, "20% risk budget", ha="right", va="bottom", fontsize=6, color=INK2)
        if err > 5:
            ax.axhline(err, color=MUTED, lw=0.6, zorder=1)
            ax.text(0.99, err + 1.5, "no abstention", ha="right", va="bottom", fontsize=6, color=MUTED)
        ax.set_title(BLAB[b], loc="left", pad=4)
        ax.set_xlabel("Coverage (fraction of predictions kept)")
        rows += [[b, f"{g:.4f}", f"{m:.4f}", f"{c.min():.4f}", f"{c.max():.4f}"]
                 for g, m, c in zip(grid, mu, C.T)]
    axes[0].set_ylabel("Error rate among kept predictions (%)")
    save(fig, "fig_riskcoverage", rows, ["benchmark", "coverage", "mean_risk_pct", "min_risk_pct", "max_risk_pct"])


if __name__ == "__main__":
    import sys
    ALL = (fig_overconfidence, fig_reliability, fig_capture_bias, fig_architecture, fig_collapse,
           fig_faithfulness, fig_riskcoverage)
    wanted = set(sys.argv[1:])
    for f in ALL:
        if not wanted or f.__name__ in wanted:
            print(f.__name__); f()
