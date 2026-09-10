"""LaTeX tables for the new submission, generated from numbers.json and R3/results.

No value in these tables is typed by hand; re-run after any result changes.
"""
import json, glob
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
import os
RES = Path(os.environ.get("RESULTS_DIR") or
           (HERE / "results" if (HERE / "results").is_dir() else HERE.parent / "R3" / "results"))
N = json.loads((HERE / "numbers.json").read_text())
OUT = HERE / "sections"; OUT.mkdir(exist_ok=True)
ARCH = [("effv2s", "EfficientNetV2-S"), ("swinv2t", "Swin V2-T"),
        ("hybrid_gated", "Hybrid, gated linear fusion"), ("hybrid_tokenattn", "Hybrid, token cross-attention")]
SHORT = {"effv2s": "EfficientNetV2-S", "swinv2t": "Swin V2-T",
         "hybrid_gated": "Hybrid (gated)", "hybrid_tokenattn": "Hybrid (cross-attn.)"}
BENCH = [("plantvillage_test", "PlantVillage test"), ("plantdoc", "PlantDoc"), ("plantwild", "PlantWild")]


def pm(d, scale=100, k=2):
    if d is None:
        return "--"
    return f"${d['mean']*scale:.{k}f} \\pm {d['sd']*scale:.{k}f}$"


def write(name, s):
    (OUT / name).write_text(s)
    print("  wrote", name)


def tab_architecture():
    eff = N["efficiency"]["models"]
    rows = []
    for a, lab in ARCH:
        m, v = eff[a], N["by_arch"][a]
        b1 = m["timing_batch1"]
        rows.append(
            f"{lab} & {m['params_total_M']:.2f} & {m['macs_G']:.2f} & "
            f"${b1['per_image_ms_mean']:.2f} \\pm {b1['per_image_ms_sd']:.2f}$ & "
            f"{m['timing_batch32']['throughput_img_per_s']:.0f} & "
            f"{pm(v['plantvillage_test_aspect']['accuracy'], k=3)} & "
            f"{pm(v['plantdoc_aspect']['accuracy'])} & {pm(v['plantwild_aspect']['accuracy'])} \\\\")
    write("tab_architecture.tex", r"""\begin{table}[!ht]
\centering
\caption{Cost and accuracy of the four architectures. Accuracy is the mean $\pm$
standard deviation over three seeds; PlantDoc and PlantWild are evaluated zero-shot.
Parameters and multiply--accumulate operations (MACs) are measured at
$256\times256$; latency (mean $\pm$ SD over 120 timed calls) and throughput are
model-only and FP32, on an RTX~3050~Ti Laptop GPU.}
\label{tab:arch}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lrrrrrrr}
\toprule
 & Params & MACs & Latency, batch 1 & Throughput, batch 32 & \multicolumn{3}{c}{Top-1 accuracy (\%)} \\
\cmidrule(lr){6-8}
Architecture & (M) & (G) & (ms) & (images/s) & PlantVillage & PlantDoc & PlantWild \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def tab_field():
    am = N["all_models"]
    t0 = next(iter(N["models"]))

    def cells(key, scale=100, k=2, square=False):
        out = []
        for b, _ in BENCH:
            bk = f"{b}_{'square' if square else 'aspect'}"
            d = am[bk][key] if bk in am else None
            out.append(pm(d, scale, k))
        return " & ".join(out)

    ns = " & ".join(f"{N['models'][t0][f'{b}_aspect']['n']:,}".replace(",", "\\,") for b, _ in BENCH)
    body = [
        f"Images & {ns} \\\\",
        f"Top-1 accuracy (\\%) & {cells('accuracy')} \\\\",
        f"Macro-F\\textsubscript{{1}} (\\%) & {cells('macro_f1')} \\\\",
        f"Balanced accuracy (\\%) & {cells('balanced_accuracy')} \\\\",
        f"Matthews correlation coefficient & {cells('mcc', 1, 3)} \\\\",
        f"Crop-species accuracy (\\%) & {cells('crop_species_accuracy')} \\\\",
        f"Largest share of predictions on one class (\\%) & {cells('max_single_class_prediction_share')} \\\\",
        "\\midrule",
        f"Mean confidence (\\%) & {cells('mean_confidence')} \\\\",
        f"\\quad correct predictions (\\%) & {cells('mean_confidence_correct')} \\\\",
        f"\\quad incorrect predictions (\\%) & {cells('mean_confidence_incorrect')} \\\\",
        f"Expected calibration error & {cells('ece_15bin', 1, 3)} \\\\",
        f"Maximum calibration error & {cells('mce_15bin', 1, 3)} \\\\",
        "\\midrule",
        "\\multicolumn{4}{l}{\\emph{With a non-aspect-preserving square resize}} \\\\",
        f"Top-1 accuracy (\\%) & {cells('accuracy', square=True)} \\\\",
        f"Expected calibration error & {cells('ece_15bin', 1, 3, square=True)} \\\\",
    ]
    write("tab_field.tex", r"""\begin{table}[!ht]
\centering
\caption{In-domain and zero-shot field performance, mean $\pm$ standard deviation over
all 12 models (four architectures, three seeds). Confidence is the maximum softmax
probability; calibration errors use 15 equal-width bins. The square-resize rows
repeat the field evaluation with a non-aspect-preserving square resize.}
\label{tab:field}
\footnotesize
\begin{tabular}{lccc}
\toprule
 & PlantVillage test & PlantDoc & PlantWild \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def tab_pairwise():
    rows = []
    for key, v in N["stats"]["pairwise"].items():
        a, b = key.split("_vs_")
        msd = ("--\\textsuperscript{a}" if v["min_significant_difference_pp"] is None
               else f"{v['min_significant_difference_pp']:.3f}")
        rows.append(f"{SHORT[a]} vs {SHORT[b]} & ${v['acc_diff_pp']:+.3f}$ & {v['n_discordant']} & "
                    f"{v['exact_mcnemar_p']:.3f} & {v['holm_adjusted_p']:.3f} & {msd} \\\\")
    write("tab_pairwise.tex", r"""\begin{table}[!ht]
\centering
\caption{Pairwise comparison of architectures on the PlantVillage test split
(seed-42 models, $n=""" + f"{N['stats']['n_test']:,}".replace(",", "\\,") + r"""$). The difference is the accuracy of
the first model minus the second. The minimum significant difference is the
smallest accuracy gap that the observed number of discordant images could have
declared significant at $\alpha=0.05$. The analysis is exploratory.}
\label{tab:pairwise}
\footnotesize
\begin{tabular}{lrrrrr}
\toprule
 & Difference & Discordant & Exact McNemar & Holm-adjusted & Min.\ significant \\
Comparison & (pp) & images & $p$ & $p$ & difference (pp) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}\\[2pt]
\raggedright\footnotesize\textsuperscript{a}\,No division of four discordant images reaches $p<0.05$.
\end{table}
""")


def tab_finetune():
    rows = []
    for key, lab in (("finetune_plantdoc", "PlantDoc"), ("finetune_plantwild", "PlantWild")):
        j = N[key]
        h = j["hyperparams"]
        rows.append(
            f"{lab} & {j['n_fit']:,}/{j['n_val']:,}/{j['n_test']:,} & {j['n_classes']} & "
            f"{100*j['zero_shot_test_acc']:.2f} & {100*j['finetuned_test_acc']:.2f} & "
            f"{100*j['finetuned_test_macro_f1']:.2f} & {j['selected_epoch']}/{h['epochs']} & "
            f"{100*j['r2_protocol_running_max_test_acc']:.2f} & ${j['optimism_pp']:+.2f}$ \\\\".replace(",", "\\,"))
    write("tab_finetune.tex", r"""\begin{table}[!ht]
\centering
\caption{Target-domain fine-tuning of the seed-42 gated-fusion hybrid. The epoch
is selected on the validation split and the test partition is scored once. The
last two columns show what selecting on the test partition itself would have
reported, and by how much that would have overstated the result; they were never
used for selection.}
\label{tab:finetune}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lrrrrrrrr}
\toprule
 & Fit/val/test & & \multicolumn{3}{c}{Test partition (\%)} & Selected & Best test over & Optimism \\
\cmidrule(lr){4-6}
Benchmark & images & Classes & Zero-shot & Fine-tuned & Macro-F\textsubscript{1} & epoch & all epochs (\%) & (pp) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def tab_ft_calibration():
    files = sorted(glob.glob(str(RES / "eval_finetuned_*.json")))
    if not files:
        write("tab_ft_calibration.tex", "% eval_finetuned results not yet available; re-run make_tables.py\n")
        return
    rows = []
    for f in files:
        j = json.loads(Path(f).read_text())
        lab = {"plantdoc": "PlantDoc", "plantwild": "PlantWild"}[j["dataset"]]
        for mname, mlab in (("zero_shot", "Zero-shot"), ("fine_tuned", "Fine-tuned")):
            for cal, clab in (("uncalibrated", "none"), ("T_indomain_val", "in-domain $T$"),
                              ("T_target_val", "target $T$")):
                r = j["models"][mname][cal]
                rows.append(f"{lab} & {mlab} & {clab} & {r['temperature']:.3f} & {100*r['accuracy']:.2f} & "
                            f"{100*r['mean_confidence']:.2f} & {r['ece_15bin']:.3f} & {r['aurc']:.3f} & "
                            f"{100*r['coverage_at_risk_0.20']:.1f} \\\\")
            rows.append("\\addlinespace[2pt]")
    write("tab_ft_calibration.tex", r"""\begin{table}[!ht]
\centering
\caption{Calibration and selective prediction on the held-out field test partitions,
before and after fine-tuning, with no temperature scaling, with the temperature
fitted on the in-domain validation split, and with the temperature fitted on the
target validation split. Coverage is the largest fraction of test images that can
be retained with an error rate of at most 20\%.}
\label{tab:ftcalib}
\footnotesize
\begin{tabular}{lllrrrrrr}
\toprule
Benchmark & Model & Scaling & $T$ & Accuracy (\%) & Confidence (\%) & ECE & AURC & Coverage (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def tab_literature():
    """Published PlantDoc results. Literature values were checked against the
    source papers (Krishna et al. 2025, Table 1; Salman et al. 2025, Table 3);
    our values come from numbers.json."""
    t0 = next(iter(N["models"]))
    zs = N["all_models"]["plantdoc_aspect"]["accuracy"]
    ft = N["finetune_plantdoc"]
    n_all = f"{N['models'][t0]['plantdoc_aspect']['n']:,}".replace(",", "\\,")
    rows = [
        r"\citet{krishna2025multidataset}\textsuperscript{a} & EfficientNet-B3 & PlantDoc & PlantDoc & 27 & n.s. & 71.63 \\",
        r"\citet{salman2025wild} & ViT with mixture of experts & Balanced PlantVillage subset & PlantDoc subset\textsuperscript{b} & n.s. & n.s. & 74 \\",
        f"This work, zero-shot & Four architectures, 12 models & PlantVillage & PlantDoc & 28 & {n_all} & {pm(zs)} \\\\",
        f"This work, fine-tuned & Gated-fusion hybrid & PlantVillage, then PlantDoc & PlantDoc test partition & {ft['n_classes']} & {ft['n_test']} & {100*ft['finetuned_test_acc']:.2f} \\\\",
    ]
    write("tab_literature.tex", r"""\begin{table}[!ht]
\centering
\caption{Top-1 accuracy reported on PlantDoc. The rows differ in whether PlantDoc
images are used for training and in which images are evaluated, so they are not
directly comparable: \citet{krishna2025multidataset} train on PlantDoc, whereas the
other rows start from a PlantVillage-trained model.}
\label{tab:literature}
\footnotesize
\begin{tabularx}{\textwidth}{>{\hsize=1.15\hsize}Y>{\hsize=0.95\hsize}Y>{\hsize=0.95\hsize}Y>{\hsize=0.95\hsize}Yrrr}
\toprule
Study & Model & Training data & Evaluated on & Classes & Test $n$ & Top-1 (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}\\[2pt]
\raggedright\footnotesize
\textsuperscript{a}\,Value from that study's Experiment~1 results table; its abstract,
comparison table and conclusions state 73.31\% and its Key Findings paragraph 72.46\%
for the same experiment.
\textsuperscript{b}\,Class and image counts of the evaluated subset are not stated.
n.s.\ = not stated in the source.
\end{table}
""")


def tab_abstention():
    S = N["abstention"]["summary"]
    if S.get("n_models", 0) < 12:
        write("tab_abstention.tex", f"% abstention summary covers {S.get('n_models', 0)} of 12 models; re-run later\n")
        return
    B = ("plantvillage_test", "plantdoc", "plantwild")
    # does temperature scaling change the ranking? compare AURC per model
    per = N["abstention"]["per_model"]
    dmax = max(abs(j["benchmarks"][b]["uncalibrated"]["msp"]["aurc"] - j["benchmarks"][b]["temp_scaled"]["msp"]["aurc"])
               for j in per.values() for b in B)

    def cells(cal, key, scale=100, k=2):
        return " & ".join(pm(S[b][cal][key], scale, k) for b in B)

    def ood(score):
        return "-- & " + " & ".join(pm(S["ood_auroc_best_score_uncal"][f][score], 1, 3) for f in ("plantdoc", "plantwild"))

    body = [
        f"Area under risk--coverage curve & {cells('uncalibrated', 'aurc', 1, 3)} \\\\",
        f"Coverage at \\SI{{20}}{{\\percent}} risk (\\%) & {cells('uncalibrated', 'coverage_at_risk_0.20')} \\\\",
        f"Coverage at \\SI{{50}}{{\\percent}} risk (\\%) & {cells('uncalibrated', 'coverage_at_risk_0.50')} \\\\",
        f"Accuracy of most confident 10\\% (\\%) & {cells('uncalibrated', 'selective_acc_at_coverage_0.10')} \\\\",
        f"Accuracy of most confident 25\\% (\\%) & {cells('uncalibrated', 'selective_acc_at_coverage_0.25')} \\\\",
        f"Accuracy of most confident 50\\% (\\%) & {cells('uncalibrated', 'selective_acc_at_coverage_0.50')} \\\\",
        "\\midrule",
        f"ECE, no scaling & {cells('uncalibrated', 'ece_15bin', 1, 3)} \\\\",
        f"ECE, in-domain temperature & {cells('temp_scaled', 'ece_15bin', 1, 3)} \\\\",
        f"Mean confidence, in-domain temperature (\\%) & {cells('temp_scaled', 'mean_confidence')} \\\\",
        "\\midrule",
        "\\multicolumn{4}{l}{\\emph{Separating each field benchmark from the PlantVillage test split (AUROC)}} \\\\",
        f"Maximum softmax probability & {ood('msp')} \\\\",
        f"Negative predictive entropy & {ood('neg_entropy')} \\\\",
        f"Negative free energy & {ood('neg_energy')} \\\\",
    ]
    T = S["temperature"]
    rank_note = ("Temperature scaling leaves the ranking by confidence unchanged, so the selective-prediction "
                 "rows are identical with and without it" if dmax < 1e-9 else
                 f"With the in-domain temperature, AURC changes by at most {dmax:.4f}")
    write("tab_abstention.tex", r"""\begin{table}[!ht]
\centering
\caption{Selective prediction, calibration and distribution-level detection, mean $\pm$
standard deviation over the 12 models. Predictions are ranked by maximum softmax
probability; coverage is the largest fraction of predictions that can be kept with the
stated error rate among them. The in-domain temperature ($T = """ + f"{T['mean']:.3f} \\pm {T['sd']:.3f}" + r"""$) is
fitted on the PlantVillage validation split and applied unchanged. """ + rank_note + r""".}
\label{tab:abstention}
\footnotesize
\begin{tabular}{lccc}
\toprule
 & PlantVillage test & PlantDoc & PlantWild \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


if __name__ == "__main__":
    for fn in (tab_architecture, tab_field, tab_pairwise, tab_finetune, tab_ft_calibration, tab_literature,
               tab_abstention):
        fn()
