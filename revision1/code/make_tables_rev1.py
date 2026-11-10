"""Write the revision's LaTeX tables directly from the result JSONs (no hand transcription).
Outputs into ../manuscript/sections/: tab_calibration.tex, tab_selective.tex,
tab_finetune.tex, tab_ft_calibration.tex"""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"
S = Path(__file__).resolve().parents[1] / "manuscript" / "sections"
P = json.load(open(R / "posthoc_summary.json"))
FT = json.load(open(R / "ft_summary.json"))
PF = json.load(open(R / "post_ft_calib.json"))["summary"]
TTA = json.load(open(R / "tta_scores.json"))


def tta_stat(b, path):
    import numpy as np
    v = []
    for m in TTA.values():
        x = m
        for k in path(b):
            x = x[k]
        v.append(x)
    v = np.array(v, float)
    return [v.mean(), v.std(ddof=1), len(v)]


def pm(x, d=3, scale=1.0):
    return f"${x[0]*scale:.{d}f} \\pm {x[1]*scale:.{d}f}$"


# ---------------------------------------------------------------- calibration methods
names = [("uncalibrated", "None"), ("temperature", "Temperature scaling"),
         ("vector_scaling", "Vector scaling"), ("dirichlet_odir", "Dirichlet (ODIR)"),
         ("isotonic_toplabel", "Isotonic (top label)"), ("mc_dropout_head", "Monte Carlo dropout")]
rows = []
for k, lab in names:
    c = [P["calibration"][b][k] for b in ("plantdoc", "plantwild")]
    rows.append(f"{lab} & " + " & ".join(f"{pm(x['mean_conf'],1,100)} & {pm(x['ece'])} & {pm(x['nll'],2)}" for x in c) + r" \\")
e = [P["ensembles"][b]["calibration"] for b in ("plantdoc", "plantwild")]
rows.append(r"\addlinespace[2pt]" + "\nDeep ensemble (3 seeds)$^{a}$ & " + " & ".join(f"{pm(x['mean_conf'],1,100)} & {pm(x['ece'])} & {pm(x['nll'],2)}" for x in e) + r" \\")
o = [P["calibration"][b]["oracle_temperature_field_half"] for b in ("plantdoc", "plantwild")]
rows.append(r"\addlinespace[2pt]" + "\nTemperature fitted on field labels$^{b}$ & " + " & ".join(f"{pm(x['mean_conf'],1,100)} & {pm(x['ece'])} & {pm(x['nll'],2)}" for x in o) + r" \\")
acc = [P["calibration"][b]["uncalibrated"]["acc"] for b in ("plantdoc", "plantwild")]
eacc = [P["ensembles"][b]["calibration"]["acc"] for b in ("plantdoc", "plantwild")]
T = P["temperature"]
tex = r"""\begin{table}[!ht]
\centering
\caption{Calibration on the field benchmarks after post-hoc recalibration fitted on the
PlantVillage validation partition, mean $\pm$ standard deviation over the 12 models.
Accuracy without recalibration is """ + f"{pm(acc[0],1,100)}\\% on PlantDoc and {pm(acc[1],1,100)}\\% on PlantWild" + r"""; the
in-domain temperature is """ + f"$T = {T[0]:.3f} \\pm {T[1]:.3f}$" + r""". Confidence is the mean largest
softmax probability; ECE, expected calibration error (15 bins); NLL, negative
log-likelihood. $^{a}$Mean over the four architectures of the three-seed average
(accuracy """ + f"{pm(eacc[0],1,100)}\\% and {pm(eacc[1],1,100)}\\%" + r"""). $^{b}$Reference that uses field labels: fitted on
a random half of each benchmark and scored on the other half.}
\label{tab:calibration}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{3}{c}{PlantDoc} & \multicolumn{3}{c}{PlantWild} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Recalibration & Confidence (\%) & ECE & NLL & Confidence (\%) & ECE & NLL \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(S / "tab_calibration.tex").write_text(tex)

# ---------------------------------------------------------------- selective scores
sn = [("msp", "Largest softmax probability"), ("neg_entropy", "Negative entropy"),
      ("neg_energy", "Negative free energy"), ("mc_dropout_msp", "MC dropout probability"),
      ("neg_mc_mutual_info", "MC dropout mutual information"),
      ("mahalanobis", "Mahalanobis distance"), ("knn_cosine", "Nearest-neighbour similarity")]
rows = []
for b, bl in (("plantdoc", "PlantDoc"), ("plantwild", "PlantWild")):
    rows.append(r"\multicolumn{6}{l}{\emph{" + bl + r"}} \\")
    for k, lab in sn:
        x = P["selective"][b][k]; a = P["ood_auroc"][b].get(k)
        rows.append(f"\\quad {lab} & {pm(x['e_aurc'])} & {pm(x['cov_at_risk_0.20'],1,100)} & {pm(x['cov_at_risk_0.50'],1,100)} & "
                    f"{pm(x['acc_at_cov_0.10'],1,100)} & {pm(a) if a else '--'} \\\\")
    for k, lab in (("tta_prob", "TTA probability$^{b}$"), ("neg_tta_std", "TTA view standard deviation$^{b}$"),
                   ("tta_agreement", "TTA view agreement$^{b}$")):
        g = lambda key: tta_stat(b, lambda bb: (bb, "selective", k, key))
        a = tta_stat(b, lambda bb: ("ood_auroc", bb, k))
        rows.append(f"\\quad {lab} & {pm(g('e_aurc'))} & {pm(g('cov_at_risk_0.20'),1,100)} & {pm(g('cov_at_risk_0.50'),1,100)} & "
                    f"{pm(g('acc_at_cov_0.10'),1,100)} & {pm(a)} \\\\")
    for k, lab in (("ens_msp", "Ensemble probability$^{a}$"), ("neg_ens_mutual_info", "Ensemble mutual information$^{a}$")):
        x = P["ensembles"][b]["selective"][k]
        rows.append(f"\\quad {lab} & {pm(x['e_aurc'])} & {pm(x['cov_at_risk_0.20'],1,100)} & {pm(x['cov_at_risk_0.50'],1,100)} & "
                    f"{pm(x['acc_at_cov_0.10'],1,100)} & -- \\\\")
    rows.append(r"\addlinespace[3pt]")
tex = r"""\begin{table}[!ht]
\centering
\caption{Abstention on individual field images and separation of field from studio
images, by uncertainty score, mean $\pm$ standard deviation over the 12 models. E-AURC,
excess area under the risk--coverage curve over a perfect ranking (lower is better);
coverage, largest fraction of images that can be answered with at most 20\% or 50\% error
among them; top 10\%, accuracy of the 10\% of images with the most confident score;
AUROC, separation of the benchmark from the PlantVillage test images (1 is perfect, 0.5
chance). In domain, every model answers all PlantVillage test images within a 20\% error
budget. $^{a}$Four architectures, three seeds each. $^{b}$Test-time augmentation over
eight views (identity, two flips, two 90-degree rotations, three 87.5\% crops); E-AURC is
relative to the accuracy of the view-averaged prediction.}
\label{tab:selective}
\fontsize{8}{9.5}\selectfont
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}
\toprule
 & & \multicolumn{2}{c}{Coverage (\%) at risk} & & \\
\cmidrule(lr){3-4}
Score & E-AURC & 20\% & 50\% & Top 10\% (\%) & AUROC \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(S / "tab_selective.tex").write_text(tex)

# ---------------------------------------------------------------- fine-tuning
arch = [("effv2s", "EfficientNetV2-S"), ("hybrid_gated", "Gated hybrid")]
rows = []
for b, bl in (("plantdoc", "PlantDoc"), ("plantwild", "PlantWild")):
    for a, al in arch:
        f = FT[f"{b}_{a}"]
        g = lambda k: [f[k]["mean"], f[k]["sd"]]
        ep = f["selected_epoch"]["per_seed"]
        rows.append(f"{bl} & {al} & {pm(g('zero_shot_test_acc'),1,100)} & {pm(g('finetuned_test_acc'),1,100)} & "
                    f"{pm(g('finetuned_test_macro_f1'),1,100)} & {', '.join(str(x) for x in ep)} & "
                    f"{pm(g('r2_protocol_running_max_test_acc'),1,100)} & ${f['optimism_pp']['mean']:+.2f}$ \\\\")
tex = r"""\begin{table}[!ht]
\centering
\caption{Fine-tuning on labelled field images, mean $\pm$ standard deviation over three
seeds. PlantDoc: 1\,991 training, 351 validation and 236 test images (28 classes);
PlantWild: 5\,299, 760 and 1\,513 (33 classes). The epoch was selected on the validation
split and the test partition scored once. Macro-F\textsubscript{1} is averaged over the 28
or 33 classes of the benchmark. The last two columns give what selecting the epoch on the
test partition itself would have reported and the resulting overstatement in percentage
points (pp); they were not used for selection.}
\label{tab:finetune}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccccc}
\toprule
 & & \multicolumn{3}{c}{Test partition (\%)} & Selected & Best test over & Optimism \\
\cmidrule(lr){3-5}
Benchmark & Model & Zero-shot & Fine-tuned & Macro-F\textsubscript{1} & epochs & all epochs (\%) & (pp) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(S / "tab_finetune.tex").write_text(tex)

# ---------------------------------------------------------------- after fine-tuning
rows = []
for b, bl in (("plantdoc", "PlantDoc"), ("plantwild", "PlantWild")):
    for a, al in arch:
        s = PF[f"{b}_{a}"]
        for sc, scl in (("none", "none"), ("target_T", "target $T$")):
            x = s[sc]
            Tcell = "1" if sc == "none" else f"${s['T_target'][0]:.2f} \\pm {s['T_target'][1]:.2f}$"
            rows.append(f"{bl} & {al} & {scl} & {Tcell} & {pm(x['mean_conf'],1,100)} & {pm(x['ece'])} & "
                        f"{pm(x['sel_msp']['aurc'])} & {pm(x['sel_msp']['cov_at_risk_0.20'],1,100)} \\\\")
        rows.append(r"\addlinespace[2pt]")
tex = r"""\begin{table}[!ht]
\centering
\caption{Calibration and abstention on the held-out field test partitions after
fine-tuning, mean $\pm$ standard deviation over three seeds, without recalibration and
with a temperature fitted on the field validation split. Accuracy is given in
Table~\ref{tab:finetune}. Coverage is the largest fraction of test images that can be
answered with at most 20\% error among them; images are ranked by the largest softmax
probability.}
\label{tab:ftcalib}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lllccccc}
\toprule
Benchmark & Model & Recalibration & $T$ & Confidence (\%) & ECE & AURC & Coverage (\%) \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(S / "tab_ft_calibration.tex").write_text(tex)
print("tables written")
