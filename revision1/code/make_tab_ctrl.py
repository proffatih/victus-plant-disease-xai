"""tab_ctrl: the four architectures under the controlled protocol (results/ctrl_stats.json)."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
S = json.load(open(ROOT / "results" / "ctrl_stats.json"))
NAME = {"effv2s": "EfficientNetV2-S", "swinv2t": "Swin V2-T",
        "hybrid_gated": "Hybrid, gated linear fusion", "hybrid_tokenattn": "Hybrid, token cross-attention"}
f = lambda v, d=2, s=100: f"${v[0]*s:.{d}f} \\pm {v[1]*s:.{d}f}$"
rows = []
for a in NAME:
    c, r = S["ctrl"][a], S["ref"][a]
    rows.append(" & ".join([NAME[a], f(c["pv_test"]["acc"], 3),
        f(c["plantdoc"]["acc"], 1), f(c["plantdoc"]["ece"], 3, 1), f"{r['plantdoc']['acc'][0]*100:.1f}",
        f(c["plantwild"]["acc"], 1), f(c["plantwild"]["ece"], 3, 1), f"{r['plantwild']['acc'][0]*100:.1f}"]) + r" \\")
tex = r"""\begin{table}[!ht]
\centering
\caption{The four architectures under the controlled protocol (all parameters trained,
effective batch 32 and the same schedule for every model). Mean $\pm$ standard deviation
over three seeds; PlantDoc and PlantWild are evaluated zero-shot. ``Ref.'' is the field
accuracy of the same architecture under the reference protocol (Table~\ref{tab:arch}).}
\label{tab:ctrl}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lrrrrrrr}
\toprule
 & PlantVillage & \multicolumn{3}{c}{PlantDoc} & \multicolumn{3}{c}{PlantWild} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Architecture & accuracy (\%) & Accuracy (\%) & ECE & Ref. & Accuracy (\%) & ECE & Ref. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(ROOT / "manuscript" / "sections" / "tab_ctrl.tex").write_text(tex)
print(tex)
