"""Mean +- SD over seeds of full fine-tuning runs (RUN_OUT/0011_rev1/ft) -> results/ft_summary.json"""
import json, numpy as np
from pathlib import Path
FT = Path.home()/"RUN_OUT/0011_rev1/ft"; RES = Path(__file__).resolve().parents[1]/"results"
out = {}
for d in ("plantdoc", "plantwild"):
    for a in ("effv2s", "hybrid_gated"):
        J = [json.load(open(FT/f"ft_{d}_{a}_s{s}.json")) for s in (42, 1337, 2024)]
        out[f"{d}_{a}"] = {k: {"per_seed": [j[k] for j in J], "mean": float(np.mean([j[k] for j in J])), "sd": float(np.std([j[k] for j in J], ddof=1))}
                           for k in ("zero_shot_test_acc", "zero_shot_test_macro_f1", "finetuned_test_acc", "finetuned_test_macro_f1", "r2_protocol_running_max_test_acc", "optimism_pp", "selected_epoch")}
        out[f"{d}_{a}"]["n_test"] = J[0]["n_test"]; out[f"{d}_{a}"]["n_fit"] = J[0]["n_fit"]; out[f"{d}_{a}"]["n_val"] = J[0]["n_val"]
json.dump(out, open(RES/"ft_summary.json", "w"), indent=1); print("ok")
