"""REV1 (R2-1, R3-1): evaluate the 12 controlled-protocol models.

Controlled protocol = every parameter trainable in every architecture, effective batch
32 (16 x 2 accumulation), identical schedule (train_ctrl.py). This script

  1. dumps logits and penultimate features of each controlled model on the same image
     sets as the reference models (dump_features.run), into the controlled-run
     directory, so that posthoc_rev1.py can be re-run on them unchanged
     (RUN_OUT=<ctrl dir> POSTHOC_OUT=posthoc_ctrl.json);
  2. runs the branch ablation of the controlled hybrids (gradcam_r3.branch_attribution:
     one branch's pooled features replaced by zeros) on the full PlantVillage test
     split, PlantDoc and PlantWild.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

CTRL = Path.home() / "RUN_OUT" / "0011_rev1" / "ctrl"
os.environ["RUN_OUT"] = str(CTRL)          # dump_features writes into OUT = $RUN_OUT
sys.path.insert(0, str(Path(__file__).parent))
from dump_features import load_model, train_subset, run, R3RES      # noqa: E402
from dataset import LeafDataset, build_transforms, load_split_csv, index_plantdoc_shared  # noqa: E402
from plantwild import index_plantwild_shared                        # noqa: E402
from gradcam_r3 import branch_attribution                           # noqa: E402

OUTD = Path(__file__).resolve().parents[1] / "results"
ARCHS = ["effv2s", "swinv2t", "hybrid_gated", "hybrid_tokenattn"]
SEEDS = [42, 1337, 2024]


def sets_for(c2i):
    return {
        "pv_trainsub": train_subset(),
        "pv_val": load_split_csv(R3RES / "split_val.csv"),
        "pv_test": load_split_csv(R3RES / "split_test.csv"),
        "plantdoc": [(p, c) for p, c in index_plantdoc_shared(c2i) if c in c2i],
        "plantwild": index_plantwild_shared(c2i),
    }


def main():
    device = "cuda"
    abl_path = OUTD / "ctrl_branch_ablation.json"
    abl = json.load(open(abl_path)) if abl_path.exists() else {}
    for a in ARCHS:
        for s in SEEDS:
            tag = f"{a}_s{s}"
            dst = CTRL / f"feats_{tag}.npz"
            need_dump = not dst.exists()
            need_abl = a.startswith("hybrid") and tag not in abl
            if not (need_dump or need_abl):
                print(f"[skip] {tag}"); continue
            model, c2i, S = load_model(CTRL / f"best_ctrl_{tag}.pt", device)
            sets = sets_for(c2i)
            if need_dump:
                blob = {}
                for k, items in sets.items():
                    lg, ft, y = run(model, items, c2i, S, device)
                    blob[f"{k}_logits"], blob[f"{k}_feats"], blob[f"{k}_y"] = lg, ft.astype(np.float16), y
                    blob[f"{k}_paths"] = np.array([p for p, _ in items])
                    print(f"[{tag}] {k}: n={len(y)} acc={(lg.argmax(1) == y).mean():.4f}", flush=True)
                ln, lin = model.head[0], model.head[2]
                blob["ln_w"], blob["ln_b"] = ln.weight.detach().cpu().numpy(), ln.bias.detach().cpu().numpy()
                blob["lin_w"], blob["lin_b"] = lin.weight.detach().cpu().numpy(), lin.bias.detach().cpu().numpy()
                np.savez_compressed(dst, **blob)
            if need_abl:
                tf = build_transforms(S, train=False, preproc="aspect")
                abl[tag] = {}
                for k in ("pv_test", "plantdoc", "plantwild"):
                    dl = DataLoader(LeafDataset(sets[k], c2i, tf), batch_size=32,
                                    shuffle=False, num_workers=6)
                    abl[tag][k] = branch_attribution(model, dl, device, n_batches=10**9)
                    print(f"[{tag}] ablation {k}: {abl[tag][k]}", flush=True)
                json.dump(abl, open(abl_path, "w"), indent=1)
            del model; torch.cuda.empty_cache()
    print("[done]")


if __name__ == "__main__":
    main()
