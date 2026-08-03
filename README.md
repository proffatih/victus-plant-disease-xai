# Reproducibility archive

Open code, results and figures for a manuscript currently under journal peer
review. Manuscript not distributed via this repository.

## Layout

- `code/`     — dataset loader, model, training, XAI, evaluation, ablation,
                target-domain fine-tuning, and statistics
- `data/`     — README pointing to public dataset sources
- `results/`  — training/eval metrics, per-image predictions, ablation and
                fine-tuning outputs, bootstrap-CI / McNemar summary (CSV/JSON)
- `figures/`  — generated figures (PDF/PNG)

## Reproducing the numbers

Set `DATA_ROOT` to the folder holding `plantvillage/` and `plantdoc/`, and
`RESULTS_DIR` to an output folder (defaults to `results`).

```bash
# 1. Splits and main hybrid model
python code/dataset.py
python code/train.py --model hybrid --epochs 15 --batch 16 --img 256
python code/evaluate.py            # in-domain + zero-shot PlantDoc metrics/predictions

# 2. Single-backbone ablation
python code/train.py --model effv2s  --epochs 15 --batch 16 --tag effv2s
python code/train.py --model swinv2t --epochs 15 --batch 16 --tag swinv2t
python code/eval_backbone.py --tag hybrid
python code/eval_backbone.py --tag effv2s
python code/eval_backbone.py --tag swinv2t

# 3. Target-domain fine-tuning on PlantDoc (train split -> disjoint test split)
python code/finetune_plantdoc.py --epochs 30 --lr 5e-5

# 4. Bootstrap 95% CIs + McNemar tests
python code/compute_stats.py       # writes results/stats_summary.json

# 5. Figures
python code/gradcam.py
python code/make_figures.py
```

Trained on a laptop RTX 3050 Ti Mobile GPU (4 GB) with FP16 mixed-precision.
Model checkpoints (`*.pt`) are excluded from version control; all reported
numbers regenerate from the code above on the public PlantVillage and PlantDoc
datasets (see `data/README.md`).

## Key results

- PlantVillage test top-1 accuracy 99.63% (macro-F1 99.33%), bootstrap 95% CI reported.
- Single-backbone ablation: the cross-attention fusion matches the best single
  backbone in-domain (McNemar: no significant difference).
- Zero-shot cross-dataset accuracy on PlantDoc quantifies the studio-to-field gap;
  target-domain fine-tuning on the disjoint PlantDoc test split recovers a large
  fraction of it (21.19% -> 63.14%).
