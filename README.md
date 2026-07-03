# Reproducibility archive

Open code, results and figures for a manuscript currently under journal peer
review. Manuscript not distributed via this repository.

## Layout

- `code/`     — dataset loader, model, training, XAI, evaluation
- `data/`     — README pointing to public dataset sources
- `results/`  — training/eval metrics (CSV/JSON)
- `figures/`  — generated figures (PDF/PNG)

## Reproducing the numbers

```bash
python code/dataset.py            # writes split_{train,val,test}.csv and plantdoc_shared.csv
python code/train.py --model hybrid --epochs 15 --batch 16 --img 256
python code/evaluate.py           # writes *_predictions.csv, *_metrics.json, confusion_matrix_pv.npy
python code/gradcam.py            # writes fig5_gradcam_samples.{pdf,png}
python code/make_figures.py       # writes fig1..fig8
```

Trained on a laptop RTX~3050~Ti Mobile GPU (4~GB) with FP16 mixed-precision.
