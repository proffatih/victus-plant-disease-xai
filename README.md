# Confidence calibration and abstention in multi-crop plant disease classification

Code, per-image predictions, trained checkpoints and analysis for the manuscript
*Confidence calibration and abstention in multi-crop plant disease classification and
cross-dataset generalisation*.

Twelve classifiers — EfficientNetV2-S, Swin V2-T, and two EfficientNetV2-S + Swin V2-T
hybrids (one with gated linear late fusion, one with token-level cross-attention), each
trained with three seeds — are trained on PlantVillage and evaluated on the PlantVillage
test split and, zero-shot, on PlantDoc and PlantWild. Every number in the manuscript is
derived from the files in `results/` and collected in `numbers.json`.

## Contents

| Path | Contents |
|---|---|
| `code/` | Dataset handling, models, training, evaluation, calibration and abstention, fine-tuning, Grad-CAM, efficiency, capture-bias and near-duplicate controls, statistics |
| `analysis/` | Aggregates `results/` into `numbers.json` and regenerates the figures and LaTeX tables |
| `results/` | Split manifests, class mappings, per-image predictions and all metrics |
| `figures/` | Figures (PDF, PNG) and the plotted values behind each one (`figures/data/`) |
| `numbers.json` | Every aggregate quoted in the manuscript |
| `requirements.txt` | Pinned environment |
| Release [`checkpoints-v1`](https://github.com/proffatih/victus-plant-disease-xai/releases/tag/checkpoints-v1) | All 14 trained checkpoints (12 PlantVillage models, 2 fine-tuned models) |
| `archive/earlier_release/` | The repository as previously released, kept unchanged for reference (see below) |

Image paths in the manifests and prediction files are relative to `DATA_ROOT`.

### Key result files

Model tags are `{effv2s,swinv2t,hybrid_gated,hybrid_tokenattn}_s{42,1337,2024}`.

| File | Contents |
|---|---|
| `split_{train,val,test}.csv`, `class_to_idx.json` | PlantVillage split (80/10/10, seed 42) and label indices |
| `plantdoc_shared.csv`, `plantwild_mapping.json`, `plantwild_split_42.csv` | Field-benchmark label mappings, excluded classes and the PlantWild fine-tuning split |
| `test_final_<tag>.json`, `train_log_<tag>.csv` | Training runs and selected checkpoints |
| `eval_<tag>.json`, `preds_<tag>_<benchmark>_<aspect\|square>.csv` | Evaluation metrics and per-image predictions with confidence |
| `abstention_<tag>.json`, `riskcov_*.npy` | Calibration, temperature scaling, selective prediction and distribution-level detection |
| `ft_{pd,pw}_gated_s42.json`, `ftlog_*.csv`, `eval_finetuned_*.json` | Fine-tuning with validation-based selection, and calibration after it |
| `efficiency_measured.json` | Parameters, MACs, model-only latency, throughput and memory |
| `stats_r3.json` | Exact McNemar tests, Holm correction, minimum significant differences |
| `background_bias.json`, `duplicate_analysis.json`, `duplicate_test_nn.csv` | Low-level colour probes and perceptual-hash near-duplicate analysis |
| `gradcam_summary_*.json`, `gradcam_records_*.csv`, `gradcam_*_curve_*.npy` | Grad-CAM faithfulness, off-leaf saliency and branch ablation |
| `fusion_degeneracy_proof.json` | Numerical check that attention over length-one sequences is a fixed linear map |

## Reproducing

```bash
pip install -r requirements.txt
export DATA_ROOT=/path/to/datasets
#   $DATA_ROOT/plantvillage/PV_root/Plant_leave_diseases_dataset_without_augmentation/
#   $DATA_ROOT/plantdoc/PlantDoc-Dataset/{train,test}/      https://github.com/pratikkayal/PlantDoc-Dataset
#   $DATA_ROOT/plantwild/extracted/plantwild/images/        https://huggingface.co/datasets/uqtwei2/PlantWild

cd code
bash run_queue.sh            # 12 training runs, about 36 h on an RTX 3050 Ti Laptop GPU (4 GB)
bash run_after_training.sh   # evaluation, efficiency, fine-tuning, Grad-CAM, statistics
for t in {effv2s,swinv2t,hybrid_gated,hybrid_tokenattn}_s{42,1337,2024}; do
  python abstention.py --ckpt ../results/best_$t.pt --tag $t
done
python eval_finetuned.py --base_ckpt ../results/best_hybrid_gated_s42.pt \
    --ft_ckpt ../results/best_pd_gated_s42.pt --dataset plantdoc --tag pd_gated_s42
python eval_finetuned.py --base_ckpt ../results/best_hybrid_gated_s42.pt \
    --ft_ckpt ../results/best_pw_gated_s42.pt --dataset plantwild --tag pw_gated_s42
python background_bias.py && python duplicate_check.py && python prove_degeneracy.py

cd ../analysis
python collect_numbers.py && python make_figures.py && python make_tables.py
```

To evaluate without retraining, download the checkpoints from the release into
`results/` and start from `run_after_training.sh`. The analysis scripts alone
(`analysis/`) regenerate `numbers.json` and every figure from the released results
without a GPU or the datasets.

## Earlier release

`archive/earlier_release/` is the repository as it was previously released and is kept
unchanged because the manuscript refers to its predictions. Its code has known issues that
are corrected here: the fusion block applied attention over sequences of length one and
was therefore a fixed linear map; the efficiency figure used constants rather than
measurements; latency timed the data pipeline; fine-tuning selected the epoch on the test
split; directory listings were not sorted, so the split was filesystem-dependent; and field
images were resized to a square without preserving aspect ratio. Do not use that code.

## License

See `LICENSE`.
