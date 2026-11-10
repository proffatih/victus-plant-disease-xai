# Revision 1 — code and results

Experiments added in the first revision of the manuscript (calibration, abstention and
deferral of PlantVillage-trained classifiers on field images). Every number in the
revised manuscript is produced by a script in `code/` from a file in `results/`.

| Analysis | Script(s) | Result file(s) |
|---|---|---|
| Logits and features of the 12 reference models | `dump_features.py` | (large; regenerate from the checkpoints) |
| Post-hoc recalibration (temperature, vector, Dirichlet, isotonic, MC dropout), ten uncertainty scores, ensembles, feature-space deferral rule | `posthoc_rev1.py`, `summarise_posthoc.py` | `posthoc_rev1.json`, `posthoc_summary.json` |
| Test-time augmentation (8 views) | `tta_scores.py` | `tta_scores.json` |
| Label-free adaptation (AdaBN, TENT), cross-half protocol | `tta_rev1.py` | `tta_rev1.json` |
| Fine-tuning on field data (full and 5/10/25 images per class, three seeds) | `finetune_rev1.py`, `post_ft_calib.py`, `summarise_ft.py` | `ft_summary.json`, `post_ft_calib.json` |
| Grad-CAM faithfulness and branch ablation (6 models) | `gradcam_r3.py` | `gradcam/` |
| Grad-CAM against PlantSeg lesion masks | `gradcam_lesion.py` | `gradcam_lesion/records_*.csv`, `gradcam_lesion/summary*.json` |
| Errors in biological terms, class table | `bio_analysis.py`, `class_table.py` | `bio_analysis.json`, `class_table*.{csv,json}` |
| PlantDoc evaluation protocols (whole images vs single-leaf crops, label space) | `plantdoc_crops.py` | `plantdoc_protocols.json` |
| Controlled training protocol (all parameters trained, effective batch 32) | `train_ctrl.py`, `ctrl_eval.py`, `ctrl_stats.py` | `ctrl_stats.json`, `posthoc_ctrl.json`, `ctrl_branch_ablation.json`, `ctrl_predictions/` |
| Tables and figures | `make_tables_rev1.py`, `make_tab_ctrl.py`, `make_fig_kshot.py`, `make_fig_xai_rev1.py` | `../revision1/figures/` |

The GPU runs were executed in the order of `queue_stage1.sh`, `queue_stage2.sh` and
`queue_stage3.sh` on a single 4 GB laptop GPU. Image paths in the result files are relative
to the root of each dataset's original distribution. PlantSeg images are not
redistributed here (CC BY-NC 4.0); obtain them from the PlantSeg release.

Checkpoints of the controlled-protocol and fine-tuned models are attached to the release
`checkpoints-rev1`; the 12 reference models remain in `checkpoints-v1`.
