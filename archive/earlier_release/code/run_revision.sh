#!/usr/bin/env bash
# Clean continuation pipeline: hybrid checkpoint already trained & verified
# (best_hybrid.pt, 99.72% test). Runs PlantDoc fine-tuning + the two
# single-backbone ablations. Single instance only.
set -e
cd "$(dirname "$0")/.."
export DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to your dataset root}
export RESULTS_DIR="${RESULTS_DIR:-results}"
export PYTHONUNBUFFERED=1
LOG="$RESULTS_DIR/pipeline2.log"
mark(){ echo "=== STAGE_DONE:$1 ===" >> "$RESULTS_DIR/progress2.txt"; }
echo "PIPELINE2_START $(date)" | tee -a "$LOG"

# 1) PlantDoc target-domain fine-tuning (uses verified best_hybrid.pt)
python3 code/finetune_plantdoc.py --epochs 30 --lr 5e-5 >> "$LOG" 2>&1
mark finetune_plantdoc

# 2) EfficientNetV2-S single-backbone ablation
python3 code/train.py --model effv2s --epochs 15 --batch 16 --tag effv2s --workers 4 >> "$LOG" 2>&1
mark train_effv2s
python3 code/eval_backbone.py --tag effv2s >> "$LOG" 2>&1
mark eval_effv2s

# 3) Swin V2-T single-backbone ablation
python3 code/train.py --model swinv2t --epochs 15 --batch 16 --tag swinv2t --workers 4 >> "$LOG" 2>&1
mark train_swinv2t
python3 code/eval_backbone.py --tag swinv2t >> "$LOG" 2>&1
mark eval_swinv2t

echo "PIPELINE2_DONE $(date)" | tee -a "$LOG"
mark ALL_DONE
