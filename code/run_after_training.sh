#!/usr/bin/env bash
# R3 post-training pipeline. Runs once the training queue is empty.
set -u
cd "$(dirname "$0")"
export RESULTS_DIR=../results
L=../logs
R=../results

echo "[post] waiting for the training queue to drain..."
while pgrep -f "^python3 train[.]py" >/dev/null; do sleep 120; done
echo "[post] queue drained $(date -Is)"

# --- E3/E4: efficiency on an otherwise idle GPU ---------------------------
echo "[post] measuring efficiency"
python3 measure_efficiency.py > $L/measure_efficiency.log 2>&1

# --- evaluation of every trained checkpoint on all three benchmarks -------
for ck in $R/best_hybrid_gated_s*.pt $R/best_hybrid_tokenattn_s*.pt \
          $R/best_effv2s_s*.pt $R/best_swinv2t_s*.pt; do
  [ -f "$ck" ] || continue
  tag=$(basename "$ck" .pt); tag=${tag#best_}
  if [ -f "$R/eval_${tag}.json" ]; then echo "[post] skip eval ${tag}"; continue; fi
  echo "[post] evaluating ${tag}"
  python3 evaluate_r3.py --ckpt "$ck" --tag "$tag" > $L/eval_${tag}.log 2>&1
done

# --- E12 statistics across runs and seeds --------------------------------
echo "[post] statistics"
python3 stats_r3.py > $L/stats_r3.log 2>&1

# --- E5: fine-tuning with an honest validation split ---------------------
MAIN=$R/best_hybrid_gated_s42.pt
if [ -f "$MAIN" ]; then
  echo "[post] fine-tuning on PlantDoc"
  python3 finetune_r3.py --ckpt "$MAIN" --dataset plantdoc  --epochs 30 --batch 16 \
      --tag pd_gated_s42 > $L/ft_plantdoc.log 2>&1
  echo "[post] fine-tuning on PlantWild"
  python3 finetune_r3.py --ckpt "$MAIN" --dataset plantwild --epochs 20 --batch 16 \
      --tag pw_gated_s42 > $L/ft_plantwild.log 2>&1
  # --- E10: explainability evidence -------------------------------------
  echo "[post] Grad-CAM evidence set"
  python3 gradcam_r3.py --ckpt "$MAIN" --tag hybrid_gated_s42 \
      --per_class 10 --n_fail 40 --out_suffix _indomain > $L/gradcam.log 2>&1
fi

# --- Abstention / selective prediction under domain shift -----------------
for t in hybrid_gated_s42 effv2s_s42; do
  [ -f "$R/best_${t}.pt" ] || continue
  [ -f "$R/abstention_${t}.json" ] && continue
  echo "[post] abstention ${t}"
  python3 abstention.py --ckpt "$R/best_${t}.pt" --tag "$t" > $L/abstention_${t}.log 2>&1
done

# --- E8b: leakage-conditioned accuracy on the corrected predictions -------
echo "[post] duplicate analysis"
python3 duplicate_check.py > $L/duplicate_check2.log 2>&1

echo "[post] ALL DONE $(date -Is)"
