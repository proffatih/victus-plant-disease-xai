#!/usr/bin/env bash
# REV1 GPU queue, stage 2: fine-tuning across seeds/architectures, label-budget
# curve, Grad-CAM across seeds. Waits for stage 1.
set -u
cd "$(dirname "$0")"
LOG=../logs; R3=$(readlink -f ../../R3/results)
while pgrep -f "queue_stage1.sh" > /dev/null; do sleep 60; done
echo "[stage2] start $(date -Is)"
ft () {  # ft <arch_seed> <dataset> <seed> <epochs> <extra...>
  local t=$1 d=$2 s=$3 e=$4; shift 4
  local tag="${d}_${t}${TAGSUF:-}"
  [ -f "$HOME/RUN_OUT/0011_rev1/ft/ft_${tag}.json" ] && { echo "[skip] $tag"; return; }
  python3 finetune_rev1.py --ckpt "$R3/best_${t}.pt" --dataset "$d" --epochs "$e" --batch 16 \
      --seed "$s" --tag "$tag" "$@" > "$LOG/ft_${tag}.log" 2>&1
  echo "[stage2] ft $tag rc=$? $(date -Is)"
}
for a in effv2s hybrid_gated; do for s in 42 1337 2024; do
  ft "${a}_s${s}" plantdoc  $s 30
  ft "${a}_s${s}" plantwild $s 20
done; done
for k in 5 10 25; do for s in 42 1337 2024; do
  TAGSUF="_k${k}" ft "effv2s_s${s}" plantdoc  $s 30 --kshot $k --select last
  TAGSUF="_k${k}" ft "effv2s_s${s}" plantwild $s 30 --kshot $k --select last
done; done
export RESULTS_DIR=$(readlink -f ../results/gradcam)
for a in effv2s hybrid_gated; do for s in 42 1337 2024; do
  t="${a}_s${s}"
  [ -f "$RESULTS_DIR/gradcam_summary_${t}_indomain.json" ] || python3 gradcam_r3.py --ckpt "$R3/best_${t}.pt" --tag "$t" \
      --per_class 10 --n_fail 40 --out_suffix _indomain > "$LOG/gradcam_${t}_indomain.log" 2>&1
  [ -f "$RESULTS_DIR/gradcam_summary_${t}_field.json" ] || python3 gradcam_r3.py --ckpt "$R3/best_${t}.pt" --tag "$t" \
      --per_class 8 --n_fail 60 --field_failures --out_suffix _field > "$LOG/gradcam_${t}_field.log" 2>&1
  echo "[stage2] gradcam $t $(date -Is)"
done; done
echo "[stage2] done $(date -Is)"
for a in effv2s hybrid_gated; do for s in 42 1337 2024; do
  t="${a}_s${s}"
  [ -f "../results/gradcam_lesion/summary_${t}.json" ] || python3 gradcam_lesion.py "$t" > "$LOG/gradcam_lesion_${t}.log" 2>&1
  echo "[stage2] lesion $t $(date -Is)"
done; done
echo "[stage2] lesion done $(date -Is)"
# R3 comment 3: test-time-augmentation uncertainty scores (12 reference models)
python3 tta_scores.py > "$LOG/tta_scores.log" 2>&1
echo "[stage2] tta_scores rc=$? $(date -Is)"
