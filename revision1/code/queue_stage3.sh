#!/usr/bin/env bash
# REV1 GPU queue, stage 3: re-run of the two failed stage-1 jobs, then the
# controlled architecture comparison (all parameters trainable, micro-batch 16
# x 2 accumulation = effective batch 32, identical schedule for every model).
set -u
cd "$(dirname "$0")"
LOG=../logs
# stage 2 finished 2026-11-09 06:24; the pgrep wait matched its own launcher shell, so it was removed
echo "[stage3] start $(date -Is)"
[ -f ../results/plantdoc_protocols.json ] || python3 plantdoc_crops.py > $LOG/plantdoc_crops.log 2>&1
echo "[stage3] crops rc=$? $(date -Is)"
python3 tta_rev1.py > $LOG/tta_rev1.log 2>&1
echo "[stage3] tta rc=$? $(date -Is)"
ctrl () {  # ctrl <model> <fusion> <seed> <tagbase> <extra...>
  local m=$1 f=$2 s=$3 tb=$4; shift 4
  local tag="ctrl_${tb}_s${s}"
  [ -f "$HOME/RUN_OUT/0011_rev1/ctrl/test_final_${tag}.json" ] && { echo "[skip] $tag"; return; }
  python3 train_ctrl.py --model "$m" --fusion "$f" --epochs 15 --batch 16 --accum 2 \
      --unfreeze --seed "$s" --tag "$tag" "$@" > "$LOG/train_${tag}.log" 2>&1
  echo "[stage3] $tag rc=$? $(date -Is)"
}
for s in 42 1337 2024; do
  ctrl hybrid gated_linear     $s hybrid_gated     --grad-ckpt
  ctrl hybrid token_cross_attn $s hybrid_tokenattn --grad-ckpt
  ctrl effv2s gated_linear     $s effv2s
  ctrl swinv2t gated_linear    $s swinv2t
done
echo "[stage3] done $(date -Is)"
