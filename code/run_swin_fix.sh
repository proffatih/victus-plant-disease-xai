#!/usr/bin/env bash
# swinv2t 4 GB'da batch 32 ile OOM veriyor -> batch 12 ile yeniden.
set -u
cd "$(dirname "$0")"
export RESULTS_DIR=../results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "^python3 train[.]py" >/dev/null; do sleep 60; done
for s in 42 1337 2024; do
  [ -f "../results/test_final_swinv2t_s${s}.json" ] && { echo "[fix] skip s${s}"; continue; }
  echo "[fix] === swinv2t_s${s} start $(date -Is) ==="
  python3 train.py --model swinv2t --epochs 15 --batch 12 --seed "$s" \
      --tag "swinv2t_s${s}" > "../logs/train_swinv2t_s${s}.log" 2>&1
  echo "[fix] === swinv2t_s${s} done $(date -Is) rc=$? ==="
done
echo "[fix] ALL DONE $(date -Is)"
./run_after_training.sh > ../logs/post.log 2>&1
