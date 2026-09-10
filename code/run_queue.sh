#!/usr/bin/env bash
# R3 training queue. Every configuration is retrained on the single
# deterministic split (E11a/b), three seeds each (E12a).
set -u
cd "$(dirname "$0")"
export RESULTS_DIR=../results
LOG=../logs

run () {  # run <model> <fusion> <seed> <batch>
  local m=$1 f=$2 s=$3 b=$4 tag
  if [ "$m" = "hybrid" ] && [ "$f" = "token_cross_attn" ]; then tag="hybrid_tokenattn_s${s}"; elif [ "$m" = "hybrid" ]; then tag="hybrid_gated_s${s}"; else tag="${m}_s${s}"; fi
  if [ -f "../results/test_final_${tag}.json" ]; then
    echo "[queue] skip ${tag} (already done)"; return
  fi
  echo "[queue] === ${tag} start $(date -Is) ==="
  python3 train.py --model "$m" --fusion "$f" --epochs 15 --batch "$b" \
      --seed "$s" --tag "$tag" > "${LOG}/train_${tag}.log" 2>&1
  echo "[queue] === ${tag} done $(date -Is) rc=$? ==="
}

# Seed 42 first: one complete set of four configurations.
run hybrid token_cross_attn 42 20
run hybrid gated_linear     42 20
run effv2s  gated_linear    42 32
run swinv2t gated_linear    42 32
# Then the two extra seeds for the variance estimate.
for s in 1337 2024; do
  run hybrid gated_linear     "$s" 20
  run effv2s  gated_linear    "$s" 32
  run swinv2t gated_linear    "$s" 32
  run hybrid token_cross_attn "$s" 20
done
echo "[queue] ALL DONE $(date -Is)"
