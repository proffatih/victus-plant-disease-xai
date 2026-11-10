#!/usr/bin/env bash
# REV1 GPU queue, stage 1: waits for the feature dump, then memory probe,
# PlantDoc protocol analysis and label-free test-time adaptation.
set -u
cd "$(dirname "$0")"
LOG=../logs
while pgrep -f "dump_features.py" > /dev/null; do sleep 60; done
echo "[stage1] dump finished $(date -Is)"
python3 probe_mem.py > $LOG/probe_mem.log 2>&1
echo "[stage1] probe done $(date -Is)"
python3 plantdoc_crops.py > $LOG/plantdoc_crops.log 2>&1
echo "[stage1] crops done $(date -Is)"
python3 tta_rev1.py > $LOG/tta_rev1.log 2>&1
echo "[stage1] tta done $(date -Is)"
