#!/usr/bin/env bash
# Binary-label-space counterpart of run_epoch20.sh: route-only vs.
# reasoning-plus-route, 20 epochs, single seed (42), but with
# --label-space binary (BINARY_ROUTING_LABELS = {cheap, large}, see
# common/config.py) instead of the paper's default 3-way space. Binary
# routing has never been run before this -- there is no 3-epoch
# baseline for it anywhere in the repo, so this establishes the whole
# epoch-0..20 curve in one pass rather than extending an existing run.
#
# Training reads data/student/{gsm8k,math}/v2/binary/train.jsonl
# (already built via build-binary-student-data). Evaluation needs no
# separate binary validation/test files -- evaluate-router/
# select-router-checkpoint read data/processed + the always-3-way
# cached teacher/oracle labels and collapse to binary post-hoc via
# common/config.py's map_route(), keyed off the checkpoint's own
# model.routing_labels.
#
# Usage: ./run_epoch20_binary.sh 2>&1 | tee epoch20_experiment_binary.log
set -euo pipefail

SEED=42
TEACHER="qwen2.5-3b-v2"
DATASETS=(gsm8k math)
EPOCHS=20

echo "=== route-only, binary label space (epochs=$EPOCHS) ==="
python run.py train-student-classifier \
  --dataset "${DATASETS[@]}" \
  --teacher "$TEACHER" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --label-space binary \
  --checkpoint-dir "runs_epoch20/route_only_binary/seed_${SEED}"

echo "=== route-only binary: per-epoch validation accuracy/macro-F1 ==="
python run.py select-router-checkpoint \
  --variant classifier \
  --checkpoint-root "runs_epoch20/route_only_binary/seed_${SEED}" \
  --dataset "${DATASETS[@]}" \
  --teacher "$TEACHER" \
  --freeze-to "runs_epoch20/route_only_binary/seed_${SEED}/best"

echo "=== route-only binary: test evaluation (best epoch by validation macro-F1) ==="
python run.py evaluate-router \
  --variant classifier \
  --checkpoint-dir "runs_epoch20/route_only_binary/seed_${SEED}/best" \
  --dataset "${DATASETS[@]}" \
  --teacher "$TEACHER" \
  --split test

echo "=== reasoning-plus-route, binary label space (epochs=$EPOCHS, lambda_reason=0.5) ==="
python run.py train-student-dual \
  --dataset "${DATASETS[@]}" \
  --teacher "$TEACHER" \
  --epochs "$EPOCHS" \
  --lambda-route 1.0 \
  --lambda-reason 0.5 \
  --seed "$SEED" \
  --label-space binary \
  --checkpoint-dir "runs_epoch20/reasoning_plus_route_binary/seed_${SEED}"

echo "=== reasoning-plus-route binary: per-epoch validation accuracy/macro-F1 ==="
python run.py select-router-checkpoint \
  --variant dual \
  --checkpoint-root "runs_epoch20/reasoning_plus_route_binary/seed_${SEED}" \
  --dataset "${DATASETS[@]}" \
  --teacher "$TEACHER" \
  --freeze-to "runs_epoch20/reasoning_plus_route_binary/seed_${SEED}/best"

echo "=== reasoning-plus-route binary: test evaluation (best epoch by validation macro-F1) ==="
python run.py evaluate-router \
  --variant dual \
  --checkpoint-dir "runs_epoch20/reasoning_plus_route_binary/seed_${SEED}/best" \
  --dataset "${DATASETS[@]}" \
  --teacher "$TEACHER" \
  --split test

echo "Done. Per-epoch curves are in this log; test predictions under"
echo "runs_epoch20/{route_only_binary,reasoning_plus_route_binary}/seed_${SEED}/best/eval_{gsm8k,math}_test/"
