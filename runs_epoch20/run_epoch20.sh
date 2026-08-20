#!/usr/bin/env bash
# Epoch-count ablation: does training longer (20 epochs instead of the
# paper's 3) change the route-only vs. reasoning-plus-route comparison?
# Single seed (42, the project default) -- this is not a seed-variance
# check (see multiseed/run_multiseed.sh for that), just an epoch sweep.
#
# select-router-checkpoint already evaluates every checkpoint-<step>
# (one per epoch, via student/route_head.py's SaveEpochCheckpointCallback)
# on validation and reports macro_f1/accuracy for each -- that IS the
# per-epoch accuracy curve, logged to stdout below.
#
# Usage: ./run_epoch20.sh 2>&1 | tee epoch20_experiment.log
set -euo pipefail

SEED=42
TEACHER="qwen2.5-3b-v2"
DATASETS=(gsm8k math)
EPOCHS=20

echo "=== route-only (epochs=$EPOCHS) ==="
python run.py train-student-classifier \
  --dataset "${DATASETS[@]}" \
  --teacher "$TEACHER" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --checkpoint-dir "runs_epoch20/route_only/seed_${SEED}"

echo "=== route-only: per-epoch validation accuracy/macro-F1 ==="
python run.py select-router-checkpoint \
  --variant classifier \
  --checkpoint-root "runs_epoch20/route_only/seed_${SEED}" \
  --dataset "${DATASETS[@]}" \
  --teacher "$TEACHER" \
  --freeze-to "runs_epoch20/route_only/seed_${SEED}/best"

echo "=== route-only: test evaluation (best epoch by validation macro-F1) ==="
python run.py evaluate-router \
  --variant classifier \
  --checkpoint-dir "runs_epoch20/route_only/seed_${SEED}/best" \
  --dataset "${DATASETS[@]}" \
  --teacher "$TEACHER" \
  --split test

echo "=== reasoning-plus-route (epochs=$EPOCHS, lambda_reason=0.5) ==="
python run.py train-student-dual \
  --dataset "${DATASETS[@]}" \
  --teacher "$TEACHER" \
  --epochs "$EPOCHS" \
  --lambda-route 1.0 \
  --lambda-reason 0.5 \
  --seed "$SEED" \
  --checkpoint-dir "runs_epoch20/reasoning_plus_route/seed_${SEED}"

echo "=== reasoning-plus-route: per-epoch validation accuracy/macro-F1 ==="
python run.py select-router-checkpoint \
  --variant dual \
  --checkpoint-root "runs_epoch20/reasoning_plus_route/seed_${SEED}" \
  --dataset "${DATASETS[@]}" \
  --teacher "$TEACHER" \
  --freeze-to "runs_epoch20/reasoning_plus_route/seed_${SEED}/best"

echo "=== reasoning-plus-route: test evaluation (best epoch by validation macro-F1) ==="
python run.py evaluate-router \
  --variant dual \
  --checkpoint-dir "runs_epoch20/reasoning_plus_route/seed_${SEED}/best" \
  --dataset "${DATASETS[@]}" \
  --teacher "$TEACHER" \
  --split test

echo "Done. Per-epoch curves are in this log; test predictions under"
echo "runs_epoch20/{route_only,reasoning_plus_route}/seed_${SEED}/best/eval_{gsm8k,math}_test/"
