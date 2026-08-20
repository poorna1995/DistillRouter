#!/usr/bin/env bash
# Multi-seed training/eval for the route-only vs. reasoning-plus-route
# comparison, BINARY label space ({small,medium}->cheap, large->large) --
# the ten-seed version of the paper's single-seed §6.5/§6.7 binary check.
# Same protocol as run_multiseed.sh (Section 4.3 / Table 4), 270M backbone,
# just --label-space binary added to both train-student-* calls.
#
# Usage: ./run_multiseed_binary.sh
set -euo pipefail

SEEDS=(0 1 2 3 4 5 6 7 8 9)
TEACHER="qwen2.5-3b-v2"
DATASETS=(gsm8k math)

done_already() {
  local dir="$1"
  [[ -f "${dir}/eval_gsm8k_test/predictions.jsonl" && -f "${dir}/eval_math_test/predictions.jsonl" ]]
}

for SEED in "${SEEDS[@]}"; do
  if done_already "runs_binary/route_only/seed_${SEED}/best"; then
    echo "=== seed ${SEED}: route-only (binary) -- already done, skipping ==="
  else
    echo "=== seed ${SEED}: route-only (binary) ==="
    python run.py train-student-classifier \
      --dataset "${DATASETS[@]}" \
      --teacher "$TEACHER" \
      --label-space binary \
      --seed "$SEED" \
      --checkpoint-dir "runs_binary/route_only/seed_${SEED}"

    python run.py select-router-checkpoint \
      --variant classifier \
      --checkpoint-root "runs_binary/route_only/seed_${SEED}" \
      --dataset "${DATASETS[@]}" \
      --teacher "$TEACHER" \
      --freeze-to "runs_binary/route_only/seed_${SEED}/best"

    python run.py evaluate-router \
      --variant classifier \
      --checkpoint-dir "runs_binary/route_only/seed_${SEED}/best" \
      --dataset "${DATASETS[@]}" \
      --teacher "$TEACHER" \
      --split test
  fi

  if done_already "runs_binary/reasoning_plus_route/seed_${SEED}/best"; then
    echo "=== seed ${SEED}: reasoning-plus-route (binary) -- already done, skipping ==="
  else
    echo "=== seed ${SEED}: reasoning-plus-route (binary, lambda_reason=0.5) ==="
    python run.py train-student-dual \
      --dataset "${DATASETS[@]}" \
      --teacher "$TEACHER" \
      --label-space binary \
      --lambda-route 1.0 \
      --lambda-reason 0.5 \
      --seed "$SEED" \
      --checkpoint-dir "runs_binary/reasoning_plus_route/seed_${SEED}"

    python run.py select-router-checkpoint \
      --variant dual \
      --checkpoint-root "runs_binary/reasoning_plus_route/seed_${SEED}" \
      --dataset "${DATASETS[@]}" \
      --teacher "$TEACHER" \
      --freeze-to "runs_binary/reasoning_plus_route/seed_${SEED}/best"

    python run.py evaluate-router \
      --variant dual \
      --checkpoint-dir "runs_binary/reasoning_plus_route/seed_${SEED}/best" \
      --dataset "${DATASETS[@]}" \
      --teacher "$TEACHER" \
      --split test
  fi
done

echo "All seeds done. Predictions are under runs_binary/{route_only,reasoning_plus_route}/seed_*/best/eval_{gsm8k,math}_test/predictions.jsonl"
