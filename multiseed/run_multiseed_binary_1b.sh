#!/usr/bin/env bash
# Same protocol as run_multiseed_binary.sh, but at the 1B student backbone
# (google/gemma-3-1b-it, same combination run_multiseed_1b.sh already
# proved works for the 3-way label space -- runs_1b/, Tables 6/10).
# Completes the last missing cell of the 2x2 grid (label space x backbone):
# 3-way x 270M (done), 3-way x 1B (done), binary x 270M (done), binary x 1B
# (this script).
#
# Usage: ./run_multiseed_binary_1b.sh
set -euo pipefail

SEEDS=(0 1 2 3 4 5 6 7 8 9)
TEACHER="qwen2.5-3b-v2"
DATASETS=(gsm8k math)
MODEL_ID="google/gemma-3-1b-it"
RUNS_DIR="runs_binary_1b"

done_already() {
  local dir="$1"
  [[ -f "${dir}/eval_gsm8k_test/predictions.jsonl" && -f "${dir}/eval_math_test/predictions.jsonl" ]]
}

for SEED in "${SEEDS[@]}"; do
  if done_already "${RUNS_DIR}/route_only/seed_${SEED}/best"; then
    echo "=== [1B binary] seed ${SEED}: route-only -- already done, skipping ==="
  else
    echo "=== [1B binary] seed ${SEED}: route-only ==="
    python run.py train-student-classifier \
      --dataset "${DATASETS[@]}" \
      --teacher "$TEACHER" \
      --model-id "$MODEL_ID" \
      --label-space binary \
      --seed "$SEED" \
      --checkpoint-dir "${RUNS_DIR}/route_only/seed_${SEED}"

    python run.py select-router-checkpoint \
      --variant classifier \
      --checkpoint-root "${RUNS_DIR}/route_only/seed_${SEED}" \
      --dataset "${DATASETS[@]}" \
      --teacher "$TEACHER" \
      --freeze-to "${RUNS_DIR}/route_only/seed_${SEED}/best"

    python run.py evaluate-router \
      --variant classifier \
      --checkpoint-dir "${RUNS_DIR}/route_only/seed_${SEED}/best" \
      --dataset "${DATASETS[@]}" \
      --teacher "$TEACHER" \
      --split test
  fi

  if done_already "${RUNS_DIR}/reasoning_plus_route/seed_${SEED}/best"; then
    echo "=== [1B binary] seed ${SEED}: reasoning-plus-route -- already done, skipping ==="
  else
    echo "=== [1B binary] seed ${SEED}: reasoning-plus-route (lambda_reason=0.5) ==="
    python run.py train-student-dual \
      --dataset "${DATASETS[@]}" \
      --teacher "$TEACHER" \
      --model-id "$MODEL_ID" \
      --label-space binary \
      --lambda-route 1.0 \
      --lambda-reason 0.5 \
      --seed "$SEED" \
      --checkpoint-dir "${RUNS_DIR}/reasoning_plus_route/seed_${SEED}"

    python run.py select-router-checkpoint \
      --variant dual \
      --checkpoint-root "${RUNS_DIR}/reasoning_plus_route/seed_${SEED}" \
      --dataset "${DATASETS[@]}" \
      --teacher "$TEACHER" \
      --freeze-to "${RUNS_DIR}/reasoning_plus_route/seed_${SEED}/best"

    python run.py evaluate-router \
      --variant dual \
      --checkpoint-dir "${RUNS_DIR}/reasoning_plus_route/seed_${SEED}/best" \
      --dataset "${DATASETS[@]}" \
      --teacher "$TEACHER" \
      --split test
  fi
done

echo "All [1B binary] seeds done. Predictions are under ${RUNS_DIR}/{route_only,reasoning_plus_route}/seed_*/best/eval_{gsm8k,math}_test/predictions.jsonl"
