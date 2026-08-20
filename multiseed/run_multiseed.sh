#!/usr/bin/env bash
# Multi-seed training/eval for the route-only vs. reasoning-plus-route
# comparison (Section 5.3/Table 4). Trains, selects, and evaluates both
# variants across the same seed list so evaluate_multiseed.py can run a
# paired bootstrap over the shared test queries.
#
# Protocol matches research.md/research.tex Section 4.3 exactly: select
# checkpoints by validation macro-F1 (select-router-checkpoint), evaluate
# once on test (evaluate-router) -- only the seed varies across runs.
#
# Usage: ./run_multiseed.sh
set -euo pipefail

SEEDS=(0 1 2 3 4 5 6 7 8 9)
TEACHER="qwen2.5-3b-v2"
DATASETS=(gsm8k math)

# Skip a seed+variant already fully evaluated (predictions.jsonl present for
# both datasets) -- lets this be re-run to extend the seed list (e.g. 5->10)
# without redoing the seeds already done.
done_already() {
  local dir="$1"
  [[ -f "${dir}/eval_gsm8k_test/predictions.jsonl" && -f "${dir}/eval_math_test/predictions.jsonl" ]]
}

for SEED in "${SEEDS[@]}"; do
  if done_already "runs/route_only/seed_${SEED}/best"; then
    echo "=== seed ${SEED}: route-only -- already done, skipping ==="
  else
    echo "=== seed ${SEED}: route-only ==="
    python run.py train-student-classifier \
      --dataset "${DATASETS[@]}" \
      --teacher "$TEACHER" \
      --seed "$SEED" \
      --checkpoint-dir "runs/route_only/seed_${SEED}"

    python run.py select-router-checkpoint \
      --variant classifier \
      --checkpoint-root "runs/route_only/seed_${SEED}" \
      --dataset "${DATASETS[@]}" \
      --teacher "$TEACHER" \
      --freeze-to "runs/route_only/seed_${SEED}/best"

    python run.py evaluate-router \
      --variant classifier \
      --checkpoint-dir "runs/route_only/seed_${SEED}/best" \
      --dataset "${DATASETS[@]}" \
      --teacher "$TEACHER" \
      --split test
  fi

  if done_already "runs/reasoning_plus_route/seed_${SEED}/best"; then
    echo "=== seed ${SEED}: reasoning-plus-route -- already done, skipping ==="
  else
    echo "=== seed ${SEED}: reasoning-plus-route (lambda_reason=0.5, the value Section 6.1 already selected) ==="
    python run.py train-student-dual \
      --dataset "${DATASETS[@]}" \
      --teacher "$TEACHER" \
      --lambda-route 1.0 \
      --lambda-reason 0.5 \
      --seed "$SEED" \
      --checkpoint-dir "runs/reasoning_plus_route/seed_${SEED}"

    python run.py select-router-checkpoint \
      --variant dual \
      --checkpoint-root "runs/reasoning_plus_route/seed_${SEED}" \
      --dataset "${DATASETS[@]}" \
      --teacher "$TEACHER" \
      --freeze-to "runs/reasoning_plus_route/seed_${SEED}/best"

    python run.py evaluate-router \
      --variant dual \
      --checkpoint-dir "runs/reasoning_plus_route/seed_${SEED}/best" \
      --dataset "${DATASETS[@]}" \
      --teacher "$TEACHER" \
      --split test
  fi
done

echo "All seeds done. Predictions are under runs/{route_only,reasoning_plus_route}/seed_*/best/eval_{gsm8k,math}_test/predictions.jsonl"
echo "Next: python evaluate_multiseed.py"
