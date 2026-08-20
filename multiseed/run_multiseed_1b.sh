#!/usr/bin/env bash
# Same protocol as run_multiseed.sh, but at a larger student backbone
# (google/gemma-3-1b-it instead of the default google/gemma-3-270m-it --
# the "large" candidate tier, already cached locally, same Gemma-3
# family) -- answers Section 7's "Single student model scale" limitation:
# does the route-only vs. reasoning-plus-route result change with more
# student capacity? Kept as a separate script rather than parameterizing
# run_multiseed.sh so this can be written/launched without touching that
# script while it may still be mid-run.
#
# First 5 seeds (0-4) reversed the 270M finding: reasoning-plus-route came
# out reliably ahead on teacher-fidelity macro-F1 (95% CI entirely negative),
# the opposite direction from 270M's trend. Extended to 10 seeds to give
# that result the same rigor as the 270M study before it goes in the paper.
# done_already() skips the first 5, already-finished seeds.
#
# Usage: ./run_multiseed_1b.sh
set -euo pipefail

SEEDS=(0 1 2 3 4 5 6 7 8 9)
TEACHER="qwen2.5-3b-v2"
DATASETS=(gsm8k math)
MODEL_ID="google/gemma-3-1b-it"
RUNS_DIR="runs_1b"

done_already() {
  local dir="$1"
  [[ -f "${dir}/eval_gsm8k_test/predictions.jsonl" && -f "${dir}/eval_math_test/predictions.jsonl" ]]
}

for SEED in "${SEEDS[@]}"; do
  if done_already "${RUNS_DIR}/route_only/seed_${SEED}/best"; then
    echo "=== [1B] seed ${SEED}: route-only -- already done, skipping ==="
  else
    echo "=== [1B] seed ${SEED}: route-only ==="
    python run.py train-student-classifier \
      --dataset "${DATASETS[@]}" \
      --teacher "$TEACHER" \
      --model-id "$MODEL_ID" \
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
    echo "=== [1B] seed ${SEED}: reasoning-plus-route -- already done, skipping ==="
  else
    echo "=== [1B] seed ${SEED}: reasoning-plus-route (lambda_reason=0.5, same setting Section 6.1 selected at 270M -- not re-swept at this scale, see note below) ==="
    python run.py train-student-dual \
      --dataset "${DATASETS[@]}" \
      --teacher "$TEACHER" \
      --model-id "$MODEL_ID" \
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

echo "All [1B] seeds done. Predictions are under ${RUNS_DIR}/{route_only,reasoning_plus_route}/seed_*/best/eval_{gsm8k,math}_test/predictions.jsonl"
echo "Next: python evaluate_multiseed.py --runs-dir runs_1b (or the 1B-specific eval script)"
