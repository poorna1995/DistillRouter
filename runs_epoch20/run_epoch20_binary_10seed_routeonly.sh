#!/usr/bin/env bash
# 10-seed extension of run_epoch20_binary.sh's route-only half only --
# tests whether the 20-epoch validation-macro-F1 gains seen at seed 42
# (single-seed epoch-count ablation) hold up as a real, statistically
# confirmed test-set improvement over this paper's actual 3-epoch/
# 10-seed main-protocol result for binary/270M route-only (Table 3:
# 0.651 +/- 0.005 macro-F1 vs. teacher), the paper's primary
# configuration. Reasoning-plus-route and 3-way are deliberately not
# included here -- binary/270M route-only was singled out as the one
# configuration worth spending compute on before submission.
#
# Resumable: skips any seed whose test predictions already exist.
#
# Usage: ./run_epoch20_binary_10seed_routeonly.sh
set -euo pipefail

SEEDS=(0 1 2 3 4 5 6 7 8 9)
TEACHER="qwen2.5-3b-v2"
DATASETS=(gsm8k math)
EPOCHS=20
RUNS_DIR="runs_epoch20_binary_10seed/route_only"

done_already() {
  local dir="$1"
  [[ -f "${dir}/eval_gsm8k_test/predictions.jsonl" && -f "${dir}/eval_math_test/predictions.jsonl" ]]
}

for SEED in "${SEEDS[@]}"; do
  if done_already "${RUNS_DIR}/seed_${SEED}/best"; then
    echo "=== [20ep binary RO] seed ${SEED}: already done, skipping ==="
    continue
  fi

  echo "=== [20ep binary RO] seed ${SEED}: training (epochs=${EPOCHS}) ==="
  python run.py train-student-classifier \
    --dataset "${DATASETS[@]}" \
    --teacher "$TEACHER" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --label-space binary \
    --checkpoint-dir "${RUNS_DIR}/seed_${SEED}"

  echo "=== [20ep binary RO] seed ${SEED}: checkpoint selection (validation macro-F1) ==="
  python run.py select-router-checkpoint \
    --variant classifier \
    --checkpoint-root "${RUNS_DIR}/seed_${SEED}" \
    --dataset "${DATASETS[@]}" \
    --teacher "$TEACHER" \
    --freeze-to "${RUNS_DIR}/seed_${SEED}/best"

  echo "=== [20ep binary RO] seed ${SEED}: test evaluation ==="
  python run.py evaluate-router \
    --variant classifier \
    --checkpoint-dir "${RUNS_DIR}/seed_${SEED}/best" \
    --dataset "${DATASETS[@]}" \
    --teacher "$TEACHER" \
    --split test
done

echo "All [20ep binary RO] seeds done. Predictions under ${RUNS_DIR}/seed_*/best/eval_{gsm8k,math}_test/predictions.jsonl"
