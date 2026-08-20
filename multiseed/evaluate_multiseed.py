"""Paired bootstrap + multi-seed aggregation for the route-only vs.
reasoning-plus-route comparison (Section 5.3/Table 4). Answers the
question Section 6.2 already raised but didn't quantify: is the 0.018
macro-F1 gap real, given seed-to-seed spread was measured at 0.023
(post-fix) to 0.060 (pre-fix)?

Run run_multiseed.sh first -- this script only reads the predictions.jsonl
files it produces, it runs no training itself.

Two separate uncertainty estimates, not one, per the reviewer's point:
  1. Query uncertainty within a seed: paired bootstrap over the 1,000
     pooled (gsm8k+math) test queries, same resampled indices for both
     variants.
  2. Seed uncertainty across seeds: the per-seed point deltas themselves,
     treated as independent samples of "how much does route-only beat
     reasoning-plus-route by."
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from common.config import PROJECT_ROOT, ROUTING_LABELS

DATASETS = ["gsm8k", "math"]
N_BOOT = 10000
BOOT_SEED = 12345  # resampling RNG seed, unrelated to the training SEEDS below


def load_preds(checkpoint_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pools eval_gsm8k_test/predictions.jsonl + eval_math_test/predictions.jsonl
    under checkpoint_dir -- matches Table 4's "pooled across both datasets"
    methodology, not a per-dataset comparison."""
    qids, y_true, y_pred = [], [], []
    for dataset_name in DATASETS:
        path = checkpoint_dir / f"eval_{dataset_name}_test" / "predictions.jsonl"
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                qids.append(row["query_id"])
                y_true.append(row["teacher_route"])
                y_pred.append(row["predicted_route"])
    return np.array(qids), np.array(y_true), np.array(y_pred)


def score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    if metric == "macro_f1":
        # labels= and zero_division=0 mirror evaluation/metrics.py's
        # classification_metrics() -- the *small* tier is ~6% of pooled
        # test data, so a resample missing it entirely must not silently
        # shrink the label set the macro average is taken over.
        return f1_score(y_true, y_pred, labels=list(ROUTING_LABELS), average="macro", zero_division=0)
    raise ValueError(metric)


def paired_bootstrap(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, metric: str,
    n_boot: int = N_BOOT, seed: int = BOOT_SEED,
) -> tuple[float, float, tuple[float, float]]:
    """Returns (score_a - score_b point estimate, that same point estimate
    again for convenience, (95% CI lo, hi)) -- same resampled indices used
    for both systems on every draw, so query-difficulty variance cancels."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = np.arange(n)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(idx, size=n, replace=True)
        deltas[i] = score(y_true[sample], pred_a[sample], metric) - score(y_true[sample], pred_b[sample], metric)
    point = score(y_true, pred_a, metric) - score(y_true, pred_b, metric)
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return point, (lo, hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir", default="runs", metavar="DIR",
        help="Directory containing route_only/ and reasoning_plus_route/ (relative to repo root, "
        "or absolute). Default: runs (the 270M-backbone study). Pass runs_1b for the "
        "gemma-3-1b-it backbone study.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(range(10)), metavar="N",
        help="Seed values to look for under --runs-dir (default: 0-9).",
    )
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = PROJECT_ROOT / runs_dir

    per_seed = {metric: [] for metric in ("accuracy", "macro_f1")}
    variant_scores = {"route_only": {"accuracy": [], "macro_f1": []},
                       "reasoning_plus_route": {"accuracy": [], "macro_f1": []}}

    for seed in args.seeds:
        dir_a = runs_dir / "route_only" / f"seed_{seed}" / "best"
        dir_b = runs_dir / "reasoning_plus_route" / f"seed_{seed}" / "best"
        if not (dir_a / "eval_gsm8k_test" / "predictions.jsonl").exists():
            print(f"[seed {seed}] missing predictions under {dir_a} -- run run_multiseed.sh first, skipping")
            continue

        qids_a, y_true_a, pred_a = load_preds(dir_a)
        qids_b, y_true_b, pred_b = load_preds(dir_b)
        assert np.array_equal(qids_a, qids_b), f"seed {seed}: query_id order mismatch between variants"
        assert np.array_equal(y_true_a, y_true_b), f"seed {seed}: teacher_route mismatch between variants"

        print(f"\n=== seed {seed} (n={len(y_true_a)} pooled test queries) ===")
        for metric in ("accuracy", "macro_f1"):
            score_a = score(y_true_a, pred_a, metric)
            score_b = score(y_true_a, pred_b, metric)
            point, ci = paired_bootstrap(y_true_a, pred_a, pred_b, metric)
            print(f"  {metric}: route_only={score_a:.4f}  reasoning_plus_route={score_b:.4f}  "
                  f"delta={point:+.4f}  95% CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                  f"{'  (CI excludes 0)' if ci[0] > 0 or ci[1] < 0 else '  (CI includes 0)'}")
            per_seed[metric].append(point)
            variant_scores["route_only"][metric].append(score_a)
            variant_scores["reasoning_plus_route"][metric].append(score_b)

    n_done = len(per_seed["accuracy"])
    if n_done == 0:
        print("\nNo completed seeds found -- nothing to aggregate.")
        return

    print(f"\n=== aggregated across {n_done} seed(s) ===")
    for metric in ("accuracy", "macro_f1"):
        for variant in ("route_only", "reasoning_plus_route"):
            vals = np.array(variant_scores[variant][metric])
            print(f"  {variant} {metric}: mean={vals.mean():.4f}  std={vals.std(ddof=1) if len(vals) > 1 else 0.0:.4f}"
                  f"  (n={len(vals)}, per-seed={[round(v, 4) for v in vals]})")
        deltas = np.array(per_seed[metric])
        mean_delta = deltas.mean()
        if len(deltas) > 1:
            ci_lo, ci_hi = np.quantile(deltas, [0.025, 0.975])
        else:
            ci_lo = ci_hi = mean_delta
        verdict = "route-only reliably ahead" if ci_lo > 0 else (
            "reasoning-plus-route reliably ahead" if ci_hi < 0 else
            "not statistically reliable -- CI (across seeds) includes 0"
        )
        print(f"  paired delta ({metric}): mean={mean_delta:+.4f}  seed-spread 95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]"
              f"  -> {verdict}")


if __name__ == "__main__":
    main()
