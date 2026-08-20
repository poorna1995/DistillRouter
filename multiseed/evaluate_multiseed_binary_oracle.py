"""Paired bootstrap + multi-seed aggregation for the route-only vs.
reasoning-plus-route comparison, BINARY label space, scored against the
independently constructed oracle instead of the teacher.

Companion to evaluate_multiseed_binary.py (teacher-fidelity, now the
paper's primary Table 3) the same way evaluate_multiseed_oracle.py is
the companion to evaluate_multiseed.py: this closes the one gap flagged
in the Limitations section before binary became the paper's primary
label space -- oracle-correctness for binary had only ever been checked
at a single seed (the exploratory 20-epoch run). Reuses the exact same
checkpoints run_multiseed_binary.sh already produced; no retraining.

Run run_multiseed_binary.sh first -- this script only reads the
predictions.jsonl files it produces.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from common.config import BINARY_ROUTING_LABELS, ORACLE_DIR, PROJECT_ROOT, map_route
from common.schema import read_oracle_labels

DATASETS = ["gsm8k", "math"]
N_BOOT = 10000
BOOT_SEED = 12345


def load_oracle_routes() -> dict[str, str]:
    """query_id -> oracle routing_label collapsed to binary, pooled across
    datasets, test split. The oracle's own labels are always 3-way; each
    is mapped down to {cheap, large} the same way a binary-trained
    checkpoint's predictions already are (predicted_route in
    predictions.jsonl), so only this side needs collapsing."""
    routes = {}
    for dataset_name in DATASETS:
        path = ORACLE_DIR / dataset_name / "test.labels.jsonl"
        for label in read_oracle_labels(path):
            routes[label.query_id] = map_route(label.routing_label, BINARY_ROUTING_LABELS)
    return routes


def load_preds_vs_oracle(checkpoint_dir: Path, oracle_routes: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qids, y_true, y_pred = [], [], []
    for dataset_name in DATASETS:
        path = checkpoint_dir / f"eval_{dataset_name}_test" / "predictions.jsonl"
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                oracle_route = oracle_routes.get(row["query_id"])
                if oracle_route is None:
                    continue
                qids.append(row["query_id"])
                y_true.append(oracle_route)
                y_pred.append(row["predicted_route"])
    return np.array(qids), np.array(y_true), np.array(y_pred)


def score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    if metric == "macro_f1":
        return f1_score(y_true, y_pred, labels=list(BINARY_ROUTING_LABELS), average="macro", zero_division=0)
    raise ValueError(metric)


def paired_bootstrap(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, metric: str,
    n_boot: int = N_BOOT, seed: int = BOOT_SEED,
) -> tuple[float, tuple[float, float]]:
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
    parser.add_argument("--runs-dir", default="runs_binary", metavar="DIR")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)), metavar="N")
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = PROJECT_ROOT / runs_dir

    oracle_routes = load_oracle_routes()

    per_seed = {metric: [] for metric in ("accuracy", "macro_f1")}
    variant_scores = {"route_only": {"accuracy": [], "macro_f1": []},
                       "reasoning_plus_route": {"accuracy": [], "macro_f1": []}}

    for seed in args.seeds:
        dir_a = runs_dir / "route_only" / f"seed_{seed}" / "best"
        dir_b = runs_dir / "reasoning_plus_route" / f"seed_{seed}" / "best"
        if not (dir_a / "eval_gsm8k_test" / "predictions.jsonl").exists():
            print(f"[seed {seed}] missing predictions under {dir_a} -- skipping")
            continue

        qids_a, y_true_a, pred_a = load_preds_vs_oracle(dir_a, oracle_routes)
        qids_b, y_true_b, pred_b = load_preds_vs_oracle(dir_b, oracle_routes)
        assert np.array_equal(qids_a, qids_b), f"seed {seed}: query_id order mismatch between variants"
        assert np.array_equal(y_true_a, y_true_b), f"seed {seed}: oracle route mismatch between variants"

        print(f"\n=== seed {seed} vs. oracle (n={len(y_true_a)} pooled test queries) ===")
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

    print(f"\n=== aggregated vs. oracle across {n_done} seed(s) ===")
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
        verdict = "route-only reliably ahead vs. oracle" if ci_lo > 0 else (
            "reasoning-plus-route reliably ahead vs. oracle" if ci_hi < 0 else
            "not statistically reliable -- CI (across seeds) includes 0"
        )
        print(f"  paired delta vs. oracle ({metric}): mean={mean_delta:+.4f}  "
              f"seed-spread 95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  -> {verdict}")


if __name__ == "__main__":
    main()
