"""Compares a teacher's or student's predicted routes against oracle
ground truth. Reads teacher output (data/teacher/<dataset>/
<output_version>/<split>.jsonl), a student's saved predictions
(evaluation/router_eval.py's predictions.jsonl), and oracle labels
(data/oracle/<dataset>/<split>.labels.jsonl) from disk; no model calls."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from common.config import ORACLE_DIR, ROUTING_LABELS, TEACHER_DIR, map_route
from common.schema import TeacherLabel, read_jsonl, read_oracle_labels
from evaluation.metrics import classification_metrics


def teacher_accuracy_vs_oracle(dataset_names: list[str], split: str, output_version: str) -> dict:
    """Aggregate + per-tier accuracy. Reports n_matched vs n_oracle_labeled
    separately since teacher coverage may lag oracle coverage."""
    correct = 0
    matched = 0
    oracle_total = 0
    correct_by_tier = {tier: 0 for tier in ROUTING_LABELS}
    total_by_tier = {tier: 0 for tier in ROUTING_LABELS}

    for dataset_name in dataset_names:
        teacher_path = TEACHER_DIR / dataset_name / output_version / f"{split}.jsonl"
        oracle_path = ORACLE_DIR / dataset_name / f"{split}.labels.jsonl"
        if not oracle_path.exists():
            continue
        oracle_labels = read_oracle_labels(oracle_path)
        oracle_total += len(oracle_labels)
        if not teacher_path.exists():
            continue

        teacher_routes = {label.query_id: label.teacher_route for label in read_jsonl(teacher_path, TeacherLabel)}
        for oracle_label in oracle_labels:
            teacher_route = teacher_routes.get(oracle_label.query_id)
            if teacher_route is None:
                continue
            matched += 1
            is_correct = teacher_route == oracle_label.routing_label
            correct += int(is_correct)
            total_by_tier[oracle_label.routing_label] += 1
            correct_by_tier[oracle_label.routing_label] += int(is_correct)

    result = {
        "accuracy": round(correct / matched, 4) if matched else 0.0,
        "n_matched": matched,
        "n_oracle_labeled": oracle_total,
        "per_tier_accuracy": {
            tier: round(correct_by_tier[tier] / total_by_tier[tier], 4) if total_by_tier[tier] else None
            for tier in ROUTING_LABELS
        },
        "per_tier_support": total_by_tier,
    }
    print(
        f"teacher_accuracy_vs_oracle({dataset_names}, split={split!r}): "
        f"accuracy={result['accuracy']} ({matched}/{oracle_total} matched), "
        f"per_tier={result['per_tier_accuracy']}"
    )
    return result


def student_metrics_vs_oracle(
    prediction_paths: dict[str, str], split: str, target_labels: Optional[tuple[str, ...]] = None
) -> dict:
    """Compares a student's saved test-set predictions (evaluation/
    router_eval.py's predictions.jsonl, one path per dataset name in
    prediction_paths) against oracle ground truth on the same split.
    Reuses classification_metrics() -- the same scorer Table 3/4 already
    use -- so this is computed identically to teacher-vs-oracle
    (accuracy) and student-vs-teacher (accuracy + macro P/R/F1 +
    confusion matrix), just against a different reference. Reports
    n_oracle_labeled alongside n_matched since prediction coverage may
    lag oracle coverage.

    target_labels: same meaning as evaluation/router_eval.py's use of
    common.config.map_route() -- the oracle's routing_label is always
    3-way, so pass the checkpoint's model.routing_labels (e.g.
    BINARY_ROUTING_LABELS) here when scoring a binary-trained student's
    predictions.jsonl against oracle ground truth; predicted_route in
    that file is already in that label space (see save_router_
    predictions), so only the oracle side needs collapsing."""
    y_true, y_pred = [], []
    oracle_total = 0

    for dataset_name, pred_path in prediction_paths.items():
        oracle_path = ORACLE_DIR / dataset_name / f"{split}.labels.jsonl"
        if not oracle_path.exists():
            continue
        oracle_routes = {label.query_id: label.routing_label for label in read_oracle_labels(oracle_path)}
        oracle_total += len(oracle_routes)

        with Path(pred_path).open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                oracle_route = oracle_routes.get(row["query_id"])
                if oracle_route is None:
                    continue
                if target_labels is not None:
                    oracle_route = map_route(oracle_route, target_labels)
                y_true.append(oracle_route)
                y_pred.append(row["predicted_route"])

    report = classification_metrics(y_true, y_pred, labels=list(target_labels) if target_labels else None)
    report["n_oracle_labeled"] = oracle_total
    print(
        f"student_metrics_vs_oracle(split={split!r}): accuracy={report.get('accuracy')}, "
        f"macro_f1={report.get('macro_f1')} ({report.get('n_matched', 0)}/{oracle_total} matched)"
    )
    return report
