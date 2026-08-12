"""Compares a teacher's predicted routes against oracle ground truth.
Reads teacher output (data/teacher/<dataset>/<output_version>/<split>.jsonl)
and oracle labels (data/oracle/<dataset>/<split>.labels.jsonl) from disk;
no model calls."""
from __future__ import annotations

from common.config import ORACLE_DIR, ROUTING_LABELS, TEACHER_DIR
from common.schema import TeacherLabel, read_jsonl, read_oracle_labels


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

    return {
        "accuracy": round(correct / matched, 4) if matched else 0.0,
        "n_matched": matched,
        "n_oracle_labeled": oracle_total,
        "per_tier_accuracy": {
            tier: round(correct_by_tier[tier] / total_by_tier[tier], 4) if total_by_tier[tier] else None
            for tier in ROUTING_LABELS
        },
        "per_tier_support": total_by_tier,
    }
