"""Generic classification/latency metrics and checkpoint utilities,
shared by every student evaluation path (currently evaluation/
router_eval.py, for the classifier and dual-head variants). Nothing
here is specific to any one student architecture -- that's the point of
keeping it separate: these functions only know about {query_id, route}
predictions and TeacherLabel records, not how a prediction was produced.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from typing import Optional

from common.config import ROUTING_LABELS, TEACHER_DIR, map_route
from common.schema import TeacherLabel, read_jsonl


def collect_true_pred_vs_teacher(
    predictions: list[dict],
    dataset_name: str,
    teacher_output_version: str,
    split: str,
    target_labels: Optional[tuple[str, ...]] = None,
) -> tuple[list[str], list[str]]:
    """(y_true, y_pred) -- teacher_route vs predicted route, for whichever
    predictions have a matching teacher label on `split`. predictions is
    a list of {"query_id": ..., "route": ...} dicts; the caller owns how
    those routes were produced.

    target_labels: the label space `pred["route"]` was produced in (e.g.
    a checkpoint's model.routing_labels). The teacher's cached label is
    always 3-way regardless of what the student was trained on, so when
    target_labels differs from ROUTING_LABELS (e.g. a binary-trained
    checkpoint), each teacher_route is collapsed via common.config.
    map_route() before comparison. None (default) skips this -- 3-way vs
    3-way, unchanged from before this parameter existed."""
    teacher_path = TEACHER_DIR / dataset_name / teacher_output_version / f"{split}.jsonl"
    teacher_routes = {label.query_id: label.teacher_route for label in read_jsonl(teacher_path, TeacherLabel)}

    y_true, y_pred = [], []
    for pred in predictions:
        teacher_route = teacher_routes.get(pred["query_id"])
        if teacher_route is None:
            continue
        if target_labels is not None:
            teacher_route = map_route(teacher_route, target_labels)
        y_true.append(teacher_route)
        y_pred.append(pred["route"])
    return y_true, y_pred


def classification_metrics(
    y_true: list[str], y_pred: list[str], labels: Optional[list[str]] = None
) -> dict:
    """accuracy, macro precision/recall/F1, per-tier P/R/F1, confusion
    matrix -- the sklearn-backed core every evaluation/selection path
    scores against, so a pooled multi-dataset (y_true, y_pred) is scored
    the same way as a single-dataset one.

    labels: the label space to score over (row/column order for per-tier
    metrics and the confusion matrix). None (default) falls back to
    common.config.ROUTING_LABELS (3-way) -- pass a checkpoint's
    model.routing_labels explicitly when scoring a binary-trained
    student."""
    from sklearn.metrics import classification_report, confusion_matrix

    if not y_true:
        return {"n_matched": 0}

    labels = list(labels) if labels is not None else list(ROUTING_LABELS)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "n_matched": len(y_true),
        "accuracy": round(report["accuracy"], 4),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall": round(report["macro avg"]["recall"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "per_tier": {
            tier: {
                "precision": round(report[tier]["precision"], 4),
                "recall": round(report[tier]["recall"], 4),
                "f1": round(report[tier]["f1-score"], 4),
                "support": int(report[tier]["support"]),
            }
            for tier in labels
        },
        "confusion_matrix": {
            "labels": labels,  # rows = true tier, cols = predicted tier
            "matrix": matrix.tolist(),
        },
    }


def latency_comparison(student_predictions: list[dict], teacher_latencies: list[float]) -> dict:
    """Average per-query latency, student vs teacher. teacher_latencies
    should be TeacherLabel.latency_seconds for the same queries (already
    the 2-call sum for the two-stage teacher) for a fair comparison."""
    student_avg = sum(p["latency_seconds"] for p in student_predictions) / len(student_predictions)
    teacher_avg = sum(teacher_latencies) / len(teacher_latencies) if teacher_latencies else 0.0
    return {
        "student_avg_latency_seconds": round(student_avg, 4),
        "teacher_avg_latency_seconds": round(teacher_avg, 4),
        "speedup": round(teacher_avg / student_avg, 2) if student_avg else None,
    }


def freeze_checkpoint(source: str, dest: str) -> None:
    """Copies the winning checkpoint to `dest` (e.g. checkpoints/student-
    classifier-best/), leaving `source` untouched. A separate, deliberate
    call -- nothing else freezes a checkpoint automatically, so "no
    further tuning after this point" stays an explicit decision, not an
    implicit side effect."""
    dest_path = Path(dest)
    if dest_path.exists():
        shutil.rmtree(dest_path)
    shutil.copytree(source, dest_path)
    print(f"Froze {source} -> {dest_path}")
