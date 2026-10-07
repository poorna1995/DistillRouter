"""Evaluates a trained router checkpoint that predicts via a single
forward pass -- the route-only ClassifierRouter (student/train_classifier.py)
or the reasoning-plus-route DualHeadRouter (student/train_dual.py), both of which expose the same
load_*_checkpoint(dir) -> (model, tokenizer) and predict_route(model,
tokenizer, query) -> {route, probabilities, latency_seconds} contract.

Reuses evaluation/metrics.py's classification_metrics()/
collect_true_pred_vs_teacher()/latency_comparison()/freeze_checkpoint()
rather than reimplementing them -- those are generic (query_id, route)
utilities with no dependency on either architecture.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from common.config import PROCESSED_DIR, TEACHER_DIR
from common.schema import Example, TeacherLabel, read_jsonl
from evaluation.metrics import classification_metrics, collect_true_pred_vs_teacher, latency_comparison

__all__ = [
    "generate_router_predictions",
    "save_router_predictions",
    "evaluate_router_checkpoint",
    "select_best_router_checkpoint",
]


def generate_router_predictions(
    load_fn: Callable[[str], tuple], predict_fn: Callable, checkpoint_dir: str, dataset_name: str, split: str,
) -> tuple[list[dict], tuple[str, ...]]:
    """Runs a router checkpoint on every query in dataset_name/split.
    `split` should be one the checkpoint wasn't trained on (validation/
    test), not train, or accuracy measures memorization. Returns
    (predictions, routing_labels) -- routing_labels is the checkpoint's
    own model.routing_labels (small/large), read off the loaded model
    so callers can score/collapse-compare against it without having to
    know in advance which label space this checkpoint was trained on."""
    model, tokenizer = load_fn(checkpoint_dir)
    routing_labels = tuple(model.routing_labels)
    examples = read_jsonl(PROCESSED_DIR / dataset_name / f"{split}.jsonl", Example)

    predictions = []
    for example in examples:
        result = predict_fn(model, tokenizer, example.query)
        predictions.append({
            "query_id": example.id,
            "route": result["route"],
            "probabilities": result["probabilities"],
            "latency_seconds": result["latency_seconds"],
        })
    return predictions, routing_labels


def save_router_predictions(
    predictions: list[dict],
    dataset_name: str,
    teacher_output_version: str,
    split: str,
    output_path: str,
) -> int:
    """One row per example -- query_id, teacher_route, predicted_route,
    correct, the full probability distribution (whichever label space the
    checkpoint predicts in), and latency -- the artifact for error
    analysis and qualitative inspection, since the aggregate metrics alone
    can't answer "which queries did this checkpoint get wrong."""
    teacher_path = TEACHER_DIR / dataset_name / teacher_output_version / f"{split}.jsonl"
    teacher_labels = {label.query_id: label for label in read_jsonl(teacher_path, TeacherLabel)}

    rows = []
    for pred in predictions:
        teacher_label = teacher_labels.get(pred["query_id"])
        if teacher_label is None:
            continue
        teacher_route = teacher_label.teacher_route
        rows.append({
            "query_id": pred["query_id"],
            "teacher_route": teacher_route,
            "predicted_route": pred["route"],
            "correct": pred["route"] == teacher_route,
            "probabilities": pred["probabilities"],
            "latency_seconds": pred["latency_seconds"],
        })

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def evaluate_router_checkpoint(
    load_fn: Callable[[str], tuple],
    predict_fn: Callable,
    checkpoint_dir: str,
    dataset_name: str,
    teacher_output_version: str,
    split: str = "validation",
) -> dict:
    """Full report for one router checkpoint on one dataset/split:
    accuracy, macro P/R/F1, per-tier P/R/F1, confusion matrix -- all vs
    teacher_route, i.e. distillation fidelity, not oracle correctness
    (see evaluation/oracle_check.py for that) -- plus latency vs the
    teacher's own latency if available. Saves classification_report.json
    + predictions.jsonl under checkpoint_dir."""
    predictions, routing_labels = generate_router_predictions(load_fn, predict_fn, checkpoint_dir, dataset_name, split)
    y_true, y_pred = collect_true_pred_vs_teacher(
        predictions, dataset_name, teacher_output_version, split
    )
    report = classification_metrics(y_true, y_pred, labels=list(routing_labels))

    teacher_path = TEACHER_DIR / dataset_name / teacher_output_version / f"{split}.jsonl"
    if teacher_path.exists():
        teacher_labels = read_jsonl(teacher_path, TeacherLabel)
        report["latency"] = latency_comparison(predictions, [label.latency_seconds for label in teacher_labels])

    eval_dir = Path(checkpoint_dir) / f"eval_{dataset_name}_{split}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "classification_report.json").write_text(json.dumps(report, indent=2))
    n_saved = save_router_predictions(
        predictions, dataset_name, teacher_output_version, split, str(eval_dir / "predictions.jsonl"),
    )
    print(f"[{dataset_name}/{split}] Saved classification_report.json + predictions.jsonl ({n_saved} rows) -> {eval_dir}")
    return report


def select_best_router_checkpoint(
    load_fn: Callable[[str], tuple],
    predict_fn: Callable,
    checkpoint_root: str,
    dataset_names: list[str],
    teacher_output_version: str,
    split: str = "validation",
) -> dict:
    """Evaluates every checkpoint-<step> saved under checkpoint_root (one
    per epoch, from student/route_head.py's SaveEpochCheckpointCallback)
    against `split`, pooled across dataset_names, and picks the one with
    the highest macro F1 -- the same selection rule for both student variants. Doesn't
    freeze/copy anything -- see evaluation/metrics.py's
    freeze_checkpoint() for that."""
    all_dirs = [p for p in Path(checkpoint_root).glob("checkpoint-*") if p.is_dir()]
    numbered, skipped = [], []
    for p in all_dirs:
        (numbered if p.name.split("-")[-1].isdigit() else skipped).append(p)
    if skipped:
        print(f"select_best_router_checkpoint: skipping non-standard checkpoint dir(s): {[p.name for p in skipped]}")
    checkpoint_dirs = sorted(numbered, key=lambda p: int(p.name.split("-")[-1]))
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint-<step> directories found under {checkpoint_root}")

    results = []
    for ckpt in checkpoint_dirs:
        y_true_all: list[str] = []
        y_pred_all: list[str] = []
        routing_labels: tuple[str, ...] = ()
        for name in dataset_names:
            predictions, routing_labels = generate_router_predictions(load_fn, predict_fn, str(ckpt), name, split)
            y_true, y_pred = collect_true_pred_vs_teacher(
                predictions, name, teacher_output_version, split
            )
            y_true_all.extend(y_true); y_pred_all.extend(y_pred)

            eval_dir = ckpt / f"eval_{name}_{split}"
            save_router_predictions(
                predictions, name, teacher_output_version, split, str(eval_dir / "predictions.jsonl"),
            )

        report = classification_metrics(y_true_all, y_pred_all, labels=list(routing_labels) or None)
        report["checkpoint"] = str(ckpt)
        (ckpt / "classification_report.json").write_text(json.dumps(report, indent=2))
        results.append(report)
        print(f"  {ckpt.name}: macro_f1={report.get('macro_f1')}, accuracy={report.get('accuracy')}")

    best = max(results, key=lambda r: r.get("macro_f1", -1))
    print(f"Best checkpoint: {best['checkpoint']} (macro_f1={best['macro_f1']})")
    return {"best": best, "all_results": results}
