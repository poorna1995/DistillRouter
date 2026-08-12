"""Evaluates a trained student checkpoint: routing accuracy vs teacher
(distillation fidelity), latency vs teacher, and reasoning-diversity
stats. Cross-domain generalization isn't separate code — it's the same
accuracy function called with a checkpoint trained on one dataset and
evaluated on another dataset's split.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from common.config import PROCESSED_DIR, ROUTING_LABELS, TEACHER_DIR
from common.hf_generation import generate_chat_response, load_causal_lm
from common.schema import Example, TeacherLabel, read_jsonl
from student.data import _STUDENT_INSTRUCTION

_ROUTE_RE = re.compile(r"<ROUTE>\s*(\w+)", re.IGNORECASE)
_REASON_RE = re.compile(r"<REASON>\s*(.*?)(?:<ROUTE>|$)", re.IGNORECASE | re.DOTALL)


def parse_student_response(raw_text: str) -> tuple[str, str]:
    """(reasoning, route) from a student's <REASON>/<ROUTE> tagged output.
    route falls back to "large" (same conservative default the teacher's
    JSON parser uses) if no tag is found or the value isn't a real tier."""
    reason_match = _REASON_RE.search(raw_text)
    reasoning = reason_match.group(1).strip() if reason_match else ""

    route_match = _ROUTE_RE.search(raw_text)
    route = route_match.group(1).strip().lower() if route_match else ""
    if route not in ROUTING_LABELS:
        route = "large"
    return reasoning, route


def generate_student_predictions(
    checkpoint_dir: str, dataset_name: str, split: str, max_new_tokens: int = 80
) -> list[dict]:
    """Runs the student checkpoint on every query in dataset_name/split.
    Returns one dict per example: query_id, route, reasoning,
    latency_seconds. `split` should be one the checkpoint wasn't trained
    on (validation/test), not train, or "accuracy" measures memorization."""
    tokenizer, model = load_causal_lm(checkpoint_dir)
    examples = read_jsonl(PROCESSED_DIR / dataset_name / f"{split}.jsonl", Example)

    predictions = []
    for example in examples:
        messages = [{"role": "user", "content": _STUDENT_INSTRUCTION.format(query=example.query)}]
        result = generate_chat_response(tokenizer, model, messages, max_new_tokens)
        reasoning, route = parse_student_response(result.text)
        predictions.append({
            "query_id": example.id,
            "route": route,
            "reasoning": reasoning,
            "latency_seconds": result.latency_seconds,
        })
    return predictions


def route_accuracy_vs_teacher(
    predictions: list[dict], dataset_name: str, teacher_output_version: str, split: str
) -> dict:
    """Student's predicted route vs the teacher_route it was trained to
    imitate, on `split`. Measures distillation fidelity (imitation
    quality), not correctness — see evaluation/oracle_check.py for
    teacher-vs-oracle; a student-vs-oracle variant would need oracle
    labels on `split`, which don't exist yet (oracle only covers
    train/calibration right now)."""
    teacher_path = TEACHER_DIR / dataset_name / teacher_output_version / f"{split}.jsonl"
    teacher_routes = {label.query_id: label.teacher_route for label in read_jsonl(teacher_path, TeacherLabel)}

    matched = 0
    correct = 0
    correct_by_tier = {tier: 0 for tier in ROUTING_LABELS}
    total_by_tier = {tier: 0 for tier in ROUTING_LABELS}
    for pred in predictions:
        teacher_route = teacher_routes.get(pred["query_id"])
        if teacher_route is None:
            continue
        matched += 1
        is_correct = pred["route"] == teacher_route
        correct += int(is_correct)
        total_by_tier[teacher_route] += 1
        correct_by_tier[teacher_route] += int(is_correct)

    return {
        "accuracy": round(correct / matched, 4) if matched else 0.0,
        "n_matched": matched,
        "n_predictions": len(predictions),
        "per_tier_accuracy": {
            tier: round(correct_by_tier[tier] / total_by_tier[tier], 4) if total_by_tier[tier] else None
            for tier in ROUTING_LABELS
        },
    }


def collect_true_pred_vs_teacher(
    predictions: list[dict], dataset_name: str, teacher_output_version: str, split: str
) -> tuple[list[str], list[str]]:
    """(y_true, y_pred) — teacher_route vs predicted route, for whichever
    predictions have a matching teacher label on `split`. Shared by
    teacher_agreement_report() (one dataset) and select_best_checkpoint()
    (pools several datasets before scoring)."""
    teacher_path = TEACHER_DIR / dataset_name / teacher_output_version / f"{split}.jsonl"
    teacher_routes = {label.query_id: label.teacher_route for label in read_jsonl(teacher_path, TeacherLabel)}

    y_true, y_pred = [], []
    for pred in predictions:
        teacher_route = teacher_routes.get(pred["query_id"])
        if teacher_route is None:
            continue
        y_true.append(teacher_route)
        y_pred.append(pred["route"])
    return y_true, y_pred


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """accuracy, macro precision/recall/F1, per-tier P/R/F1, confusion
    matrix — the sklearn-backed core of teacher_agreement_report(),
    factored out so select_best_checkpoint() can score a pooled
    multi-dataset (y_true, y_pred) the same way."""
    from sklearn.metrics import classification_report, confusion_matrix

    if not y_true:
        return {"n_matched": 0}

    labels = list(ROUTING_LABELS)
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


def teacher_agreement_report(
    predictions: list[dict], dataset_name: str, teacher_output_version: str, split: str
) -> dict:
    """Teacher Agreement (Distillation Fidelity) — full classification
    report of student route vs teacher route: accuracy, macro
    precision/recall/F1, per-tier precision/recall/F1, and a confusion
    matrix. Name it that way in any writeup: this measures how well the
    student imitates the teacher, not whether either of them is actually
    correct (that would be a student-vs-oracle report, which needs oracle
    labels this split doesn't have yet). Complements
    route_accuracy_vs_teacher()'s plain-accuracy summary — with the real
    class imbalance in this data (small is a small minority tier, see
    label distributions measured earlier), a student that collapsed to
    always predicting medium/large could still score high raw accuracy;
    macro F1 and the confusion matrix are what actually catch that."""
    y_true, y_pred = collect_true_pred_vs_teacher(predictions, dataset_name, teacher_output_version, split)
    return classification_metrics(y_true, y_pred)


def save_predictions(
    predictions: list[dict], dataset_name: str, teacher_output_version: str, split: str, output_path: str
) -> int:
    """Writes one row per example — query_id, teacher_route, student_route,
    correct, student_reasoning, teacher_reasoning — to output_path. The
    aggregate metrics (accuracy, macro F1, ...) can't answer "which
    queries did the student get wrong, and what did it say instead" —
    this is the artifact for error analysis, DPO pair construction, and
    qualitative examples in a writeup. Returns the row count written."""
    teacher_path = TEACHER_DIR / dataset_name / teacher_output_version / f"{split}.jsonl"
    teacher_labels = {label.query_id: label for label in read_jsonl(teacher_path, TeacherLabel)}

    rows = []
    for pred in predictions:
        teacher_label = teacher_labels.get(pred["query_id"])
        if teacher_label is None:
            continue
        rows.append({
            "query_id": pred["query_id"],
            "teacher_route": teacher_label.teacher_route,
            "student_route": pred["route"],
            "correct": pred["route"] == teacher_label.teacher_route,
            "student_reasoning": pred["reasoning"],
            "teacher_reasoning": teacher_label.routing_reasoning,
        })

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def select_best_checkpoint(
    checkpoint_root: str, dataset_names: list[str], teacher_output_version: str, split: str = "validation"
) -> dict:
    """Evaluates every checkpoint-<step> saved under checkpoint_root (one
    per epoch, from train_student's save_strategy="epoch") against
    `split`, pooled across dataset_names, and picks the one with the
    highest macro F1 — the actual target metric, not a loss proxy. Saves
    classification_report.json + predictions.jsonl under each checkpoint
    evaluated (not just the winner) — useful for comparing error patterns
    across epochs later, same schema evaluate-student's test-split run
    uses. Doesn't copy or rename anything; freezing the winner is
    freeze_checkpoint(), a separate, deliberate call."""
    checkpoint_dirs = sorted(
        (p for p in Path(checkpoint_root).glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint-<step> directories found under {checkpoint_root}")

    results = []
    for ckpt in checkpoint_dirs:
        y_true_all: list[str] = []
        y_pred_all: list[str] = []
        for name in dataset_names:
            predictions = generate_student_predictions(str(ckpt), name, split)
            y_true, y_pred = collect_true_pred_vs_teacher(predictions, name, teacher_output_version, split)
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)

            eval_dir = ckpt / f"eval_{name}_{split}"
            save_predictions(predictions, name, teacher_output_version, split, str(eval_dir / "predictions.jsonl"))

        report = classification_metrics(y_true_all, y_pred_all)
        report["checkpoint"] = str(ckpt)
        (ckpt / "classification_report.json").write_text(json.dumps(report, indent=2))
        results.append(report)
        print(f"  {ckpt.name}: macro_f1={report.get('macro_f1')}, accuracy={report.get('accuracy')}")

    best = max(results, key=lambda r: r.get("macro_f1", -1))
    print(f"Best checkpoint: {best['checkpoint']} (macro_f1={best['macro_f1']})")
    return {"best": best, "all_results": results}


def freeze_checkpoint(source: str, dest: str) -> None:
    """Copies the winning checkpoint to `dest` (e.g. checkpoints/student-best/),
    leaving `source` untouched. A separate, deliberate call — nothing
    else freezes a checkpoint automatically, so "no further tuning after
    this point" stays an explicit decision, not an implicit side effect."""
    import shutil

    dest_path = Path(dest)
    if dest_path.exists():
        shutil.rmtree(dest_path)
    shutil.copytree(source, dest_path)
    print(f"Froze {source} -> {dest_path}")


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


def reasoning_diversity(reasoning_texts: list[str]) -> dict:
    """Average length (words) and exact-string uniqueness ratio. Cheap,
    no embedding model. See reasoning_embedding_similarity() for a
    semantic (not just exact-string) diversity check."""
    non_empty = [r for r in reasoning_texts if r]
    lengths = [len(r.split()) for r in non_empty]
    unique_ratio = len(set(non_empty)) / len(non_empty) if non_empty else 0.0
    return {
        "n": len(non_empty),
        "avg_word_length": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        "unique_ratio": round(unique_ratio, 4),
    }


def reasoning_embedding_similarity(reasoning_texts: list[str], sample_size: int = 200) -> dict:
    """Average pairwise cosine similarity over a random sample of
    reasoning_texts (sentence-transformers embeddings). High average
    similarity across genuinely different queries signals collapse into
    near-identical phrasing that exact-string matching alone can miss
    (paraphrased boilerplate isn't caught by unique_ratio)."""
    import random

    from sentence_transformers import SentenceTransformer

    non_empty = [r for r in reasoning_texts if r]
    sample = random.sample(non_empty, min(sample_size, len(non_empty)))
    if len(sample) < 2:
        return {"n": len(sample), "avg_pairwise_cosine_similarity": None}

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sample, normalize_embeddings=True)
    similarity_matrix = embeddings @ embeddings.T
    n = len(sample)
    off_diagonal_sum = similarity_matrix.sum() - n  # subtract the n self-similarities (always 1.0)
    avg_similarity = off_diagonal_sum / (n * (n - 1))
    return {"n": n, "avg_pairwise_cosine_similarity": round(float(avg_similarity), 4)}
