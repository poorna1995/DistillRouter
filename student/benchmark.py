"""Head-to-head comparison of the teacher (Qwen2.5-3B, generative) and
student (Gemma-3-270m, classification head) routers: latency, accuracy vs
real oracle ground truth, and compression ratio. This directly answers the
project's stated problem — "the router itself introduces significant
latency into the critical inference path."

Teacher latency comes from real recorded telemetry already on disk
(TeacherLabel.latency_seconds, logged for every one of the ~4,500 queries
already labeled) — no re-running needed. Student latency is benchmarked
fresh here, with explicit CUDA synchronization: GPU ops are asynchronous
in PyTorch, so timing without torch.cuda.synchronize() would just measure
how fast the op was *queued*, not how long it actually took to run.

Usage: python -m student.benchmark
"""
from __future__ import annotations

import statistics
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common.config import ID_TO_LABEL, ORACLE_DIR, PROCESSED_DIR, TEACHER_DIR
from common.schema import Example, OracleLabel, TeacherLabel, read_jsonl

STUDENT_PATH = "data/student/gemma-3-270m-router"
TEACHER_MODEL_ID = "qwen2.5-3b"
TEACHER_NOMINAL_PARAMS = 3_090_000_000  # Qwen2.5-3B-Instruct's published parameter count
DATASETS = ("gsm8k", "math")


def _stats(values: list[float]) -> dict:
    values = sorted(values)
    n = len(values)
    return {
        "n": n,
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "p95": round(values[int(n * 0.95)], 4) if n > 0 else None,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _teacher_latency_stats() -> dict:
    """Aggregated from real, already-recorded TeacherLabel telemetry —
    every query the teacher has ever actually labeled, not a fresh sample.
    """
    latencies = []
    for dataset_name in DATASETS:
        for split in ("train", "validation", "test", "calibration"):
            path = TEACHER_DIR / dataset_name / f"{split}.jsonl"
            if not path.exists():
                continue
            for label in read_jsonl(path, TeacherLabel):
                if label.teacher_name == TEACHER_MODEL_ID:
                    latencies.append(label.latency_seconds)
    return _stats(latencies)


def _teacher_accuracy_vs_oracle() -> tuple[float, int]:
    correct, total = 0, 0
    for dataset_name in DATASETS:
        teacher_path = TEACHER_DIR / dataset_name / "calibration.jsonl"
        if not teacher_path.exists():
            continue
        teacher_labels = {
            label.query_id: label
            for label in read_jsonl(teacher_path, TeacherLabel)
            if label.teacher_name == TEACHER_MODEL_ID
        }
        for oracle_label in read_jsonl(ORACLE_DIR / dataset_name / "calibration.labels.jsonl", OracleLabel):
            teacher_label = teacher_labels.get(oracle_label.query_id)
            if teacher_label is None:
                continue
            correct += int(teacher_label.routing_label == oracle_label.routing_label)
            total += 1
    return (correct / total if total else 0.0), total


def _load_student():
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(STUDENT_PATH).to("cuda").eval()
    return tokenizer, model


def _predict_student(tokenizer, model, query: str) -> str:
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=256).to("cuda")
    with torch.no_grad():
        pred_id = model(**inputs).logits.argmax(-1).item()
    return ID_TO_LABEL[pred_id]


def _benchmark_student_latency(tokenizer, model, examples: list[Example]) -> dict:
    # Warm-up: the first CUDA call in a session pays one-time kernel
    # compilation/dispatch cost (see conversation history) — exclude it.
    _predict_student(tokenizer, model, examples[0].query)
    torch.cuda.synchronize()

    latencies = []
    for example in examples:
        inputs = tokenizer(example.query, return_tensors="pt", truncation=True, max_length=256).to("cuda")
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.no_grad():
            model(**inputs)
        torch.cuda.synchronize()
        latencies.append(time.monotonic() - start)

    return _stats(latencies)


def _student_accuracy_vs_oracle(tokenizer, model) -> tuple[float, int]:
    correct, total = 0, 0
    for dataset_name in DATASETS:
        examples_by_id = {ex.id: ex for ex in read_jsonl(PROCESSED_DIR / dataset_name / "calibration.jsonl")}
        for oracle_label in read_jsonl(ORACLE_DIR / dataset_name / "calibration.labels.jsonl", OracleLabel):
            example = examples_by_id[oracle_label.query_id]
            predicted = _predict_student(tokenizer, model, example.query)
            correct += int(predicted == oracle_label.routing_label)
            total += 1
    return (correct / total if total else 0.0), total


def main() -> None:
    print("=== Teacher (Qwen2.5-3B, generative, boxed-format answer + JSON label) ===")
    teacher_latency = _teacher_latency_stats()
    print("latency (s):", teacher_latency)
    teacher_accuracy, teacher_n = _teacher_accuracy_vs_oracle()
    print(f"accuracy vs oracle ground truth (calibration, n={teacher_n}): {teacher_accuracy:.1%}")

    print("\nLoading student...")
    tokenizer, model = _load_student()
    student_param_count = sum(p.numel() for p in model.parameters())

    latency_sample = []
    for dataset_name in DATASETS:
        latency_sample.extend(read_jsonl(PROCESSED_DIR / dataset_name / "validation.jsonl")[:100])

    print("\n=== Student (Gemma-3-270m, single-forward-pass classification head) ===")
    student_latency = _benchmark_student_latency(tokenizer, model, latency_sample)
    print("latency (s):", student_latency)
    student_accuracy, student_n = _student_accuracy_vs_oracle(tokenizer, model)
    print(f"accuracy vs oracle ground truth (calibration, n={student_n}): {student_accuracy:.1%}")

    print("\n=== Comparison ===")
    print(f"Parameter count: teacher ~{TEACHER_NOMINAL_PARAMS:,} vs student {student_param_count:,} "
          f"({TEACHER_NOMINAL_PARAMS / student_param_count:.1f}x smaller)")
    print(f"Mean latency speedup: {teacher_latency['mean'] / student_latency['mean']:.1f}x")
    print(f"p95 latency speedup: {teacher_latency['p95'] / student_latency['p95']:.1f}x")
    print(f"Accuracy: teacher {teacher_accuracy:.1%} vs student {student_accuracy:.1%} "
          f"(both vs oracle, both n={teacher_n}/{student_n} — small sample, see module docstring)")


if __name__ == "__main__":
    main()
