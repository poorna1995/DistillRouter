"""Builds student training/eval datasets from teacher-labeled data.

`TeacherLabel` only stores `query_id`, not the query text — this joins it
back against the corresponding `Example` (from data/processed/) so a query
can actually be tokenized and fed to the student.
"""
from __future__ import annotations

from common.config import LABEL_TO_ID, PROCESSED_DIR, TEACHER_DIR
from common.schema import TeacherLabel, read_jsonl


def load_labeled_queries(dataset_names: list[str], split: str) -> list[dict]:
    """Returns rows pooled across every dataset in `dataset_names`, for one
    canonical split: [{"query": str, "label_id": int, "dataset": str,
    "query_id": str}, ...]. A dataset with no teacher labels for `split`
    yet is skipped rather than erroring, so this can be called before every
    registered dataset has been fully labeled.
    """
    rows = []
    for dataset_name in dataset_names:
        teacher_path = TEACHER_DIR / dataset_name / f"{split}.jsonl"
        if not teacher_path.exists():
            continue
        examples_by_id = {ex.id: ex for ex in read_jsonl(PROCESSED_DIR / dataset_name / f"{split}.jsonl")}
        for label in read_jsonl(teacher_path, TeacherLabel):
            example = examples_by_id.get(label.query_id)
            if example is None:
                continue
            rows.append(
                {
                    "query": example.query,
                    "label_id": LABEL_TO_ID[label.routing_label],
                    "dataset": dataset_name,
                    "query_id": label.query_id,
                }
            )
    return rows
