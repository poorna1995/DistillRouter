"""Orchestrates teacher labeling over a processed dataset split.

For each Example: skip if already cached under the teacher's current
prompt_version; otherwise skip the teacher call too if an identical query
string was already labeled earlier in this run (reusing its label under
the new query_id); otherwise call the teacher and cache the result.

The teacher itself is instantiated lazily, on the first example that
actually needs a real prediction — not upfront — so a run where every
example is already cached (or every model-loading teacher, like
QwenTeacher) never pays the load cost at all. Reading `prompt_version` for
the cache-key check therefore comes from the *class*, not an instance —
every TeacherModel subclass sets it as a class attribute (see
teacher/base.py, teacher/heuristic.py, teacher/qwen_teacher.py) precisely
so it's readable without instantiating.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

from common.config import PROCESSED_DIR, TEACHER_DIR
from common.schema import TeacherLabel, read_jsonl
from teacher.base import get_teacher_class
from teacher.cache import TeacherLabelCache


def label_dataset(
    dataset_name: str, teacher_name: str, split: str = "train", limit: Optional[int] = None
) -> dict[str, int]:
    """`limit` caps how many examples are processed — for trying a real
    teacher (real cost per call) on a handful of queries before committing
    to a full split.
    """
    processed_path = PROCESSED_DIR / dataset_name / f"{split}.jsonl"
    examples = read_jsonl(processed_path)
    if limit is not None:
        examples = examples[:limit]

    teacher_cls = get_teacher_class(teacher_name)
    prompt_version = teacher_cls.prompt_version
    cache = TeacherLabelCache(TEACHER_DIR / dataset_name / f"{split}.jsonl")
    teacher = None  # instantiated lazily below, only once a real prediction is actually needed

    cache_hits = 0
    deduped = 0
    newly_labeled = 0
    labels_by_query_text: dict[str, TeacherLabel] = {}
    total = len(examples)

    for i, example in enumerate(examples, 1):
        progress = f"[{dataset_name}/{split}] {i}/{total} {example.id}"

        cached_label = cache.get(example.id, prompt_version)
        if cached_label is not None:
            cache_hits += 1
            labels_by_query_text.setdefault(example.query, cached_label)
            print(f"{progress} -> {cached_label.routing_label} (cached)", flush=True)
            continue

        duplicate_label = labels_by_query_text.get(example.query)
        if duplicate_label is not None:
            cache.add(replace(duplicate_label, query_id=example.id))
            deduped += 1
            print(f"{progress} -> {duplicate_label.routing_label} (deduped)", flush=True)
            continue

        if teacher is None:
            teacher = teacher_cls()
        new_label = teacher.predict(example)
        cache.add(new_label)
        labels_by_query_text[example.query] = new_label
        newly_labeled += 1
        print(f"{progress} -> {new_label.routing_label} ({new_label.latency_seconds:.2f}s)", flush=True)

    return {
        "cache_hits": cache_hits,
        "deduped": deduped,
        "newly_labeled": newly_labeled,
        "total_cached": len(cache),
    }
