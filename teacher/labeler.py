"""Labels a processed dataset split with a teacher: cache-check -> dedupe
identical query text -> predict -> cache. Teacher is instantiated lazily,
only once a real prediction is needed.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

from common.config import PROCESSED_DIR, TEACHER_DIR
from common.schema import TeacherLabel, read_jsonl
from teacher.base import get_teacher_class
from teacher.cache import TeacherLabelCache


def label_dataset(
    dataset_name: str,
    teacher_name: str,
    split: str = "train",
    limit: Optional[int] = None,
    allow_self_referential_calibration: bool = False,
) -> dict[str, int]:
    """`limit` caps how many examples are processed.

    Refuses by default to label `teacher_cls.fewshot_source_split` (self-
    referential: a query could be its own few-shot demonstration). Set
    `allow_self_referential_calibration=True` to bypass, for debugging.
    """
    teacher_cls = get_teacher_class(teacher_name)
    if (
        teacher_cls.fewshot_source_split
        and split == teacher_cls.fewshot_source_split
        and not allow_self_referential_calibration
    ):
        raise ValueError(
            f"Refusing to label '{dataset_name}/{split}' with teacher '{teacher_name}': its few-shot "
            f"demonstrations come from the '{teacher_cls.fewshot_source_split}' split, so predictions "
            f"there would be self-referential. Label 'train'/'validation' instead, or pass "
            f"allow_self_referential_calibration=True (--allow-self-referential-calibration) to override."
        )

    processed_path = PROCESSED_DIR / dataset_name / f"{split}.jsonl"
    examples = read_jsonl(processed_path)
    if limit is not None:
        examples = examples[:limit]

    prompt_version = teacher_cls.prompt_version
    output_dir = TEACHER_DIR / dataset_name
    if teacher_cls.output_version:
        output_dir = output_dir / teacher_cls.output_version
    output_path = output_dir / f"{split}.jsonl"
    cache = TeacherLabelCache(output_path)
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
            print(f"{progress} -> {cached_label.teacher_route} (cached)", flush=True)
            continue

        duplicate_label = labels_by_query_text.get(example.query)
        if duplicate_label is not None:
            cache.add(replace(duplicate_label, query_id=example.id))
            deduped += 1
            print(f"{progress} -> {duplicate_label.teacher_route} (deduped)", flush=True)
            continue

        if teacher is None:
            teacher = teacher_cls()
            teacher.allow_self_referential = allow_self_referential_calibration  # thread the escape hatch through
        new_label = teacher.predict(example)
        cache.add(new_label)
        labels_by_query_text[example.query] = new_label
        newly_labeled += 1
        print(f"{progress} -> {new_label.teacher_route} ({new_label.latency_seconds:.2f}s)", flush=True)

    return {
        "cache_hits": cache_hits,
        "deduped": deduped,
        "newly_labeled": newly_labeled,
        "total_cached": len(cache),
        "output_path": str(output_path),
    }
