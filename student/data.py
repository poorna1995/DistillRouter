"""Assembles the distillation dataset from teacher output: one record
per labeled example, {query, teacher_route, soft_large, routing_reasoning}. No
answers/solutions/oracle labels — the teacher's own output is the entire
supervision signal, matching what the student sees at inference (query
text only). Read directly by both student variants (student/
train_classifier.py, student/train_dual.py); each decides for itself
which fields it needs -- the classifier ignores routing_reasoning, the
dual-head uses it as an auxiliary training signal.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

from common.config import PROCESSED_DIR, STUDENT_DIR, TEACHER_DIR
from common.schema import Example, TeacherLabel, read_jsonl, write_jsonl


@dataclass
class DistillationExample:
    """One training record: query in, teacher's route + reasoning out."""

    query: str
    teacher_route: str
    soft_large: float          # share of teacher votes for "large" (0.0 or 1.0 for a single vote)
    routing_reasoning: str


def build_distillation_dataset(
    dataset_name: str, prompt_version: str, output_version: str = "", split: str = "train"
) -> list[DistillationExample]:
    """Join processed query text with teacher output by query_id/id.
    Only rows under `prompt_version` are used — a teacher's output file
    can span multiple prompt versions over time (same issue hit with
    candidate attempts in oracle/), so this pins to exactly one, same
    discipline TeacherLabelCache already applies. Writes
    data/student/<dataset>/[<output_version>/]<split>.jsonl.
    """
    examples_by_id = {
        ex.id: ex for ex in read_jsonl(PROCESSED_DIR / dataset_name / f"{split}.jsonl", Example)
    }

    teacher_dir = TEACHER_DIR / dataset_name
    if output_version:
        teacher_dir = teacher_dir / output_version
    labels = read_jsonl(teacher_dir / f"{split}.jsonl", TeacherLabel)

    records = []
    skipped_version = 0
    skipped_missing = 0
    for label in labels:
        if label.prompt_version != prompt_version:
            skipped_version += 1
            continue
        example = examples_by_id.get(label.query_id)
        if example is None:
            skipped_missing += 1
            continue
        records.append(
            DistillationExample(
                query=example.query,
                teacher_route=label.teacher_route,
                soft_large=(label.soft_large if label.soft_large is not None
                            else float(label.teacher_route == "large")),
                routing_reasoning=label.routing_reasoning,
            )
        )

    output_dir = STUDENT_DIR / dataset_name
    if output_version:
        output_dir = output_dir / output_version
    output_path = output_dir / f"{split}.jsonl"
    write_jsonl(output_path, records)

    print(
        f"[{dataset_name}/{split}] {len(records)} distillation records -> {output_path} "
        f"(skipped {skipped_version} other prompt_version, {skipped_missing} missing example)"
    )
    return records


def oversample_minority_tiers(
    examples: list[DistillationExample], target_ratio: float = 1.0
) -> list[DistillationExample]:
    """Repeats minority-tier examples toward the majority tier's count.
    target_ratio=1.0 -> full balance, 0.5 -> halfway there. A tier
    already at or above the target is left untouched (this is pure
    oversampling -- it only ever adds duplicates, never removes real
    examples from a majority tier). Whole-example duplication -- the
    Trainer's RandomSampler still shuffles every copy independently,
    and each duplicate gets its own dropout mask during training, so
    it's not a bit-identical forward pass every time it recurs.

    Aggressive ratios duplicate a small unique pool heavily: at
    target_ratio=1.0 on this project's actual pooled train data (small
    tier ~154 unique queries vs. large's ~1,160), small gets repeated
    ~7.5x each -- real overfitting risk on that specific 154-query text,
    not just a reweighting. Start lower (e.g. 0.5) unless you have a
    specific reason to want full balance.
    """
    by_tier = collections.defaultdict(list)
    for ex in examples:
        by_tier[ex.teacher_route].append(ex)
    before_counts = {tier: len(v) for tier, v in by_tier.items()}
    max_count = max(before_counts.values())
    target = int(max_count * target_ratio)

    oversampled = []
    after_counts = {}
    for tier, tier_examples in by_tier.items():
        if target <= len(tier_examples):
            oversampled.extend(tier_examples)  # already at/above target -- leave untouched
            after_counts[tier] = len(tier_examples)
        else:
            reps, remainder = divmod(target, len(tier_examples))
            oversampled.extend(tier_examples * reps + tier_examples[:remainder])
            after_counts[tier] = reps * len(tier_examples) + remainder

    print(
        f"oversample_minority_tiers(target_ratio={target_ratio}): {before_counts} -> {after_counts} "
        f"({len(examples)} -> {len(oversampled)} total examples)"
    )
    return oversampled


# Fixed across every example, on purpose — the student learns tier semantics from
# thousands of paired examples during training, not from a rulebook restated each
# call like the teacher's ICL prompt needs. Keeping it constant is what lets training
# teach the response format/behavior consistently (see AWS Nova SFT guidance).
_STUDENT_INSTRUCTION = (
    "You are DistillRouter.\n\n"
    "What reasoning requirements are visible in this query? Then predict the minimum model "
    "capability required to answer it reliably. Do not solve the query.\n\n"
    "Query:\n{query}"
)
