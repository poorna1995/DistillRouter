"""Assembles the SFT distillation dataset from teacher output: one record
per labeled example, {query, teacher_route, routing_reasoning}. No
answers/solutions/oracle labels — the teacher's own output is the entire
supervision signal, matching what the student sees at inference (query
text only). to_sft_record()/build_sft_records() then convert that into
the {"messages": [...]} chat format TRL/Axolotl/Unsloth read directly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from common.config import PROCESSED_DIR, STUDENT_DIR, TEACHER_DIR
from common.schema import Example, TeacherLabel, read_jsonl, write_jsonl


@dataclass
class DistillationExample:
    """One SFT training record: query in, teacher's route + reasoning out."""

    query: str
    teacher_route: str
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
                routing_reasoning=label.routing_reasoning,
            )
        )

    output_dir = STUDENT_DIR / dataset_name
    if output_version:
        output_dir = output_dir / output_version
    output_path = output_dir / f"{split}.jsonl"
    write_jsonl(output_path, records)

    print(
        f"[{dataset_name}/{split}] {len(records)} SFT records -> {output_path} "
        f"(skipped {skipped_version} other prompt_version, {skipped_missing} missing example)"
    )
    return records


# Fixed across every example, on purpose — the student learns tier semantics from
# thousands of paired examples during training, not from a rulebook restated each
# call like the teacher's ICL prompt needs. Keeping it constant is what lets SFT
# teach the response format/behavior consistently (see AWS Nova SFT guidance).
_STUDENT_INSTRUCTION = (
    "You are DistillRouter.\n\n"
    "What reasoning requirements are visible in this query? Then predict the minimum model "
    "capability required to answer it reliably. Do not solve the query.\n\n"
    "Query:\n{query}"
)


def to_sft_record(example: DistillationExample, include_reasoning: bool = True) -> dict:
    """DistillationExample -> {"messages": [...]}. Reasoning before route in
    the assistant turn, matching the teacher's own generation order — see
    teacher/fewshot.py's docstring for why reversing it would make the
    reasoning a post-hoc gloss on an already-decided route.

    include_reasoning=False drops the <REASON> block entirely (route-only
    ablation) — same instruction and query, so this is the one-variable
    comparison against the reasoning+route baseline: does the reasoning
    supervision actually earn its keep, or does label-only do just as well?
    """
    if include_reasoning:
        assistant_content = f"<REASON>\n{example.routing_reasoning}\n\n<ROUTE>\n{example.teacher_route}"
    else:
        assistant_content = f"<ROUTE>\n{example.teacher_route}"
    return {
        "messages": [
            {"role": "user", "content": _STUDENT_INSTRUCTION.format(query=example.query)},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def build_sft_records(
    dataset_name: str, prompt_version: str, output_version: str = "", split: str = "train",
    include_reasoning: bool = True,
) -> list[dict]:
    """build_distillation_dataset() + to_sft_record(), written to
    data/student/<dataset>/[<output_version>/]<split>.sft[-route-only].jsonl
    — the file an SFT trainer reads directly. include_reasoning=False
    writes the route-only ablation to its own file, alongside (not
    overwriting) the reasoning+route baseline."""
    examples = build_distillation_dataset(dataset_name, prompt_version, output_version=output_version, split=split)
    records = [to_sft_record(ex, include_reasoning=include_reasoning) for ex in examples]

    output_dir = STUDENT_DIR / dataset_name
    if output_version:
        output_dir = output_dir / output_version
    suffix = "sft" if include_reasoning else "sft-route-only"
    output_path = output_dir / f"{split}.{suffix}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[{dataset_name}/{split}] {len(records)} SFT-format records ({suffix}) -> {output_path}")
    return records
