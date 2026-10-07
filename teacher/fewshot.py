"""Shared code for the teacher: few-shot loading, prompt text, response
parsing, self-referential guard.

What: the teacher is two-stage. How: Stage 1 extracts one-sentence
reasoning, zero-shot, route not yet known; Stage 2 predicts the route
from it, few-shot. Why: reasoning generated after the route would just
rationalize an answer already picked, not drive it. Stage 1 has no
few-shot demos because oracle data carries no ground-truth reasoning to
build them from.

routing_reasoning is one sentence, not a list — cheaper on the (tight)
token budget than JSON-array syntax, and the natural shape for the
student to imitate autoregressively.
"""
from __future__ import annotations

import json
import re

from common.config import LABELS, ORACLE_DIR, PROCESSED_DIR
from common.schema import Example, OracleLabel, read_jsonl, read_oracle_labels

FEWSHOT_PER_TIER = 2  # demonstration examples per tier, pooled across every oracle-labeled dataset
FEWSHOT_SOURCE_SPLIT = "calibration"  # split load_fewshot_examples() draws demonstrations from


def load_fewshot_examples() -> list[tuple[Example, OracleLabel]]:
    """Up to FEWSHOT_PER_TIER demonstrations per label, pooled across every
    oracle-labeled dataset. Unsolvable questions (no oracle label) are skipped."""
    pool: dict[str, list[tuple[Example, OracleLabel]]] = {tier: [] for tier in LABELS}
    for dataset_dir in sorted(p for p in ORACLE_DIR.glob("*") if p.is_dir()):
        labels_path = dataset_dir / f"{FEWSHOT_SOURCE_SPLIT}.labels.jsonl"
        if not labels_path.exists():
            continue
        dataset_name = dataset_dir.name
        examples_path = PROCESSED_DIR / dataset_name / f"{FEWSHOT_SOURCE_SPLIT}.jsonl"
        examples_by_id = {ex.id: ex for ex in read_jsonl(examples_path)}
        for label in read_oracle_labels(labels_path):
            example = examples_by_id.get(label.query_id)
            if example is not None and label.routing_label is not None:
                pool[label.routing_label].append((example, label))

    fewshot = []
    per_tier_used = {}
    for tier in LABELS:
        selected = pool[tier][:FEWSHOT_PER_TIER]
        fewshot.extend(selected)
        per_tier_used[tier] = f"{len(selected)}/{FEWSHOT_PER_TIER}"

    short_tiers = [tier for tier in LABELS if len(pool[tier]) < FEWSHOT_PER_TIER]
    print(
        f"load_fewshot_examples(): pool sizes {({t: len(v) for t, v in pool.items()})}, "
        f"selected {per_tier_used}"
        + (f" -- SHORT on {short_tiers} (fewer than {FEWSHOT_PER_TIER} available, using all there is)" if short_tiers else "")
    )
    return fewshot


def load_fewshot_examples_or_raise(teacher_name: str) -> list[tuple[Example, OracleLabel]]:
    """load_fewshot_examples(), raising a clear error if the pool is empty."""
    fewshot = load_fewshot_examples()
    if not fewshot:
        raise RuntimeError(
            f"No oracle-labeled calibration examples found under "
            f"data/oracle/<dataset>/{FEWSHOT_SOURCE_SPLIT}.labels.jsonl. "
            f"{teacher_name}'s few-shot block needs real oracle ground truth. Run `python run.py "
            f"oracle-label --dataset <name> --split {FEWSHOT_SOURCE_SPLIT}` first."
        )
    return fewshot


def format_fewshot_block(fewshot: list[tuple[Example, OracleLabel]]) -> str:
    """Render fewshot as the "Problem: ... {"teacher_route": ...}"
    label-only demonstration block, used by the teacher's route stage."""
    blocks = []
    for example, label in fewshot:
        target = {"teacher_route": label.routing_label}
        blocks.append(f"Problem: {example.query}\n{json.dumps(target)}")
    return "\n\n".join(blocks)


class SelfReferentialFewshotError(Exception):
    """The example about to be predicted is itself one of the teacher's
    own few-shot demonstrations."""


def check_not_self_referential(example_id: str, fewshot_ids: set[str], allow: bool = False) -> None:
    """Raise SelfReferentialFewshotError if example_id is one of
    fewshot_ids, unless allow=True (set from
    TeacherModel.allow_self_referential)."""
    if allow or example_id not in fewshot_ids:
        return
    raise SelfReferentialFewshotError(
        f"Refusing to predict on {example_id!r}: it is one of this teacher's own few-shot "
        f"demonstrations, so predicting on it would be self-referential."
    )


_ROLE_INTRO = "You are the teacher router for DistillRouter.\n\n"

LABEL_ONLY_SYSTEM_PREAMBLE = (
    _ROLE_INTRO +
    "Your task is to decide whether a small model is enough to answer the given query "
    "correctly, or whether a large model is needed. Do NOT solve the query.\n\n"

    "Available models:\n"
    "- small: a ~1-2B parameter model; reliable on short, direct, single- or few-step problems.\n"
    "- large: a ~14B parameter model; needed for long reasoning chains, advanced or symbolic "
    "reasoning, and multiple interacting constraints.\n\n"

    "If reasoning requirements for the query are given below, base your decision on them alone "
    "rather than re-deriving your own. Choose small whenever the reasoning requirements "
    "justify it as sufficient to answer the query correctly.\n\n"

    "Return JSON only:\n"
    "{\"teacher_route\": \"small|large\"}"
)


REASONING_EXTRACTOR_PREAMBLE = (
    _ROLE_INTRO +
    "Your task is to characterize, in one concise sentence, what KIND of reasoning this query "
    "demands — not how you would carry that reasoning out.\n"
    "Do NOT solve the query. Do NOT lay out a plan or sequence of steps toward solving it. "
    "Do NOT decide which model tier is required — that decision is made separately, after "
    "your explanation is read.\n\n"

    "Your explanation must:\n"
    "- describe an observable property of THIS query (its structure, its dependencies, what "
    "kind of manipulation it needs)\n"
    "- explain why that property increases the reasoning capability required to answer it\n"
    "- be specific to the query (not a generic phrase like 'requires complex reasoning' or "
    "'multi-step reasoning')\n"
    "- never describe a solution plan or sequence of steps — no \"first ... then ...\", no "
    "listing what to compute or in what order, even without stating any numbers\n"
    "- not solve the problem or compute any intermediate or final result\n"
    "- not mention model names, model tiers, or a routing decision\n"
    "- be ONE sentence, about 15-20 words — do not write more than one sentence\n\n"

    "Return JSON only:\n"
    "{\"routing_reasoning\": \"...\"}"
)


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}")  # flat (non-nested) object


def _extract_json_dict(raw_text: str) -> dict | None:
    """Tries every flat {...} span in raw_text, most-recent first, returning
    the first one that parses as valid JSON. A single greedy first-{-to-
    last-} match (the previous approach) captures garbage and fails outright
    if the model's completion has any stray brace before/after the real
    JSON object -- this instead tolerates that the same way common/
    scoring.py's extract_json_answer already does for candidate answers."""
    for candidate_text in reversed(_JSON_OBJECT_RE.findall(raw_text)):
        try:
            parsed = json.loads(candidate_text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_label_only_response(raw_text: str) -> str:
    """Extract teacher_route from JSON in raw_text; falls back to "large"
    (same conservative default oracle labeling uses) if unparseable."""
    parsed = _extract_json_dict(raw_text)
    if parsed:
        label = parsed.get("teacher_route")
        if label in LABELS:
            return label
    return "large"


def parse_reasoning_only_response(raw_text: str) -> str:
    """Extract routing_reasoning (one sentence) from JSON in raw_text.
    Requested shape == stored shape, so nothing is lost in transit."""
    parsed = _extract_json_dict(raw_text)
    if parsed:
        reasoning = _coerce_reasoning_text(parsed.get("routing_reasoning"))
        if reasoning:
            return reasoning
    return "fallback: response was not parseable JSON"


def _coerce_reasoning_text(reasoning: object) -> str:
    """Normally a plain string. Tolerates a list too, in case the model
    ignores the single-sentence instruction and emits items anyway."""
    if isinstance(reasoning, str):
        return reasoning.strip()
    if isinstance(reasoning, list):
        return " ".join(str(item).strip() for item in reasoning if isinstance(item, str) and item.strip())
    return ""
