
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Type, TypeVar


@dataclass
class Example:
    id: str                              # globally unique: "<dataset>-<split>-<index>"
    dataset: str                         # e.g. "gsm8k", "math"
    domain: str                          # e.g. "math", "knowledge", "code"
    split: str                           # "train" | "validation" | "test"
    query: str                           # the input text a router/model would see
    reference_answer: str                # ground-truth final answer, for correctness scoring
    solution: Optional[str] = None       # full reference solution / chain-of-thought, if available
    difficulty: Optional[str] = None     # dataset-native difficulty tag, if any (e.g. MATH's level "5")
    metadata: dict[str, Any] = field(default_factory=dict)  # domain-specific extras (subject, hop count, ...)


@dataclass
class TeacherLabel:
    """One teacher prediction for one Example. See teacher/base.py.

    `query_id` joins back to `Example.id`. `prompt_version` is the cache key
    alongside `query_id` — bumping it (e.g. after editing a teacher's
    prompt) invalidates only the labels produced under the old version, so
    a rerun re-labels the minimum necessary instead of everything.
    """

    query_id: str
    dataset: str
    teacher_name: str                    # registry name of the TeacherModel that produced this label
    prompt_version: str
    routing_label: str                   # one of common.config.ROUTING_LABELS
    probabilities: dict[str, float]      # soft distribution over ROUTING_LABELS, the distillation target
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    inference_cost: float


@dataclass
class OracleAttempt:
    """One candidate tier's raw attempt at one query. See oracle/labeler.py.

    This is the expensive, cached unit of oracle labeling — a real model
    call. Correctness is deliberately NOT stored here; it's derived from
    `answer_text` via `common.scoring` at label-assembly time, so fixing a
    scoring bug later never requires re-running the candidate model, only
    re-reading this file.
    """

    query_id: str
    dataset: str
    tier: str                            # one of common.config.ROUTING_LABELS
    model_id: str                        # cache-invalidates automatically if the candidate roster changes
    prompt_version: str                  # cache-invalidates automatically if CandidateModel's instruction changes
    answer_text: str                     # raw generation, unparsed
    latency_seconds: float
    input_tokens: int
    output_tokens: int


@dataclass
class OracleLabel:
    """Final oracle routing label for one query — the cheapest tier whose
    OracleAttempt scored correct, or "large" if none did (see
    oracle/labeler.py). Cheap to recompute from cached OracleAttempts, so
    this file is fully rewritten on every run rather than incrementally
    cached like OracleAttempt is.

    `succeeded` disambiguates the two ways `routing_label` can end up
    "large": genuinely the cheapest tier that got it right, vs. every tier
    failing and "large" being used as the best-tried fallback (see
    oracle/labeler.py). Without this field the two are indistinguishable
    from the record alone — which matters anywhere "large" examples get
    used as ground truth (e.g. teacher/qwen_teacher.py's few-shot
    selection), since a fallback-defaulted example teaches "nothing
    works here", not "large is the right tier".
    """

    query_id: str
    dataset: str
    routing_label: str
    succeeded: bool
    attempted_tiers: list[str]           # tiers actually executed, in cost order, until success or exhaustion


T = TypeVar("T")


def write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_jsonl(path: Path, record_type: Type[T] = Example) -> list[T]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(record_type(**json.loads(line)))
    return records
