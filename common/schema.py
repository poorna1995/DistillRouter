
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Type, TypeVar


@dataclass
class Example:
    """One dataset query, normalized across all datasets."""

    id: str                              # "<dataset>-<split>-<index>"
    dataset: str
    domain: str
    split: str                           # "train" | "validation" | "test" | "calibration"
    query: str
    reference_answer: str
    solution: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TeacherLabel:
    """A teacher's routing prediction for one Example. `teacher_route` is
    the teacher's guess, not verified ground truth (see OracleLabel). No
    probabilities field — no teacher prompt here asks for a real
    predictive distribution.
    """

    query_id: str
    dataset: str
    teacher_name: str
    prompt_version: str
    teacher_route: str                   # one of common.config.ROUTING_LABELS
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    inference_cost: float
    routing_reasoning: str = ""          # one coherent sentence, not itemized — see teacher/fewshot.py


@dataclass
class CandidateAttempt:
    """One candidate tier's raw attempt at one query. `correct` is scored
    once at attempt time (common.scoring) and cached alongside the raw
    answer, so it never needs re-deriving from answer_text later."""

    query_id: str
    dataset: str
    tier: str                            # one of common.config.ROUTING_LABELS
    model_id: str
    prompt_version: str
    answer_text: str
    correct: bool
    latency_seconds: float
    input_tokens: int
    output_tokens: int


@dataclass
class OracleLabel:
    """Ground-truth routing label: the cheapest tier that answered
    correctly, or "large" if none did. oracle/labeler.py runs every tier
    on every query (no early-exit), so *_correct is always fully known,
    not just the winning tier's."""

    query_id: str
    dataset: str
    routing_label: str
    succeeded: bool
    small_correct: bool
    medium_correct: bool
    large_correct: bool


T = TypeVar("T")


def read_oracle_labels(path: Path) -> list[OracleLabel]:
    """read_jsonl(path, OracleLabel), named for call-site clarity."""
    return read_jsonl(path, OracleLabel)


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
