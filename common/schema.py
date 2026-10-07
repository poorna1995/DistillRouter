
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Type, TypeVar


@dataclass
class Example:
    """One question, in the same format for every dataset."""

    id: str                              # e.g. "gsm8k-train-12", "math500-3"
    dataset: str
    split: str                           # "train" | "validation" | "calibration" | "test"
    query: str
    reference_answer: str
    difficulty: Optional[str] = None     # MATH level "1".."5"; used for stratified sampling
    metadata: dict[str, Any] = field(default_factory=dict)  # e.g. {"subject": ...}; used for stratified sampling


@dataclass
class TeacherLabel:
    """A teacher's routing prediction for one Example. `teacher_route` is
    the teacher's guess (majority vote), not verified ground truth (see
    OracleLabel). `soft_large` is the share of the teacher's votes for
    "large"; None when the teacher voted once.
    """

    query_id: str
    dataset: str
    teacher_name: str
    prompt_version: str
    teacher_route: str                   # one of common.config.LABELS
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    inference_cost: float
    routing_reasoning: str = ""          # one coherent sentence, not itemized — see teacher/fewshot.py
    soft_large: Optional[float] = None   # share of votes for "large", e.g. 3 of 5 -> 0.6


@dataclass
class CandidateAttempt:
    """One candidate tier's raw attempt at one query. `correct` is scored
    once at attempt time (common.scoring) and cached alongside the raw
    answer, so it never needs re-deriving from answer_text later."""

    query_id: str
    dataset: str
    tier: str                            # one of common.config.LABELS
    model_id: str
    prompt_version: str
    answer_text: str
    correct: bool
    latency_seconds: float
    input_tokens: int
    output_tokens: int


@dataclass
class OracleLabel:
    """Ground-truth routing label: "small" if the small model answers
    correctly, else "large" if the large model does, else None (unsolvable:
    no routing decision is correct, so it is left out of routing accuracy).
    oracle/labeler.py runs both models on every query, so *_correct is
    always fully known."""

    query_id: str
    dataset: str
    routing_label: Optional[str]         # None = unsolvable
    succeeded: bool                      # False = unsolvable
    small_correct: bool
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
