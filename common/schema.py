
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


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


def write_jsonl(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Example]:
    examples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(Example(**json.loads(line)))
    return examples
