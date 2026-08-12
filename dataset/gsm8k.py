"""GSM8K loader — grade-school arithmetic word problems (Cobbe et al., 2021).
Source: huggingface.co/datasets/openai/gsm8k. Native splits: train (7,473)
/ test (1,319); no validation split, carved from train by base.py.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from datasets import load_dataset

from dataset.base import BaseDatasetLoader, register
from common.schema import Example
from common.scoring import extract_candidate_answer, register_scorer

_ANSWER_DELIM_RE = re.compile(r"####\s*(.+)")


def _extract_final_answer(solution: str) -> str:
    """GSM8K reference solutions end with a line '#### <final answer>'."""
    match = _ANSWER_DELIM_RE.search(solution)
    return match.group(1).strip() if match else solution.strip()


@register_scorer("gsm8k")
def _score(candidate_answer_text: str, reference_answer: str) -> bool:
    """Exact-match on the final numeric answer."""
    extracted = extract_candidate_answer(candidate_answer_text)
    if extracted is None:
        return False
    candidate_number = _normalize_number(extracted)
    reference_number = _normalize_number(reference_answer)
    return candidate_number is not None and candidate_number == reference_number


def _normalize_number(text: str) -> Optional[str]:
    cleaned = text.strip().replace(",", "").replace("$", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return str(int(value)) if value == int(value) else str(value)


@register
class GSM8KLoader(BaseDatasetLoader):
    name = "gsm8k"
    domain = "math"
    native_splits = ("train", "test")

    def download(self) -> None:
        for split in self.native_splits:
            raw_path = self.raw_dir / f"{split}.jsonl"
            if raw_path.exists():
                continue
            ds = load_dataset("openai/gsm8k", "main", split=split)
            with raw_path.open("w", encoding="utf-8") as f:
                for row in ds:
                    f.write(json.dumps(row) + "\n")
            print(f"[gsm8k] downloaded {len(ds)} '{split}' examples -> {raw_path}")

    def load_native_split(self, split: str) -> list[Example]:
        raw_path = self.raw_dir / f"{split}.jsonl"
        examples = []
        with raw_path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                examples.append(
                    Example(
                        id=f"gsm8k-{split}-{i}",
                        dataset=self.name,
                        domain=self.domain,
                        split=split,
                        query=row["question"],
                        reference_answer=_extract_final_answer(row["answer"]),
                        solution=row["answer"],
                        difficulty=None,  # GSM8K carries no explicit difficulty label
                        metadata={},
                    )
                )
        return examples
