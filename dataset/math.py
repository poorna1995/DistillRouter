"""MATH loader — competition mathematics problems (Hendrycks et al., 2021).
Source: huggingface.co/datasets/EleutherAI/hendrycks_math (7 subject
configs, reproduces the 7,500 train / 5,000 test split). No validation
split, carved from train by base.py.

Note: this module's name shadows stdlib `math` by leaf name. Safe as-is
since only the project root is on sys.path, not dataset/ itself — don't
add dataset/ to sys.path without re-checking that.
"""
from __future__ import annotations

import json

from datasets import load_dataset

from dataset.base import BaseDatasetLoader, register
from common.schema import Example
from common.scoring import extract_boxed_answer, extract_candidate_answer, register_scorer

_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


@register_scorer("math")
def _score(candidate_answer_text: str, reference_answer: str) -> bool:
    """Exact-match on the extracted final answer, after light LaTeX
    normalization. Known gap: no symbolic-equivalence check (e.g. "0.5"
    vs "\\frac{1}{2}" scores incorrect)."""
    extracted = extract_candidate_answer(candidate_answer_text)
    if extracted is None:
        return False
    return _normalize(extracted) == _normalize(reference_answer)


def _normalize(text: str) -> str:
    normalized = text.strip()
    for token in (" ", "\\!", "\\,", "\\;", "\\left", "\\right"):
        normalized = normalized.replace(token, "")
    return normalized.strip("$").rstrip(".")


@register
class MATHLoader(BaseDatasetLoader):
    name = "math"
    domain = "math"
    native_splits = ("train", "test")

    def download(self) -> None:
        for split in self.native_splits:
            raw_path = self.raw_dir / f"{split}.jsonl"
            if raw_path.exists():
                continue
            rows = []
            for subject in _SUBJECTS:
                ds = load_dataset("EleutherAI/hendrycks_math", subject, split=split)
                rows.extend(dict(row, subject=subject) for row in ds)
            with raw_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            print(f"[math] downloaded {len(rows)} '{split}' examples -> {raw_path}")

    def load_native_split(self, split: str) -> list[Example]:
        raw_path = self.raw_dir / f"{split}.jsonl"
        examples = []
        skipped = 0
        with raw_path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                answer = extract_boxed_answer(row["solution"])
                if answer is None:
                    skipped += 1
                    continue
                level = row["level"].replace("Level ", "") if row.get("level") else None
                examples.append(
                    Example(
                        id=f"math-{split}-{i}",
                        dataset=self.name,
                        domain=self.domain,
                        split=split,
                        query=row["problem"],
                        reference_answer=answer,
                        solution=row["solution"],
                        difficulty=level,  # "1".."5", per Hendrycks et al.
                        metadata={"subject": row["subject"]},
                    )
                )
        if skipped:
            print(f"[math] warning: {skipped} '{split}' examples had no \\boxed{{}} answer and were skipped")
        return examples
