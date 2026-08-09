"""MATH loader — competition mathematics problems (Hendrycks et al., 2021).

Source: https://huggingface.co/datasets/EleutherAI/hendrycks_math — the
canonical per-subject mirror that reproduces the original paper's 7,500
train / 5,000 test split.

Note: the mirror originally referenced during design (`nlile/hendrycks-
MATH-benchmark`) turned out, on inspection, to have been re-split by its
uploader into 12,000/500 and does NOT match the literature split sizes
cited in this project's design spec. `EleutherAI/hendrycks_math` was
verified at implementation time to reproduce 7,500/5,000 exactly, summed
across its 7 subject-specific configs, and is used instead.

Native splits: train (7,500) / test (5,000). No validation split is
shipped upstream, so `BaseDatasetLoader` carves one out of train (see
base.py).

Note on this file's name: it shadows the stdlib `math` module by leaf name.
This is safe as-is because nothing on sys.path ever points *inside*
`dataset/` (only the project root is added — see run.py), so `import math`
anywhere else in the project still resolves to the stdlib; this module is
only reachable as the fully-qualified `dataset.math`. Do not add `dataset/`
itself to sys.path, or add a bare `import math` inside this package,
without re-checking that assumption.
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
    """Exact-match on the final answer extracted via
    `common.scoring.extract_candidate_answer` (distinct from
    `extract_boxed_answer` below, which parses MATH's own *reference*
    solutions, not candidate output), with light string normalization
    (whitespace, spacing LaTeX commands, outer $/.).

    Known gap: this does not attempt deeper symbolic equivalence (e.g.
    "0.5" vs "\\frac{1}{2}", or algebraically-equal-but-differently-
    written expressions) — a candidate that reaches an equivalent but
    differently-formatted answer will be scored incorrect. Acceptable for
    a first pass; flagged rather than silently assumed solved.
    """
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
