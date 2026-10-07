"""MATH: competition mathematics (Hendrycks et al., 2021).

Train: Hugging Face `EleutherAI/hendrycks_math` (7 subjects, 7,500 problems);
       the reference answer is the last \\boxed{} in the solution.
Test:  MATH-500 (`HuggingFaceH4/MATH-500`), a fixed set of 500 MATH test problems.
Any training problem that also appears in MATH-500 is removed (leakage check).

Note: this module's name shadows the stdlib `math` by leaf name; safe because
only the project root is on sys.path.
"""
from __future__ import annotations

from datasets import load_dataset

from common.schema import Example
from common.scoring import extract_boxed_answer
from dataset.base import BaseDatasetLoader, normalize_question, register

SUBJECTS = (
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
)


def _subject_key(name: str) -> str:
    return name.strip().lower().replace(" & ", "_and_").replace(" ", "_")


@register
class MATHLoader(BaseDatasetLoader):
    name = "math"

    def load_native_splits(self) -> dict[str, list[Example]]:
        test = self._load_math500()
        test_questions = {normalize_question(ex.query) for ex in test}
        train = [ex for ex in self._load_train() if normalize_question(ex.query) not in test_questions]
        print(f"[math] removed {self.n_train_raw - len(train)} training problems that are in MATH-500")
        return {"train": train, "test": test}

    def _load_train(self) -> list[Example]:
        def fetch():
            for subject in SUBJECTS:
                for row in load_dataset("EleutherAI/hendrycks_math", subject, split="train"):
                    yield dict(row, subject=subject)

        rows = self.cached_rows("train.jsonl", fetch)
        examples = []
        for i, row in enumerate(rows):
            answer = extract_boxed_answer(row["solution"])
            if answer is None:
                continue  # no \boxed{} answer in the solution
            examples.append(Example(
                id=f"math-train-{i}", dataset=self.name, split="train",
                query=row["problem"], reference_answer=answer,
                difficulty=row["level"].replace("Level ", ""), metadata={"subject": row["subject"]},
            ))
        self.n_train_raw = len(examples)
        return examples

    def _load_math500(self) -> list[Example]:
        rows = self.cached_rows("math500.jsonl", lambda: load_dataset("HuggingFaceH4/MATH-500", split="test"))
        return [
            Example(
                id=f"math500-{i}", dataset=self.name, split="test",
                query=row["problem"], reference_answer=row["answer"],
                difficulty=str(row["level"]),
                metadata={"subject": _subject_key(row["subject"]), "unique_id": row["unique_id"]},
            )
            for i, row in enumerate(rows)
        ]
