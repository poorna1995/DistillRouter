"""GSM8K: grade-school maths word problems (Cobbe et al., 2021).

Source: Hugging Face `openai/gsm8k` (config `main`): train 7,473 / test 1,319.
The reference answer is the number after "####" in the solution.
"""
from __future__ import annotations

from datasets import load_dataset

from common.schema import Example
from dataset.base import BaseDatasetLoader, register


def final_answer(solution: str) -> str:
    return solution.split("####")[-1].strip().replace(",", "")


@register
class GSM8KLoader(BaseDatasetLoader):
    name = "gsm8k"

    def load_native_splits(self) -> dict[str, list[Example]]:
        return {split: self._load(split) for split in ("train", "test")}

    def _load(self, split: str) -> list[Example]:
        rows = self.cached_rows(f"{split}.jsonl", lambda: load_dataset("openai/gsm8k", "main", split=split))
        return [
            Example(
                id=f"gsm8k-{split}-{i}", dataset=self.name, split=split,
                query=row["question"], reference_answer=final_answer(row["answer"]),
            )
            for i, row in enumerate(rows)
        ]
