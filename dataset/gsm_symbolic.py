"""GSM-Symbolic (Mirzadeh et al., 2024): GSM8K test questions rewritten with new
names and numbers, to check results do not depend on memorised questions (E8).

Source: Hugging Face `apple/GSM-Symbolic` (config `main`): 100 templates x 50
variants. Test only, never used for training. 500 questions are sampled.
"""
from __future__ import annotations

from datasets import load_dataset

from common.schema import Example
from dataset.base import BaseDatasetLoader, register
from dataset.gsm8k import final_answer


@register
class GSMSymbolicLoader(BaseDatasetLoader):
    name = "gsm_symbolic"
    test_only = True

    def load_native_splits(self) -> dict[str, list[Example]]:
        rows = self.cached_rows("test.jsonl", lambda: load_dataset("apple/GSM-Symbolic", "main", split="test"))
        test = [
            Example(
                id=f"gsm_symbolic-{row['id']}-{row['instance']}", dataset=self.name, split="test",
                query=row["question"], reference_answer=final_answer(row["answer"]),
                metadata={"subject": f"template-{row['id']}", "original_gsm8k_test_id": row["original_id"]},
            )
            for row in rows
        ]
        return {"test": test}
