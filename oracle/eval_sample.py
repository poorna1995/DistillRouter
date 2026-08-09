"""Builds/grows a stratified query sample, pooled across validation+test and
both datasets, for oracle-labeling as a cost-bounded alternative to oracle-
labeling the full 2,496-query validation+test set (~11 hours real GPU
time — see conversation history for the measured estimate).

Written deterministically (reuses dataset/base.py's stratified-sampling
helper — same algorithm already used for train/test/calibration capping),
so `oracle-label --split oracle_eval` is reproducible. Allocated
proportionally to each source's size, so gsm8k/math and validation/test
each contribute roughly their fair share.

Growable: `grow_eval_sample` excludes queries already in the sample before
drawing new ones, so calling it repeatedly (as more oracle-labeling budget
becomes available) only ever adds new queries — nothing already
oracle-labeled is resampled, duplicated, or wasted.

Usage: python -m oracle.eval_sample [additional_total]   (default 250)
"""
from __future__ import annotations

from dataclasses import replace

from common.config import DEFAULT_SEED, PROCESSED_DIR
from common.schema import read_jsonl, write_jsonl
from dataset.base import _stratified_sample  # same stratified-sampling algorithm as train/test/calibration capping

SOURCES = (
    ("gsm8k", "validation"),
    ("gsm8k", "test"),
    ("math", "validation"),
    ("math", "test"),
)


def grow_eval_sample(additional_total: int) -> None:
    existing_examples = {}
    existing_ids = {}
    for dataset_name in ("gsm8k", "math"):
        out_path = PROCESSED_DIR / dataset_name / "oracle_eval.jsonl"
        current = read_jsonl(out_path) if out_path.exists() else []
        existing_examples[dataset_name] = current
        existing_ids[dataset_name] = {ex.id for ex in current}

    pools = {}
    for dataset_name, split in SOURCES:
        examples = read_jsonl(PROCESSED_DIR / dataset_name / f"{split}.jsonl")
        pools[(dataset_name, split)] = [ex for ex in examples if ex.id not in existing_ids[dataset_name]]
    total_remaining = sum(len(examples) for examples in pools.values())

    new_by_dataset: dict[str, list] = {"gsm8k": [], "math": []}
    for (dataset_name, split), examples in pools.items():
        target_size = min(round(len(examples) / total_remaining * additional_total), len(examples))
        sampled, _remaining, _group_allocations = _stratified_sample(examples, target_size, DEFAULT_SEED)
        # split relabeled to reflect where these rows actually end up (see
        # dataset/base.py's Example.split correctness fix — same reasoning);
        # id is left untouched, it's the stable cache key.
        sampled = [replace(ex, split="oracle_eval") for ex in sampled]
        new_by_dataset[dataset_name].extend(sampled)
        print(f"{dataset_name}/{split}: sampled {len(sampled)} new (from {len(examples)} not yet in sample)")

    for dataset_name, new_examples in new_by_dataset.items():
        out_path = PROCESSED_DIR / dataset_name / "oracle_eval.jsonl"
        combined = existing_examples[dataset_name] + new_examples
        write_jsonl(out_path, combined)
        print(
            f"{dataset_name}: {len(existing_examples[dataset_name])} existing + {len(new_examples)} new "
            f"= {len(combined)} total -> {out_path}"
        )


if __name__ == "__main__":
    import sys

    additional = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    grow_eval_sample(additional)
