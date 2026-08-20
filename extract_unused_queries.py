"""Extracts every native query that never made it into train/validation/
test/calibration -- the pool dataset/base.py's _cap_split_sizes() caps
away and silently discards (it computes `_remaining` and never writes
it). Robust set-difference on Example.id (native IDs minus every ID
already claimed by a processed split), not a replay of the sampling
RNG -- correct regardless of how sampling internals change later.

Writes data/unused/<dataset>/pool.jsonl, same Example schema as
data/processed/. Read-only w.r.t. existing data -- never touches
train/validation/test/calibration.

Usage:
    python extract_unused_queries.py                    # all registered datasets
    python extract_unused_queries.py --dataset gsm8k     # one dataset
    python extract_unused_queries.py --limit 500          # cap the written pool per dataset
"""
from __future__ import annotations

import argparse

from common.config import PROCESSED_DIR, RAW_DIR
from common.schema import Example, read_jsonl, write_jsonl
from dataset import datasets, get_loader_class

CANONICAL_SPLITS = ("train", "validation", "test", "calibration")


def extract_unused(dataset_name: str, limit: int | None = None) -> list[Example]:
    loader_cls = get_loader_class(dataset_name)
    loader = loader_cls(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR)

    native: dict[str, list[Example]] = {}
    for split in loader.native_splits:
        native[split] = loader.load_native_split(split)
    by_id = {ex.id: ex for examples in native.values() for ex in examples}

    used_ids: set[str] = set()
    for split in CANONICAL_SPLITS:
        path = PROCESSED_DIR / dataset_name / f"{split}.jsonl"
        if path.exists():
            used_ids |= {ex.id for ex in read_jsonl(path, Example)}

    unused = [ex for ex_id, ex in by_id.items() if ex_id not in used_ids]
    unused.sort(key=lambda ex: ex.id)  # deterministic order regardless of dict iteration
    if limit is not None:
        unused = unused[:limit]
    return unused


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", nargs="+", default=None, help="Dataset(s) (default: all registered).")
    parser.add_argument("--limit", type=int, default=None, help="Cap the pool size per dataset (default: no cap).")
    args = parser.parse_args()

    names = args.dataset or datasets()
    for name in names:
        unused = extract_unused(name, limit=args.limit)
        out_path = PROCESSED_DIR.parent / "unused" / name / "pool.jsonl"
        write_jsonl(out_path, unused)
        print(f"[{name}] {len(unused)} unused queries -> {out_path}")


if __name__ == "__main__":
    main()
