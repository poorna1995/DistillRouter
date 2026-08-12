#!/usr/bin/env python3
"""Check if the data is overlapping between train, validation, and test.

Standalone script version of `python run.py check-splits`. Reuses
common/split_integrity.py for the actual leakage detection (exact match
on example id and on exact query text), but prints a fuller report:
per-split row counts plus every offending id/query pair, instead of
just pass/fail.

Usage:
    python check_split_overlap.py                       # all registered datasets
    python check_split_overlap.py --dataset gsm8k math   # specific datasets
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common.config import PROCESSED_DIR
from common.schema import read_jsonl


def find_overlaps(dataset_name: str, processed_dir: Path = PROCESSED_DIR) -> dict:
    """Return per-split counts and any id/query overlaps across
    processed_dir/dataset_name/*.jsonl (train/validation/test/calibration).
    """
    dataset_dir = processed_dir / dataset_name
    report = {"dataset": dataset_name, "counts": {}, "id_overlaps": [], "query_overlaps": []}
    if not dataset_dir.is_dir():
        report["missing"] = True
        return report

    id_owner: dict[str, str] = {}     # id -> split that first had it
    query_owner: dict[str, str] = {}  # query text -> split that first had it

    for split_path in sorted(dataset_dir.glob("*.jsonl")):
        split = split_path.stem
        n = 0
        for example in read_jsonl(split_path):
            n += 1

            first_id_split = id_owner.setdefault(example.id, split)
            if first_id_split != split:
                report["id_overlaps"].append((example.id, first_id_split, split))

            first_query_split = query_owner.setdefault(example.query, split)
            if first_query_split != split:
                report["query_overlaps"].append((example.query, first_query_split, split))

        report["counts"][split] = n

    return report


def print_report(report: dict) -> bool:
    """Print one dataset's report; return True if it's clean."""
    name = report["dataset"]
    print(f"=== {name} ===")
    if report.get("missing"):
        print("  (no processed data found, skipping)\n")
        return True

    for split, n in report["counts"].items():
        print(f"  {split}: {n} rows")

    clean = not report["id_overlaps"] and not report["query_overlaps"]
    if clean:
        print("  OK: no id or query overlap across splits\n")
        return True

    for id_, first_split, split in report["id_overlaps"]:
        print(f"  OVERLAP id {id_!r}: appears in both {first_split!r} and {split!r}")
    for query, first_split, split in report["query_overlaps"]:
        print(
            f"  OVERLAP query in {first_split!r} and {split!r}: {query[:80]!r}"
        )
    print()
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=None,
        help="Dataset(s) to check (default: all registered datasets)",
    )
    args = parser.parse_args()

    if args.dataset:
        names = args.dataset
    else:
        from dataset import datasets

        names = datasets()

    all_clean = True
    for name in names:
        clean = print_report(find_overlaps(name))
        all_clean = all_clean and clean

    if not all_clean:
        raise SystemExit("Split leakage detected — see OVERLAP lines above.")
    print("All datasets clean: train/validation/test/calibration are disjoint.")


if __name__ == "__main__":
    main()
