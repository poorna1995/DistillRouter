"""Checks that a dataset's processed splits (train/validation/test/
calibration) share no example id or exact query text."""
from __future__ import annotations

from pathlib import Path

from common.config import PROCESSED_DIR
from common.schema import read_jsonl


class SplitLeakageError(Exception):
    """Two splits share an id or exact query text."""


def check_dataset_splits(dataset_name: str, processed_dir: Path = PROCESSED_DIR) -> None:
    """Raise SplitLeakageError if any two split files under
    processed_dir/dataset_name/*.jsonl share an id or exact query text.
    No-op if the dataset has no processed files yet. Exact-match only,
    not semantic dedup.
    """
    dataset_dir = processed_dir / dataset_name
    if not dataset_dir.is_dir():
        print(f"[{dataset_name}] no processed/ directory yet -- nothing to check, skipping")
        return

    seen_ids: dict[str, str] = {}       # id -> split that first had it
    seen_queries: dict[str, str] = {}   # query text -> split that first had it
    violations: list[str] = []
    split_counts: dict[str, int] = {}

    for split_path in sorted(dataset_dir.glob("*.jsonl")):
        split = split_path.stem
        n = 0
        for example in read_jsonl(split_path):
            n += 1
            first_id_split = seen_ids.setdefault(example.id, split)
            if first_id_split != split:
                violations.append(f"id {example.id!r} appears in both {first_id_split!r} and {split!r}")

            first_query_split = seen_queries.setdefault(example.query, split)
            if first_query_split != split:
                violations.append(
                    f"identical query text appears in both {first_query_split!r} and {split!r}: "
                    f"{example.query[:80]!r}"
                )
        split_counts[split] = n

    if violations:
        raise SplitLeakageError(
            f"Split leakage detected in '{dataset_name}' ({len(violations)} issue(s)):\n  "
            + "\n  ".join(violations)
        )

    print(
        f"[{dataset_name}] no overlap: {len(seen_ids)} unique ids, {len(seen_queries)} unique queries "
        f"across {split_counts} -- 0 shared ids, 0 shared query texts"
    )
