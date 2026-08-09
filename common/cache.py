"""Generic incremental JSONL cache.

New records are appended to disk immediately, keyed by a caller-supplied
key function, so a rerun only recomputes whatever isn't already cached.
`teacher/cache.py`'s `TeacherLabelCache` and `oracle/cache.py`'s
`OracleAttemptCache` are both thin, one-line specializations of this —
same caching mechanism, different record type and key.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Generic, Type, TypeVar

from common.schema import read_jsonl, write_jsonl

T = TypeVar("T")


class JsonlRecordCache(Generic[T]):
    def __init__(self, path: Path, record_type: Type[T], key_fn: Callable[[T], tuple]) -> None:
        self.path = path
        self.record_type = record_type
        self.key_fn = key_fn
        self._records_by_key: dict[tuple, T] = {}
        if path.exists():
            for record in read_jsonl(path, record_type):
                self._records_by_key[key_fn(record)] = record

    def get(self, key: tuple) -> T | None:
        return self._records_by_key.get(key)

    def add(self, record: T) -> None:
        """Record a new value and append it to disk immediately."""
        key = self.key_fn(record)
        already_persisted = key in self._records_by_key
        self._records_by_key[key] = record
        if not already_persisted:
            self._append_to_disk(record)

    def _append_to_disk(self, record: T) -> None:
        # write_jsonl truncates, so on first write we recreate the file
        # (empty cache -> just this record); afterwards we append a line.
        if not self.path.exists():
            write_jsonl(self.path, [record])
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._records_by_key)
