"""Generic incremental JSONL cache, keyed by a caller-supplied key
function. New records are appended to disk immediately."""
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
        """Store a record; append to disk if it's new."""
        key = self.key_fn(record)
        already_persisted = key in self._records_by_key
        self._records_by_key[key] = record
        if not already_persisted:
            self._append_to_disk(record)

    def _append_to_disk(self, record: T) -> None:
        if not self.path.exists():
            write_jsonl(self.path, [record])
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._records_by_key)
