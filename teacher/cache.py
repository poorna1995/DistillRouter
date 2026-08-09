"""Incremental cache for teacher labels.

This is what makes `label-data` idempotent: rerunning it only pays for
queries that are new or whose `prompt_version` changed, instead of
re-calling the teacher for every query every time — mirroring how
`dataset/base.py` skips re-downloading raw data already on disk.

Layout: one cache file per dataset split, `data/teacher/<dataset>/<split>.jsonl`.
Labels are appended as they're produced (not batched at the end), so an
interrupted run loses at most the one label in flight.

A thin specialization of `common.cache.JsonlRecordCache` — see there for
the actual caching mechanism, shared with `oracle/cache.py`.
"""
from __future__ import annotations

from pathlib import Path

from common.cache import JsonlRecordCache
from common.schema import TeacherLabel


class TeacherLabelCache(JsonlRecordCache[TeacherLabel]):
    """Keyed by (query_id, prompt_version) — see TeacherModel.prompt_version."""

    def __init__(self, path: Path) -> None:
        super().__init__(path, TeacherLabel, key_fn=lambda label: (label.query_id, label.prompt_version))

    def get(self, query_id: str, prompt_version: str) -> TeacherLabel | None:  # type: ignore[override]
        return super().get((query_id, prompt_version))
