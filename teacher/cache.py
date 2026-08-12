"""Incremental cache for teacher labels — makes label-data idempotent.
One file per (dataset, split), and per output_version when the teacher
sets one: data/teacher/<dataset>/[<output_version>/]<split>.jsonl — the
path itself is decided by teacher/labeler.py, not by this class."""
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
