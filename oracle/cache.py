"""Incremental cache for oracle attempts.

Mirrors teacher/cache.py exactly, same underlying mechanism
(common.cache.JsonlRecordCache), different record type and key: a rerun
only re-executes a candidate for (query, tier) pairs not already attempted
under the current candidate roster.
"""
from __future__ import annotations

from pathlib import Path

from common.cache import JsonlRecordCache
from common.schema import OracleAttempt


class OracleAttemptCache(JsonlRecordCache[OracleAttempt]):
    """Keyed by (query_id, tier, model_id, prompt_version) — model_id
    invalidates only the affected tier if the candidate roster changes;
    prompt_version invalidates everything if CandidateModel's instruction
    text changes (e.g. adding chain-of-thought), same role TeacherLabel's
    prompt_version plays for teacher caching."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            path,
            OracleAttempt,
            key_fn=lambda attempt: (attempt.query_id, attempt.tier, attempt.model_id, attempt.prompt_version),
        )

    def get(self, query_id: str, tier: str, model_id: str, prompt_version: str) -> OracleAttempt | None:  # type: ignore[override]
        return super().get((query_id, tier, model_id, prompt_version))
