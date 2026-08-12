
from __future__ import annotations

from pathlib import Path

from common.cache import JsonlRecordCache
from common.schema import CandidateAttempt


class CandidateAttemptCache(JsonlRecordCache[CandidateAttempt]):
    """Keyed by (query_id, tier, model_id, prompt_version)."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            path,
            CandidateAttempt,
            key_fn=lambda attempt: (attempt.query_id, attempt.tier, attempt.model_id, attempt.prompt_version),
        )

    def get(self, query_id: str, tier: str, model_id: str, prompt_version: str) -> CandidateAttempt | None:  # type: ignore[override]
        return super().get((query_id, tier, model_id, prompt_version))
