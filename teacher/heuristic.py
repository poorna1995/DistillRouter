"""HeuristicTeacher — a zero-cost, deterministic routing baseline.

Assigns a routing label from the dataset-native difficulty tag (e.g.
MATH's 1-5 level) when available, falling back to query length. No API
call, no cost, fully reproducible.

This exists to exercise the rest of the pipeline (caching, student
training, evaluation) end to end before a real LLM-backed teacher is
wired in. To add one: write `teacher/<name>.py` with a `TeacherModel`
subclass decorated with `@register`, add one import line to
`teacher/__init__.py`, then pass `--teacher <name>` to `label-data`.
"""
from __future__ import annotations

import time

from common.config import ROUTING_LABELS
from common.schema import Example, TeacherLabel
from teacher.base import TeacherModel, register

# Query character-length bounds used only when a dataset carries no native
# difficulty tag: below _SHORT -> "small", above _LONG -> "large", linear
# in between.
_SHORT_QUERY_CHARS = 80
_LONG_QUERY_CHARS = 200


@register
class HeuristicTeacher(TeacherModel):
    name = "heuristic"
    prompt_version = "heuristic-v1"

    def predict(self, example: Example) -> TeacherLabel:
        start_time = time.monotonic()
        difficulty_score = self._difficulty_score(example)
        routing_label, probabilities = self._build_label_distribution(difficulty_score)
        latency_seconds = time.monotonic() - start_time

        return TeacherLabel(
            query_id=example.id,
            dataset=example.dataset,
            teacher_name=self.name,
            prompt_version=self.prompt_version,
            routing_label=routing_label,
            probabilities=probabilities,
            latency_seconds=latency_seconds,
            input_tokens=len(example.query.split()),
            output_tokens=0,
            inference_cost=0.0,
        )

    @staticmethod
    def _difficulty_score(example: Example) -> float:
        """0.0 (easiest) .. 1.0 (hardest)."""
        if example.difficulty is not None:
            try:
                return min(float(example.difficulty) / 5.0, 1.0)  # MATH levels are 1-5
            except ValueError:
                pass
        query_length = len(example.query)
        query_length_range = _LONG_QUERY_CHARS - _SHORT_QUERY_CHARS
        return max(0.0, min(1.0, (query_length - _SHORT_QUERY_CHARS) / query_length_range))

    @staticmethod
    def _build_label_distribution(difficulty_score: float) -> tuple[str, dict[str, float]]:
        """Soft distribution over ROUTING_LABELS, peaked at the tier the score falls into."""
        tier_center_scores = {"small": 0.0, "medium": 0.5, "large": 1.0}
        assert set(tier_center_scores) == set(ROUTING_LABELS)
        tier_weights = {
            label: max(0.0, 1 - abs(difficulty_score - center_score) * 2) + 0.05
            for label, center_score in tier_center_scores.items()
        }
        weight_total = sum(tier_weights.values())
        probabilities = {label: round(weight / weight_total, 4) for label, weight in tier_weights.items()}
        routing_label = max(probabilities, key=probabilities.get)
        return routing_label, probabilities
