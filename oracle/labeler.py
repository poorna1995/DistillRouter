"""Runs every candidate tier on every query (no early-exit) and assigns
the cheapest correct tier as the oracle label. Ground truth — never uses
the teacher. Always exhaustive so small_correct/medium_correct/large_correct
are complete for every query, not just the winning tier.
"""
from __future__ import annotations

from typing import Optional

import dataset  # noqa: F401 — populates common.scoring's scorer registry

from common.config import DEFAULT_CANDIDATE_ROSTER, ORACLE_DIR, PROCESSED_DIR, ROUTING_LABELS
from common.schema import CandidateAttempt, OracleLabel, read_jsonl, write_jsonl
from common.scoring import score
from candidate.base import CandidateModel, HuggingFaceCandidate, get_candidate
from oracle.cache import CandidateAttemptCache

_CANDIDATE_PROMPT_VERSION = HuggingFaceCandidate.PROMPT_VERSION


def label_dataset(dataset_name: str, split: str = "calibration", limit: Optional[int] = None) -> dict:
    """Writes labels to data/oracle/<dataset>/<split>.labels.jsonl."""
    processed_path = PROCESSED_DIR / dataset_name / f"{split}.jsonl"
    examples = read_jsonl(processed_path)
    if limit is not None:
        examples = examples[:limit]

    attempt_cache = CandidateAttemptCache(ORACLE_DIR / dataset_name / f"{split}.attempts.jsonl")
    loaded_candidates: dict[str, CandidateModel] = {}  # tier -> CandidateModel, populated on first real need

    labels = []
    stats = {
        "cache_hits": 0,
        "new_attempts": 0,
        "labels_by_tier": {tier: 0 for tier in ROUTING_LABELS},
        "no_tier_succeeded": 0,
    }
    total = len(examples)

    for i, example in enumerate(examples, 1):
        tier_correct: dict[str, bool] = {}

        for tier in ROUTING_LABELS:
            model_id = DEFAULT_CANDIDATE_ROSTER[tier]
            cached_attempt = attempt_cache.get(example.id, tier, model_id, _CANDIDATE_PROMPT_VERSION)
            if cached_attempt is not None:
                stats["cache_hits"] += 1
                attempt = cached_attempt
            else:
                if tier not in loaded_candidates:
                    loaded_candidates[tier] = get_candidate(tier)
                response = loaded_candidates[tier].answer(example.query)
                attempt = CandidateAttempt(
                    query_id=example.id,
                    dataset=dataset_name,
                    tier=tier,
                    model_id=model_id,
                    prompt_version=_CANDIDATE_PROMPT_VERSION,
                    answer_text=response.answer_text,
                    correct=score(dataset_name, response.answer_text, example.reference_answer),
                    latency_seconds=response.latency_seconds,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )
                attempt_cache.add(attempt)
                stats["new_attempts"] += 1

            tier_correct[tier] = attempt.correct

        assigned_label = next((tier for tier in ROUTING_LABELS if tier_correct[tier]), None)
        succeeded = assigned_label is not None
        if not succeeded:
            assigned_label = "large"  # best tried option, even though it failed
            stats["no_tier_succeeded"] += 1

        stats["labels_by_tier"][assigned_label] += 1
        labels.append(
            OracleLabel(
                query_id=example.id,
                dataset=dataset_name,
                routing_label=assigned_label,
                succeeded=succeeded,
                small_correct=tier_correct["small"],
                medium_correct=tier_correct["medium"],
                large_correct=tier_correct["large"],
            )
        )
        outcome = assigned_label if succeeded else f"{assigned_label} (no tier succeeded)"
        print(
            f"[{dataset_name}/{split}] {i}/{total} {example.id} -> {outcome}, correct={tier_correct}",
            flush=True,
        )

    write_jsonl(ORACLE_DIR / dataset_name / f"{split}.labels.jsonl", labels)
    return stats
