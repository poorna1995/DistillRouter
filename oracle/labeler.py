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
    """Writes labels to data/oracle/<dataset>/<split>.labels.jsonl.

    Merges with whatever's already in that file, keyed on query_id --
    a `limit` smaller than (or just different from) a previous call's
    never truncates labels already computed for queries outside this
    call's slice. (Before this merge step existed, any --limit smaller
    than an already-labeled split's size silently discarded every label
    for the queries outside that slice -- the underlying per-attempt
    cache below was never affected, only this aggregated output file.)
    """
    processed_path = PROCESSED_DIR / dataset_name / f"{split}.jsonl"
    examples = read_jsonl(processed_path)
    if limit is not None:
        examples = examples[:limit]

    output_path = ORACLE_DIR / dataset_name / f"{split}.labels.jsonl"
    labels_by_id: dict[str, OracleLabel] = {}
    if output_path.exists():
        labels_by_id = {label.query_id: label for label in read_jsonl(output_path, OracleLabel)}
    n_existing = len(labels_by_id)

    attempt_cache = CandidateAttemptCache(ORACLE_DIR / dataset_name / f"{split}.attempts.jsonl")
    loaded_candidates: dict[str, CandidateModel] = {}  # tier -> CandidateModel, populated on first real need

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
        labels_by_id[example.id] = OracleLabel(
            query_id=example.id,
            dataset=dataset_name,
            routing_label=assigned_label,
            succeeded=succeeded,
            small_correct=tier_correct["small"],
            medium_correct=tier_correct["medium"],
            large_correct=tier_correct["large"],
        )
        outcome = assigned_label if succeeded else f"{assigned_label} (no tier succeeded)"
        print(
            f"[{dataset_name}/{split}] {i}/{total} {example.id} -> {outcome}, correct={tier_correct}",
            flush=True,
        )

    write_jsonl(output_path, list(labels_by_id.values()))
    print(
        f"[{dataset_name}/{split}] done: {len(labels_by_id)} labels on disk "
        f"({n_existing} pre-existing + {total} processed this call, {len(labels_by_id) - n_existing} new) "
        f"-> {output_path} (cache hits={stats['cache_hits']}, new attempts={stats['new_attempts']}, "
        f"by tier={stats['labels_by_tier']}, no tier succeeded={stats['no_tier_succeeded']})"
    )
    return stats
