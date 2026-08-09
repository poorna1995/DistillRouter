"""Oracle labeling: executes each candidate tier in cost order (small ->
medium -> large, common.config.ROUTING_LABELS), stopping at the first tier
whose answer scores correct against the dataset's correctness metric
(common.scoring). The label is that cheapest successful tier; if none
succeed, it defaults to "large" — the best tried option, even though it
failed, so every query still gets a label.

Raw candidate generations are cached independently of scoring
(oracle/cache.py) — fixing a scoring bug later never requires re-running a
candidate model. The derived label file, by contrast, is cheap to produce
from cached attempts, so it's fully rewritten every run rather than
incrementally cached.

Each candidate tier's model is loaded lazily, at most once per run, the
first time a query actually needs it (i.e. cheaper tiers already tried and
failed, or aren't cached) — a calibration set where every query resolves
at "small" never loads Medium or Large at all.
"""
from __future__ import annotations

import dataset  # noqa: F401 — populates common.scoring's per-dataset scorer registry as an import side effect

from common.config import DEFAULT_CANDIDATE_ROSTER, ORACLE_DIR, PROCESSED_DIR, ROUTING_LABELS
from common.schema import OracleAttempt, OracleLabel, read_jsonl, write_jsonl
from common.scoring import score
from candidate.base import HuggingFaceCandidate, get_candidate
from oracle.cache import OracleAttemptCache

# Every current candidate is a HuggingFaceCandidate, and prompt_version is a
# class-level constant (candidate/base.py) — reading it for the cache-key
# check must not require loading model weights, or lazy-loading below is
# defeated (the whole point is to skip loading a tier's weights when every
# query already resolves at a cheaper one).
_CANDIDATE_PROMPT_VERSION = HuggingFaceCandidate.PROMPT_VERSION


def label_dataset(dataset_name: str, split: str = "calibration") -> dict:
    processed_path = PROCESSED_DIR / dataset_name / f"{split}.jsonl"
    examples = read_jsonl(processed_path)

    attempt_cache = OracleAttemptCache(ORACLE_DIR / dataset_name / f"{split}.attempts.jsonl")
    loaded_candidates: dict[str, object] = {}  # tier -> CandidateModel, populated on first real need

    labels = []
    stats = {
        "cache_hits": 0,
        "new_attempts": 0,
        "labels_by_tier": {tier: 0 for tier in ROUTING_LABELS},
        "no_tier_succeeded": 0,
    }
    total = len(examples)

    for i, example in enumerate(examples, 1):
        attempted_tiers = []
        assigned_label = None

        for tier in ROUTING_LABELS:
            attempted_tiers.append(tier)
            model_id = DEFAULT_CANDIDATE_ROSTER[tier]

            cached_attempt = attempt_cache.get(example.id, tier, model_id, _CANDIDATE_PROMPT_VERSION)
            if cached_attempt is not None:
                stats["cache_hits"] += 1
                attempt = cached_attempt
            else:
                if tier not in loaded_candidates:
                    loaded_candidates[tier] = get_candidate(tier)
                response = loaded_candidates[tier].answer(example.query)
                attempt = OracleAttempt(
                    query_id=example.id,
                    dataset=dataset_name,
                    tier=tier,
                    model_id=model_id,
                    prompt_version=_CANDIDATE_PROMPT_VERSION,
                    answer_text=response.answer_text,
                    latency_seconds=response.latency_seconds,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )
                attempt_cache.add(attempt)
                stats["new_attempts"] += 1

            if score(dataset_name, attempt.answer_text, example.reference_answer):
                assigned_label = tier
                break

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
                attempted_tiers=attempted_tiers,
                succeeded=succeeded,
            )
        )
        outcome = assigned_label if succeeded else f"{assigned_label} (no tier succeeded)"
        print(
            f"[{dataset_name}/{split}] {i}/{total} {example.id} -> {outcome}, tried {attempted_tiers}",
            flush=True,
        )

    write_jsonl(ORACLE_DIR / dataset_name / f"{split}.labels.jsonl", labels)
    return stats
