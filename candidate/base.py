"""Candidate answering models — the models routing sends queries to, and
the models oracle labeling (oracle/labeler.py) actually executes to
determine ground-truth routing labels.

Distinct from `teacher.base.TeacherModel`: a `CandidateModel` answers the
query (real generation, real cost); a `TeacherModel` predicts, from query
text alone, which `CandidateModel` tier would answer it correctly, without
ever calling one.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from common.config import DEFAULT_CANDIDATE_ROSTER
from common.hf_generation import generate_chat_response, load_causal_lm


@dataclass
class CandidateResponse:
    answer_text: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int


class CandidateModel(ABC):
    tier: str             # one of common.config.ROUTING_LABELS
    model_id: str
    prompt_version: str   # part of OracleAttempt's cache key — bump when the instruction text changes

    @abstractmethod
    def answer(self, query: str) -> CandidateResponse:
        """Generate this tier's raw answer to `query`."""


class HuggingFaceCandidate(CandidateModel):
    """A candidate tier backed by a local Hugging Face causal LM.

    Loads weights once at construction (the expensive part — model to
    GPU), reused across every `answer()` call. Every candidate in the
    current roster (`common.config.DEFAULT_CANDIDATE_ROSTER`) uses this
    one implementation, since all three are plain HF text models — a
    future non-HF-backed candidate (e.g. an API model) would be a new
    `CandidateModel` subclass, not a change here.

    Every candidate is given the same final-answer instruction regardless
    of dataset. Extraction is deliberately lenient about which convention
    the candidate actually followed — see
    `common.scoring.extract_candidate_answer` — because two live
    experiments (conversation history) each showed that no single
    mandated format is followed reliably across this mixed-vendor roster:
    v1/v2 (\\boxed{...}, with and without a worked example) left Small's
    gsm8k attempts ~19/20 unparseable; v3 (switching wholesale to JSON)
    *reduced* Medium's compliance and cratered MATH's success rate,
    apparently because Qwen models are already well-trained on \\boxed{}
    specifically. Back on v2's instruction/version here — the best-
    performing single instruction found — with the actual fix (lenient,
    multi-format extraction) living in the scorer instead, where it can't
    regress a model that's already doing the right thing.
    """

    ANSWER_INSTRUCTION = (
        "Solve the problem. Think step by step.\n"
        "On the very last line of your response, write only the final "
        "answer in this exact format: \\boxed{answer}\n"
        "For example, if the final answer is 42, the last line must be "
        "exactly: \\boxed{42}\n"
        "Do not write \"Final answer:\" or any other phrasing — the boxed "
        "line is the only acceptable way to state the final answer."
    )
    PROMPT_VERSION = "boxed-cot-v2"
    MAX_NEW_TOKENS = 1024

    def __init__(self, tier: str, model_id: str) -> None:
        self.tier = tier
        self.model_id = model_id
        self.prompt_version = self.PROMPT_VERSION
        self._tokenizer, self._model = load_causal_lm(model_id)

    def answer(self, query: str) -> CandidateResponse:
        messages = [{"role": "user", "content": f"{query}\n\n{self.ANSWER_INSTRUCTION}"}]
        result = generate_chat_response(self._tokenizer, self._model, messages, self.MAX_NEW_TOKENS)
        return CandidateResponse(
            answer_text=result.text,
            latency_seconds=result.latency_seconds,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )


def get_candidate(tier: str) -> CandidateModel:
    """Instantiate (loads weights to GPU) the current roster's candidate for `tier`."""
    if tier not in DEFAULT_CANDIDATE_ROSTER:
        available = ", ".join(DEFAULT_CANDIDATE_ROSTER)
        raise KeyError(f"Unknown tier '{tier}'. Available: {available}")
    return HuggingFaceCandidate(tier, DEFAULT_CANDIDATE_ROSTER[tier])
