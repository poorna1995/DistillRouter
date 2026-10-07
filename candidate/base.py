"""Candidate answering models (small and large) — the models oracle labeling
executes to measure whether each one answers a question correctly."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from common.config import CANDIDATE_MODELS, MAX_NEW_TOKENS
from common.hf_generation import generate_chat_response, load_causal_lm


@dataclass
class CandidateResponse:
    answer_text: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int


class CandidateModel(ABC):
    tier: str             # one of common.config.LABELS
    model_id: str
    prompt_version: str   # part of CandidateAttempt's cache key — bump when the instruction text changes

    @abstractmethod
    def answer(self, query: str) -> CandidateResponse:
        """Generate this tier's raw answer to `query`."""


class HuggingFaceCandidate(CandidateModel):
    ANSWER_INSTRUCTION = (
        "You are an expert in solving problems, Think step by step internally, don't write it down.\n"
        "Provide the final answer only in this exact format: \\boxed{answer}\n"
        "For example, if the final answer is 42, the last line must be "
        "exactly: \\boxed{42}\n"
    )
    PROMPT_VERSION = "boxed-internal-cot-v3"
    MAX_NEW_TOKENS = MAX_NEW_TOKENS

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
    """Instantiate (loads weights to GPU) the candidate model for `tier`."""
    if tier not in CANDIDATE_MODELS:
        available = ", ".join(CANDIDATE_MODELS)
        raise KeyError(f"Unknown tier '{tier}'. Available: {available}")
    return HuggingFaceCandidate(tier, CANDIDATE_MODELS[tier])
