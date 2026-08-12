"""QwenTeacherV2 — two-stage, reasoning-first routing. Stage 1: one-sentence
reasoning, zero-shot, route unknown. Stage 2: route from that reasoning,
few-shot. See fewshot.py's docstring for why this order and shape.
Output: data/teacher/<dataset>/v2/<split>.jsonl.
"""
from __future__ import annotations

from common.hf_generation import generate_chat_response, load_causal_lm
from common.schema import Example, TeacherLabel
from teacher.base import TeacherModel, register
from teacher.fewshot import (
    FEWSHOT_SOURCE_SPLIT,
    LABEL_ONLY_SYSTEM_PREAMBLE,
    REASONING_EXTRACTOR_PREAMBLE,
    check_not_self_referential,
    format_fewshot_block,
    load_fewshot_examples_or_raise,
    parse_label_only_response,
    parse_reasoning_only_response,
)


@register
class QwenTeacherV2(TeacherModel):
    name = "qwen2.5-3b-v2"
    # v5: reasoning-first + one-sentence reasoning (~15-20 words, 50-token cap) + bars
    # solution-plan leakage, not just computed results. See fewshot.py's docstring.
    prompt_version = "qwen-twostage-reasoning-v5"
    output_version = "v2"
    fewshot_source_split = FEWSHOT_SOURCE_SPLIT
    MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
    REASONING_MAX_NEW_TOKENS = 50   # stage 1: one short sentence + JSON syntax
    ROUTE_MAX_NEW_TOKENS = 60       # stage 2: just a label, same as QwenTeacherV1

    def __init__(self) -> None:
        fewshot = load_fewshot_examples_or_raise(self.name)
        self._fewshot_block = format_fewshot_block(fewshot)
        self._fewshot_ids = {ex.id for ex, _ in fewshot}
        self._tokenizer, self._model = load_causal_lm(self.MODEL_ID)

    def predict(self, example: Example) -> TeacherLabel:
        check_not_self_referential(example.id, self._fewshot_ids, allow=self.allow_self_referential)

        reasoning_messages = [
            {"role": "system", "content": REASONING_EXTRACTOR_PREAMBLE},
            {"role": "user", "content": f"Problem: {example.query}"},
        ]
        reasoning_result = generate_chat_response(
            self._tokenizer, self._model, reasoning_messages, self.REASONING_MAX_NEW_TOKENS
        )
        reasoning = parse_reasoning_only_response(reasoning_result.text)

        route_messages = [
            {"role": "system", "content": LABEL_ONLY_SYSTEM_PREAMBLE},
            {"role": "user", "content": (
                f"{self._fewshot_block}\n\nProblem: {example.query}\n\n"
                f"Reasoning requirements: {reasoning}"
            )},
        ]
        route_result = generate_chat_response(
            self._tokenizer, self._model, route_messages, self.ROUTE_MAX_NEW_TOKENS
        )
        routing_label = parse_label_only_response(route_result.text)

        return TeacherLabel(
            query_id=example.id,
            dataset=example.dataset,
            teacher_name=self.name,
            prompt_version=self.prompt_version,
            teacher_route=routing_label,
            latency_seconds=reasoning_result.latency_seconds + route_result.latency_seconds,
            input_tokens=reasoning_result.input_tokens + route_result.input_tokens,
            output_tokens=reasoning_result.output_tokens + route_result.output_tokens,
            inference_cost=0.0,
            routing_reasoning=reasoning,
        )
