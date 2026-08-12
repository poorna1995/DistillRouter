"""QwenTeacherV1 — few-shot calibrated, label-only routing prediction (no
reasoning). See qwen_teacher_v2.py for the label+reasoning sibling; shared
plumbing lives in common/hf_generation.py and teacher/fewshot.py. Output:
data/teacher/<dataset>/v1/<split>.jsonl.
"""
from __future__ import annotations

from common.hf_generation import generate_chat_response, load_causal_lm
from common.schema import Example, TeacherLabel
from teacher.base import TeacherModel, register
from teacher.fewshot import (
    FEWSHOT_SOURCE_SPLIT,
    LABEL_ONLY_SYSTEM_PREAMBLE,
    check_not_self_referential,
    format_fewshot_block,
    load_fewshot_examples_or_raise,
    parse_label_only_response,
)


@register
class QwenTeacherV1(TeacherModel):
    name = "qwen2.5-3b-v1"
    prompt_version = "qwen-fewshot-label-only-v2"  # v2: shared preamble intro + JSON key renamed to teacher_route
    output_version = "v1"
    fewshot_source_split = FEWSHOT_SOURCE_SPLIT
    MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
    MAX_NEW_TOKENS = 60  # just a label — no reasoning to also emit

    def __init__(self) -> None:
        fewshot = load_fewshot_examples_or_raise(self.name)
        self._fewshot_block = format_fewshot_block(fewshot)
        self._fewshot_ids = {ex.id for ex, _ in fewshot}
        self._tokenizer, self._model = load_causal_lm(self.MODEL_ID)

    def predict(self, example: Example) -> TeacherLabel:
        check_not_self_referential(example.id, self._fewshot_ids, allow=self.allow_self_referential)
        messages = [
            {"role": "system", "content": LABEL_ONLY_SYSTEM_PREAMBLE},
            {"role": "user", "content": f"{self._fewshot_block}\n\nProblem: {example.query}"},
        ]
        result = generate_chat_response(self._tokenizer, self._model, messages, self.MAX_NEW_TOKENS)
        routing_label = parse_label_only_response(result.text)

        return TeacherLabel(
            query_id=example.id,
            dataset=example.dataset,
            teacher_name=self.name,
            prompt_version=self.prompt_version,
            teacher_route=routing_label,
            latency_seconds=result.latency_seconds,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            inference_cost=0.0,
            routing_reasoning="",  # not requested in this version's prompt
        )
