"""QwenTeacher — Qwen2.5-3B, prompted (not fine-tuned) to predict a routing
label + soft probability distribution from query text alone, without
executing any candidate model. See DISTILLROUTER_SPEC.md Section 5.2.

Basis for labeling: the same criterion oracle labeling uses
(oracle/labeler.py) — the cheapest candidate tier that would answer the
query correctly — but predicted blind, matching exactly what the student
router will have access to at inference time. The teacher is calibrated
toward that criterion via few-shot examples pulled directly from real
oracle labels (data/oracle/<dataset>/calibration.labels.jsonl), preferring
examples where a tier genuinely succeeded (OracleLabel.succeeded) over
ones that only defaulted to "large" after every tier failed — a defaulted
example teaches "nothing here works", not "large is the right tier", and
would miscalibrate the "large" demonstration if used.

Known limitation: the probability distribution below is the model's
self-reported ("verbalized") JSON numbers, not derived from token
logprobs. Research on LLM confidence calibration finds verbalized
confidence is measurably less well-calibrated than logprob-derived
confidence — see conversation history. Upgrading `_parse_response` to read
logprobs at the label token instead of trusting the stated numbers is a
known next step, not yet implemented here.
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.config import DEFAULT_CANDIDATE_ROSTER, ORACLE_DIR, PROCESSED_DIR, ROUTING_LABELS
from common.schema import Example, OracleLabel, TeacherLabel, read_jsonl
from teacher.base import TeacherModel, register

_FEWSHOT_PER_TIER = 2  # demonstration examples per tier, pooled across every oracle-labeled dataset

_TIER_DESCRIPTIONS = {
    "small": f"{DEFAULT_CANDIDATE_ROSTER['small']} — reliably correct only on short, "
    "single-step arithmetic or direct lookups",
    "medium": f"{DEFAULT_CANDIDATE_ROSTER['medium']} — handles standard multi-step "
    "word problems",
    "large": f"{DEFAULT_CANDIDATE_ROSTER['large']} — needed for problems requiring several "
    "dependent reasoning steps, careful algebraic manipulation, or unusual phrasing",
}

_SYSTEM_PREAMBLE = (
    "You are a query router. Given a math problem, predict which of three "
    "model tiers would give the CORRECT final answer, using the CHEAPEST "
    "tier that would succeed:\n"
    f"- small: {_TIER_DESCRIPTIONS['small']}\n"
    f"- medium: {_TIER_DESCRIPTIONS['medium']}\n"
    f"- large: {_TIER_DESCRIPTIONS['large']}\n\n"
    "You are not solving the problem yourself, and you will not see any "
    "model's actual answer — predict from the query
     text alone.\n"
    "Respond with JSON only, in this exact shape:\n"
    '{"label": "small"|"medium"|"large", "probabilities": '
    '{"small": <float>, "medium": <float>, "large": <float>}}\n'
    "The three probabilities must sum to 1."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _load_fewshot_examples() -> list[tuple[Example, OracleLabel]]:

    pool: dict[str, list[tuple[Example, OracleLabel]]] = {tier: [] for tier in ROUTING_LABELS}
    for dataset_dir in sorted(p for p in ORACLE_DIR.glob("*") if p.is_dir()):
        labels_path = dataset_dir / "calibration.labels.jsonl"
        if not labels_path.exists():
            continue
        dataset_name = dataset_dir.name
        examples_by_id = {ex.id: ex for ex in read_jsonl(PROCESSED_DIR / dataset_name / "calibration.jsonl")}
        for label in read_jsonl(labels_path, OracleLabel):
            example = examples_by_id.get(label.query_id)
            if example is not None:
                pool[label.routing_label].append((example, label))

    fewshot = []
    for tier in ROUTING_LABELS:
        candidates = sorted(pool[tier], key=lambda pair: not pair[1].succeeded)  # succeeded=True first
        fewshot.extend(candidates[:_FEWSHOT_PER_TIER])
    return fewshot


def _demo_probabilities(true_label: str) -> dict:
    """A peaked-but-not-certain distribution for the few-shot demo's target
    output. Oracle labels are a hard pass/fail per tier, not a real soft
    distribution, so this is a plausible stand-in that teaches the JSON
    *format* and general shape (confident but not absolute) — not a
    literal probability elicited from anywhere.
    """
    peak = 0.7
    remainder = round((1 - peak) / (len(ROUTING_LABELS) - 1), 2)
    return {tier: (peak if tier == true_label else remainder) for tier in ROUTING_LABELS}


def _format_fewshot_block(fewshot: list[tuple[Example, OracleLabel]]) -> str:
    blocks = []
    for example, label in fewshot:
        target = {"label": label.routing_label, "probabilities": _demo_probabilities(label.routing_label)}
        blocks.append(f"Problem: {example.query}\n{json.dumps(target)}")
    return "\n\n".join(blocks)


def _normalize_probabilities(probabilities: dict) -> Optional[dict]:
    try:
        values = {tier: float(probabilities[tier]) for tier in ROUTING_LABELS}
    except (KeyError, TypeError, ValueError):
        return None
    total = sum(values.values())
    if total <= 0:
        return None
    return {tier: round(v / total, 4) for tier, v in values.items()}


def _fallback_response() -> tuple[str, dict]:
    uniform = round(1 / len(ROUTING_LABELS), 4)
    return "large", {tier: uniform for tier in ROUTING_LABELS}


def _parse_response(raw_text: str) -> tuple[str, dict]:
    """Best-effort JSON extraction with a safe fallback: malformed output
    defaults to "large" (the same conservative default oracle labeling
    uses when no tier succeeds) with a uniform distribution, rather than
    crashing a whole labeling run over one bad response.
    """
    match = _JSON_RE.search(raw_text)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            label = parsed.get("label")
            probabilities = _normalize_probabilities(parsed.get("probabilities", {}))
            if label in ROUTING_LABELS and probabilities is not None:
                return label, probabilities
    return _fallback_response()


@register
class QwenTeacher(TeacherModel):
    name = "qwen2.5-3b"
    prompt_version = "qwen-json-v1"
    MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
    MAX_NEW_TOKENS = 200  # short JSON output — no chain-of-thought headroom needed here

    def __init__(self) -> None:
        fewshot = _load_fewshot_examples()
        if not fewshot:
            raise RuntimeError(
                "No oracle-labeled calibration examples found. Run "
                "`python run.py oracle-label --split calibration` first — "
                "QwenTeacher's few-shot block is calibrated against real "
                "oracle ground truth, not hand-written examples (see "
                "module docstring)."
            )
        self._fewshot_block = _format_fewshot_block(fewshot)
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForCausalLM.from_pretrained(self.MODEL_ID, dtype=torch.bfloat16).to("cuda")

    def predict(self, example: Example) -> TeacherLabel:
        messages = [
            {"role": "system", "content": _SYSTEM_PREAMBLE},
            {"role": "user", "content": f"{self._fewshot_block}\n\nProblem: {example.query}"},
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to("cuda")

        start_time = time.monotonic()
        output = self._model.generate(**inputs, max_new_tokens=self.MAX_NEW_TOKENS, do_sample=False)
        latency_seconds = time.monotonic() - start_time

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = output[0][input_length:]
        raw_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        routing_label, probabilities = _parse_response(raw_text)

        return TeacherLabel(
            query_id=example.id,
            dataset=example.dataset,
            teacher_name=self.name,
            prompt_version=self.prompt_version,
            routing_label=routing_label,
            probabilities=probabilities,
            latency_seconds=latency_seconds,
            input_tokens=input_length,
            output_tokens=len(generated_tokens),
            inference_cost=0.0,  # local inference, no metered API cost
        )
