"""Answer checking: is a model's answer to a maths question correct?

Two steps, the same for every dataset:
1. Extract the model's FINAL answer: the last \\boxed{...}, else an {"answer": ...}
   JSON object, else the last bold span or number in the text.
2. Compare it with the reference using math-verify, which checks mathematical
   equivalence ("1/2" == "0.5" == "\\frac{1}{2}") instead of exact text.

On the submitted test answers this never marked an old-correct answer wrong
and recovered 46 correct answers the old string match missed (DESIGN.md §6.1).
"""
from __future__ import annotations

import json
import re
from typing import Optional

from math_verify import parse, verify

_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}")              # flat (non-nested) object
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")                # markdown **bold** span
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")      # 42, 1,234, 3.5, 1/2
_THOUSANDS_RE = re.compile(r"-?\d{1,3}(,\d{3})+(\.\d+)?")  # 1,234 or 12,345.6


def is_correct(answer_text: str, reference_answer: str) -> bool:
    """True if the final answer in `answer_text` equals `reference_answer`."""
    extracted = extract_final_answer(answer_text)
    if extracted is None:
        return False
    gold = parse(f"${_clean(reference_answer)}$")
    pred = parse(f"${_clean(extracted)}$")
    return bool(verify(gold, pred))


def _clean(answer: str) -> str:
    """Drop dollar signs; drop commas only from thousands separators, so
    tuples and intervals such as (-3,2) keep theirs."""
    answer = answer.strip().replace("\\$", "").replace("$", "")
    return answer.replace(",", "") if _THOUSANDS_RE.fullmatch(answer) else answer


def extract_final_answer(text: str) -> Optional[str]:
    for extractor in (extract_boxed_answer, _extract_json_answer, _extract_last_bold_or_number):
        result = extractor(text)
        if result is not None:
            return result
    return None


def extract_boxed_answer(text: str) -> Optional[str]:
    """Content of the last \\boxed{...} (brace-matched, handles nesting)."""
    start = text.rfind("\\boxed")
    if start == -1:
        return None
    i = text.find("{", start)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None  # unbalanced braces


def _extract_json_answer(text: str) -> Optional[str]:
    for candidate in reversed(_JSON_OBJECT_RE.findall(text)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "answer" in parsed:
            return str(parsed["answer"])
    return None


def _extract_last_bold_or_number(text: str) -> Optional[str]:
    bold = _BOLD_RE.findall(text)
    if bold:
        return bold[-1].strip()
    numbers = _NUMBER_RE.findall(text)
    return numbers[-1] if numbers else None
