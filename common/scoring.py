"""Per-dataset correctness scoring: candidate answer text -> right/wrong."""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, Optional

_REGISTRY: Dict[str, Callable[[str, str], bool]] = {}


def register_scorer(dataset_name: str) -> Callable:
    """Decorator: registers a scorer function under a dataset name."""

    def decorator(fn: Callable[[str, str], bool]) -> Callable[[str, str], bool]:
        _REGISTRY[dataset_name] = fn
        return fn

    return decorator


def score(dataset_name: str, candidate_answer_text: str, reference_answer: str) -> bool:
    """True if candidate_answer_text matches reference_answer under dataset_name's scorer."""
    if dataset_name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"No scorer registered for '{dataset_name}'. Available: {available}")
    return _REGISTRY[dataset_name](candidate_answer_text, reference_answer)


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}")  # flat (non-nested) object
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")     # markdown **bold** span
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_candidate_answer(text: str) -> Optional[str]:
    """Extract a model's final answer: \\boxed{...}, else {"answer": ...}
    JSON, else the last bold span or number in the text."""
    for extractor in (extract_boxed_answer, extract_json_answer, _extract_fallback_answer):
        result = extractor(text)
        if result is not None:
            return result
    return None


def extract_json_answer(text: str) -> Optional[str]:
    """Return the "answer" field of the last valid {"answer": ...} object in text."""
    matches = _JSON_OBJECT_RE.findall(text)
    for candidate_text in reversed(matches):
        try:
            parsed = json.loads(candidate_text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "answer" in parsed:
            return str(parsed["answer"])
    return None


def extract_boxed_answer(text: str) -> Optional[str]:
    """Return the content of the final \\boxed{...} in text (brace-matched, handles nesting)."""
    key = "\\boxed"
    start = text.rfind(key)
    if start == -1:
        return None
    i = start + len(key)
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None
    depth = 0
    content_start = i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[content_start + 1 : j]
    return None  # unbalanced braces


def _extract_fallback_answer(text: str) -> Optional[str]:
    """Last resort: the last bold markdown span, else the last number in the text."""
    bold_matches = _BOLD_RE.findall(text)
    if bold_matches:
        return bold_matches[-1].strip()
    number_matches = _NUMBER_RE.findall(text)
    return number_matches[-1] if number_matches else None
