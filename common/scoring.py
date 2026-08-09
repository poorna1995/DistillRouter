"""Per-dataset correctness scoring — the metric that turns a candidate
model's raw generation into a right/wrong verdict for oracle labeling
(oracle/labeler.py).

Scoring is a pure function of (candidate answer text, reference answer),
with no model calls — so fixing a scoring bug later never requires
re-running any candidate; only `OracleAttempt`s (the expensive raw
generations) are cached, scoring itself is re-run fresh every time.

Same registry pattern as dataset/base.py and teacher/base.py: adding a new
dataset's scorer means writing one function (wherever that dataset's own
answer-extraction logic already lives, e.g. dataset/gsm8k.py) decorated
with `@register_scorer`, nothing else changes. Importing `dataset` (which
oracle/labeler.py needs anyway, for PROCESSED_DIR data) populates this
registry as a side effect, the same way dataset/__init__.py populates the
dataset-loader registry.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, Optional

_REGISTRY: Dict[str, Callable[[str, str], bool]] = {}


def register_scorer(dataset_name: str) -> Callable:
    """Function decorator: makes a scorer discoverable by dataset name."""

    def decorator(fn: Callable[[str, str], bool]) -> Callable[[str, str], bool]:
        _REGISTRY[dataset_name] = fn
        return fn

    return decorator


def score(dataset_name: str, candidate_answer_text: str, reference_answer: str) -> bool:
    """True if `candidate_answer_text` (a candidate model's raw generation)
    matches `reference_answer` under `dataset_name`'s correctness metric."""
    if dataset_name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"No scorer registered for '{dataset_name}'. Available: {available}")
    return _REGISTRY[dataset_name](candidate_answer_text, reference_answer)


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}")  # flat (non-nested) object
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")     # markdown **bold** span
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_candidate_answer(text: str) -> Optional[str]:
    """Extract a candidate model's stated final answer, trying every known
    answer-statement convention in order, falling through to the next only
    if the previous one finds nothing:

    1. \\boxed{...} — candidates are currently instructed to use this.
    2. {"answer": ...} JSON — an earlier candidate instruction; kept as a
       fallback since some models default to it regardless of what they
       were actually asked for.
    3. A last-resort scrape of the response's own natural-language ending
       (last markdown **bold** span, else the last number in the text).

    Why a chain instead of picking one mandated format: two live
    experiments (see conversation history) each tried enforcing a single
    format and each left a large fraction of responses unscorable —
    \\boxed{...} alone left Small's gsm8k attempts 19/20 unparseable;
    switching wholesale to JSON *reduced* Medium's compliance (apparently
    already trained on \\boxed{} from math benchmarks) and cratered MATH's
    success rate. Chained, lenient extraction is a safe, one-directional
    improvement over any single format: it only kicks in where the
    preferred format is already absent, so it can only recover otherwise-
    unscorable responses, never override one that already parsed.

    The fallback tier (3) is a heuristic and can occasionally grab the
    wrong number if a response rambles without ever committing to a final
    answer — accepted as a known imprecision, same spirit as the MATH
    scorer's un-handled symbolic-equivalence gap.
    """
    for extractor in (extract_boxed_answer, extract_json_answer, _extract_fallback_answer):
        result = extractor(text)
        if result is not None:
            return result
    return None


def extract_json_answer(text: str) -> Optional[str]:
    """Return the "answer" field of the last valid `{"answer": ...}` JSON
    object in `text`. Scans from the end and returns the first parse that
    has an "answer" key, so trailing commentary after the JSON object
    doesn't matter.
    """
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
    """Return the content of the final \\boxed{...} in `text`.

    Braces can nest (e.g. \\boxed{\\frac{1}{2}}), so this is a small
    brace-matching scan rather than a naive regex. Also used by
    dataset/math.py to parse \\boxed{} out of MATH's own *reference*
    solutions at prepare-data time — a fixed property of that dataset's
    source format, unrelated to its use here on candidate model output.
    """
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
    return None  # unbalanced braces; leave for manual inspection


def _extract_fallback_answer(text: str) -> Optional[str]:
    """Last resort when neither \\boxed{} nor JSON is present: the last
    markdown **bold** span if there is one (models that skip the requested
    format often still bold their final answer), else the last number
    anywhere in the text.
    """
    bold_matches = _BOLD_RE.findall(text)
    if bold_matches:
        return bold_matches[-1].strip()
    number_matches = _NUMBER_RE.findall(text)
    return number_matches[-1] if number_matches else None
