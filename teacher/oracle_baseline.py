"""OracleDirectTeacher -- a RouteLLM-style baseline, not a real teacher.

RouteLLM (Ong et al., 2024; github.com/lm-sys/RouteLLM) trains a
lightweight classifier (its `bert` router: AutoModelForSequenceClassification
on a transformer backbone -- architecturally the same family as this
project's own ClassifierRouter) directly on a ground-truth routing signal,
with no teacher-LLM intermediary at inference OR training time. RouteLLM's
actual signal is subjective: Chatbot Arena human-preference battles between
two specific chat models. That has no equivalent in this project's domain
(GSM8K/MATH have objectively verifiable correctness, not subjective
preference), and RouteLLM's other routers (`mf`, `causal_llm`) depend on
external assets that don't fit this project either (`mf` calls OpenAI's
text-embedding-3-small API; `causal_llm` ships checkpoints calibrated to a
specific GPT-4/Mixtral pair) -- see the paper's Related Work discussion.

This class reproduces RouteLLM's actual methodological contribution --
direct classifier training on ground truth, skipping the teacher-LLM
reasoning/prompting step entirely -- using this project's own ground truth
(oracle routing labels, common/oracle/labeler.py) as the domain-appropriate
substitute for Arena preference data. It is registered as a `TeacherModel`
purely so `build-student-data` / `train-student-classifier` /
`select-router-checkpoint` / `evaluate-router` all work completely
unmodified: `predict()` here is a cache lookup of a pre-computed oracle
label, not a real model call (latency/cost are reported as 0 accordingly --
never plotted as a real router's serving cost).

Requires oracle labels to already exist for whatever split is being
labeled (data/oracle/<dataset>/<split>.labels.jsonl) -- run
`oracle/labeler.py`'s label_dataset() first if they don't. Train/
calibration/test already exist as of this writing; validation does not
(see the caller-facing note in run.py / EXPERIMENTS.md) -- either compute
it the same way, or use a held-out slice of the train split for checkpoint
selection instead.
"""
from __future__ import annotations

from common.config import ORACLE_DIR
from common.schema import Example, OracleLabel, TeacherLabel, read_jsonl
from teacher.base import TeacherModel, register


@register
class OracleDirectTeacher(TeacherModel):
    name = "oracle-direct"
    prompt_version = "oracle-direct-v1"
    output_version = "oracle_direct"
    fewshot_source_split = ""  # no few-shot demos -- this isn't a prompted model

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, str]] = {}  # dataset -> {query_id: routing_label}

    def _labels_for(self, dataset: str, split: str) -> dict[str, str]:
        key = f"{dataset}/{split}"
        if key not in self._cache:
            path = ORACLE_DIR / dataset / f"{split}.labels.jsonl"
            if not path.exists():
                raise FileNotFoundError(
                    f"No oracle labels at {path} -- run oracle labeling for "
                    f"dataset={dataset!r} split={split!r} first (see oracle/labeler.py)."
                )
            labels = read_jsonl(path, OracleLabel)
            self._cache[key] = {label.query_id: label.routing_label for label in labels}
        return self._cache[key]

    def predict(self, example: Example) -> TeacherLabel:
        labels = self._labels_for(example.dataset, example.split)
        routing_label = labels[example.id]  # KeyError is a real bug here, not swallowed
        return TeacherLabel(
            query_id=example.id,
            dataset=example.dataset,
            teacher_name=self.name,
            prompt_version=self.prompt_version,
            teacher_route=routing_label,
            latency_seconds=0.0,   # not a real model call -- see module docstring
            input_tokens=0,
            output_tokens=0,
            inference_cost=0.0,
            routing_reasoning="",  # oracle labels carry no reasoning trace
        )
