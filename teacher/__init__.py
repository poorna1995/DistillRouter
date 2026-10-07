"""Teacher labeling pipeline.

- base.py           — TeacherModel ABC + registry.
- qwen_teacher.py   — QwenTeacher ("qwen2.5-14b"), binary, two-stage
                       (reasoning, then few-shot label).
                       Output: data/teacher/<dataset>/v3/<split>.jsonl.
- fewshot.py        — shared few-shot loading, prompts, parsing, self-referential guard.
- cache.py          — TeacherLabelCache, keyed on (query_id, prompt_version).
- labeler.py        — label_dataset(): read -> cache-check -> dedupe -> predict -> cache.
- oracle_baseline.py — OracleDirectTeacher ("oracle-direct"), not a real
                       teacher: a RouteLLM-style baseline that looks up the
                       pre-computed oracle label instead of calling a model.
                       Output: data/teacher/<dataset>/oracle_direct/<split>.jsonl.

The prompted teacher requires real oracle-labeled calibration data to
instantiate (see oracle/labeler.py) — there is no blind (no-few-shot)
variant. Oracle labeling itself lives in oracle/, not here.
"""
from teacher.base import available_teachers, get_teacher_class  # noqa: F401

from teacher import oracle_baseline, qwen_teacher  # noqa: F401,E402
