"""Teacher labeling pipeline.

- base.py           — TeacherModel ABC + registry.
- qwen_teacher_v1.py — QwenTeacherV1 ("qwen2.5-3b-v1"), few-shot, label-only.
                       Output: data/teacher/<dataset>/v1/<split>.jsonl.
- qwen_teacher_v2.py — QwenTeacherV2 ("qwen2.5-3b-v2"), few-shot, label + reasoning.
                       Output: data/teacher/<dataset>/v2/<split>.jsonl.
- fewshot.py        — shared few-shot loading, prompts, parsing, self-referential guard.
- cache.py          — TeacherLabelCache, keyed on (query_id, prompt_version).
- labeler.py        — label_dataset(): read -> cache-check -> dedupe -> predict -> cache.

Both registered teachers require real oracle-labeled calibration data to
instantiate (see oracle/labeler.py) — there is no blind (no-few-shot)
variant. Oracle labeling itself lives in oracle/, not here.
"""
from teacher.base import available_teachers, get_teacher_class  # noqa: F401

from teacher import qwen_teacher_v1, qwen_teacher_v2  # noqa: F401,E402
