"""Teacher labeling pipeline.

- `base.py`         — TeacherModel ABC + registry (mirrors dataset/base.py's pattern).
- `heuristic.py`    — HeuristicTeacher, a zero-cost deterministic baseline that
                       exercises the pipeline without any model calls.
- `qwen_teacher.py` — QwenTeacher, Qwen2.5-3B prompted (few-shot, calibrated
                       against real oracle.labeler.py ground truth) to predict
                       a routing label + soft distribution from query text
                       alone. See DISTILLROUTER_SPEC.md's "Teacher Model"
                       section and qwen_teacher.py's module docstring.
- `cache.py`        — TeacherLabelCache: makes labeling idempotent/incremental,
                       keyed on (query_id, prompt_version).
- `labeler.py`      — label_dataset(): read -> cache-check -> dedupe -> predict -> cache.

Oracle labeling itself (executing each candidate tier to score against
ground truth) lives in oracle/, not here — teacher/ only consumes its
output (oracle/labeler.py's calibration labels).
"""
from teacher.base import available_teachers, get_teacher_class  # noqa: F401

from teacher import heuristic, qwen_teacher  # noqa: F401,E402
