"""Base class and registry for teacher models. Add a backend by writing a
TeacherModel subclass decorated with @register."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type

from common.schema import Example, TeacherLabel

_REGISTRY: Dict[str, Type["TeacherModel"]] = {}


def register(cls: Type["TeacherModel"]) -> Type["TeacherModel"]:
    """Class decorator: makes a teacher discoverable by its `name` attribute."""
    _REGISTRY[cls.name] = cls
    return cls


def get_teacher_class(name: str) -> Type["TeacherModel"]:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown teacher '{name}'. Available: {available}")
    return _REGISTRY[name]


def available_teachers() -> list[str]:
    return sorted(_REGISTRY)


class TeacherModel(ABC):
    """One subclass = one way of producing a TeacherLabel for a query."""

    name: str
    prompt_version: str                  # cache key; bump on prompt/logic change
    output_version: str = ""             # if set, output goes to <dataset>/<output_version>/<split>.jsonl
    fewshot_source_split: str = ""       # split this teacher's few-shot demos come from, if any
    allow_self_referential: bool = False  # set by teacher/labeler.py from --allow-self-referential-calibration

    @abstractmethod
    def predict(self, example: Example) -> TeacherLabel:
        """Return a routing label + call telemetry for one query."""
