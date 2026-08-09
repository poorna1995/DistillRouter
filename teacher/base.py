"""Base class and registry for teacher models.

Same pattern as dataset/base.py: adding a new teacher backend (an
API-backed LLM, a local model, ...) means writing one new module here with
a `TeacherModel` subclass decorated with `@register` — run.py and the
labeling pipeline discover it by name, nothing else changes.
"""
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
    """One subclass = one way of producing a TeacherLabel for a query.

    `prompt_version` is part of the cache key in teacher/cache.py — bump it
    whenever a subclass's labeling logic/prompt changes, so cached labels
    from the old version are treated as stale rather than silently reused.
    """

    name: str
    prompt_version: str

    @abstractmethod
    def predict(self, example: Example) -> TeacherLabel:
        """Return a routing label + soft distribution + call telemetry for one query."""
