# src/filters/base.py

from abc import ABC
from typing import Optional

class FilterResult:
    def __init__(self, passed, reason="", metrics=None):
        self.passed = passed
        self.reason = reason
        self.metrics = metrics or {}

class BaseFilter(ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    def apply(self, scene_path: str) -> FilterResult:
        """Execute filter on a scene.

        Subclasses may implement apply() directly, or define _apply(scene_path, context).
        """
        raise NotImplementedError("Filter subclasses must implement apply() or _apply()")

    def run(self, scene_path: str, context=None) -> FilterResult:
        if hasattr(self, "_apply"):
            return self._apply(scene_path, context or {})
        return self.apply(scene_path)