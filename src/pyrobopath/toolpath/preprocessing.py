from __future__ import annotations
from typing import List, Dict
from abc import ABC, abstractmethod
import numpy as np

from pyrobopath.tools.types import *
from .toolpath_core import Toolpath
from .path import Rotation, Transform


class ToolpathPreprocessor:
    def __init__(self):
        self.steps: List[PreprocessingStep] = []

    def process(self, toolpath: Toolpath) -> Toolpath:
        for step in self.steps:
            step.apply(toolpath)
        return toolpath

    def add_step(self, step: PreprocessingStep):
        self.steps.append(step)


class PreprocessingStep(ABC):
    @abstractmethod
    def apply(self, toolpath: Toolpath) -> Toolpath:
        pass


class ScalingStep(PreprocessingStep):
    def __init__(self, scale: float):
        self._scale = scale

    def apply(self, toolpath: Toolpath) -> Toolpath:
        for c in toolpath.contours:
            for p in c.path:
                p *= self._scale
        return toolpath


class TranslateStep(PreprocessingStep):
    def __init__(self, trans: ArrayLike3):
        self._trans = np.array(trans)

    def apply(self, toolpath: Toolpath) -> Toolpath:
        for c in toolpath.contours:
            c.path = list(c.path + self._trans)
        return toolpath


class RotateStep(PreprocessingStep):
    def __init__(self, rot: Rotation):
        self._rot = rot

    def apply(self, toolpath: Toolpath) -> Toolpath:
        for c in toolpath.contours:
            c.path = [self._rot * p for p in c.path]
        return toolpath


class TranformStep(PreprocessingStep):
    def __init__(self, trans: Transform):
        self._trans = trans

    def apply(self, toolpath: Toolpath) -> Toolpath:
        for c in toolpath.contours:
            c.path = [self._trans * p for p in c.path]
        return toolpath


class SubstituteToolStep(PreprocessingStep):
    def __init__(self, tool_map: Dict):
        self._tool_map = tool_map

    def apply(self, toolpath: Toolpath) -> Toolpath:
        for c in toolpath.contours:
            if c.tool in self._tool_map:
                c.tool = self._tool_map[c.tool]
        return toolpath
