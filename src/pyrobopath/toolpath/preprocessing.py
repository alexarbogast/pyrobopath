from __future__ import annotations
from typing import List, Dict
from abc import ABC, abstractmethod
import numpy as np

from pyrobopath.tools.types import *
from pyrobopath.tools.geometry import segment_path
from .toolpath_core import Toolpath, Contour
from .path import Rotation, Transform


class ToolpathPreprocessor:
    """
    A container for a sequence of toolpath preprocessing steps.

    Methods
    -------
    process(toolpath):
        Applies all registered preprocessing steps to the given toolpath.
    add_step(step):
        Adds a preprocessing step to the pipeline.
    """

    def __init__(self):
        self.steps: List[PreprocessingStep] = []

    def process(self, toolpath: Toolpath) -> Toolpath:
        for step in self.steps:
            step.apply(toolpath)
        return toolpath

    def add_step(self, step: PreprocessingStep):
        self.steps.append(step)


class PreprocessingStep(ABC):
    """
    Abstract base class for toolpath preprocessing steps.
    """

    @abstractmethod
    def apply(self, toolpath: Toolpath) -> Toolpath:
        """
        Apply the preprocessing step to the given toolpath.

        Parameters
        ----------
        toolpath : Toolpath
            The toolpath to process.

        Returns
        -------
        Toolpath
            The processed toolpath.
        """
        pass


class ScalingStep(PreprocessingStep):
    """
    A preprocessing step that uniformly scales all path points.

    Parameters
    ----------
    scale : float
        The uniform scale factor to apply to each point.
    """

    def __init__(self, scale: float):
        self._scale = scale

    def apply(self, toolpath: Toolpath) -> Toolpath:
        for c in toolpath.contours:
            for p in c.path:
                p *= self._scale
        return toolpath


class TranslateStep(PreprocessingStep):
    """
    A preprocessing step that translates all path points.

    Parameters
    ----------
    trans : ArrayLike3
        The translation vector to add to each point.
    """

    def __init__(self, trans: ArrayLike3):
        self._trans = np.array(trans)

    def apply(self, toolpath: Toolpath) -> Toolpath:
        for c in toolpath.contours:
            c.path = list(c.path + self._trans)
        return toolpath


class RotateStep(PreprocessingStep):
    """
    A preprocessing step that rotates all path points.

    Parameters
    ----------
    rot : Rotation
        The rotation to apply to each point.
    """

    def __init__(self, rot: Rotation):
        self._rot = rot

    def apply(self, toolpath: Toolpath) -> Toolpath:
        for c in toolpath.contours:
            c.path = [self._rot * p for p in c.path]
        return toolpath


class TranformStep(PreprocessingStep):
    """
    A preprocessing step that applies a general transform to all path points.

    Parameters
    ----------
    trans : Transform
        The transformation to apply to each point.
    """

    def __init__(self, trans: Transform):
        self._trans = trans

    def apply(self, toolpath: Toolpath) -> Toolpath:
        for c in toolpath.contours:
            c.path = [self._trans * p for p in c.path]
        return toolpath


class SubstituteToolStep(PreprocessingStep):
    """
    A preprocessing step that replaces tool types in contours using a mapping.

    Parameters
    ----------
    tool_map : dict
        A dictionary mapping existing tool names to new tool names.
    """

    def __init__(self, tool_map: Dict):
        self._tool_map = tool_map

    def apply(self, toolpath: Toolpath) -> Toolpath:
        for c in toolpath.contours:
            if c.tool in self._tool_map:
                c.tool = self._tool_map[c.tool]
        return toolpath


class MaxContourLengthStep(PreprocessingStep):
    """
    A preprocessing step that splits contours into smaller segments
    if their total length exceeds a specified maximum.

    Parameters
    ----------
    length : float
        The maximum allowed length for any single contour.
    """

    def __init__(self, length: float):
        self._length = length

    def apply(self, toolpath: Toolpath) -> Toolpath:
        contours = []
        for c in toolpath.contours:
            seg_paths = segment_path(c.path, self._length)
            for p in seg_paths:
                contours.append(Contour(p, tool=c.tool))
        toolpath.contours = contours
        return toolpath
