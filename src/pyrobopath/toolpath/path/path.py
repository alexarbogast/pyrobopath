from abc import ABC, abstractmethod
from typing import List
import numpy as np

from pyrobopath.tools.types import ArrayLike3, NDArray
from pyrobopath.tools.linalg import unit_vector

from pyrobopath.toolpath.path.transform import Transform, Rotation
from pyrobopath.toolpath.path.spline import CubicBSpline, CubicBSpline2


class Path(ABC):
    @abstractmethod
    def sample(self, s: float) -> Transform:
        pass

    @abstractmethod
    def length(self) -> float:
        pass


class LinearSegment(Path):
    def __init__(self, start: Transform, end: Transform):
        self.start = start
        self.end = end

    def sample(self, s: float) -> Transform:
        return self.start.interp(self.end, s)

    def length(self) -> float:
        return float(np.linalg.norm(self.start.t - self.end.t))

    def vec(self) -> NDArray:
        return self.end.t - self.start.t

    def unit_vec(self) -> NDArray:
        return unit_vector(self.vec())


class CubicBSplineSegment(Path):
    """
    A uniform cubic B-spline defined on the knot interval [0,1]
    """

    def __init__(
        self,
        control_points: List[ArrayLike3],
        orient_s: Rotation,
        orient_e: Rotation,
    ):
        self.spline = CubicBSpline2(control_points)
        self.orient_s = orient_s
        self.orient_e = orient_e

    def sample(self, s: float) -> Transform:
        return Transform(self.spline(s), self.orient_s.interp(self.orient_e, s).quat)

    def length(self) -> float:
        raise NotImplementedError
