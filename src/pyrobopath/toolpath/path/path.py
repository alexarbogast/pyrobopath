from abc import ABC, abstractmethod
from typing import List
import numpy as np

from pyrobopath.tools.types import ArrayLike3, NDArray
from pyrobopath.tools.linalg import unit_vector, SE3, SO3

from pyrobopath.toolpath.path.spline import CubicBSpline, CubicBSpline2


class Path(ABC):
    @abstractmethod
    def sample(self, s: float) -> SE3:
        pass

    @abstractmethod
    def length(self) -> float:
        pass


class LinearSegment(Path):
    def __init__(self, start: SE3, end: SE3):
        self.start = start
        self.end = end

    def sample(self, s: float) -> SE3:
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
        orient_s: SO3,
        orient_e: SO3,
    ):
        self.spline = CubicBSpline2(control_points)
        self.orient_s = orient_s
        self.orient_e = orient_e

    def sample(self, s: float) -> SE3:
        sample = SE3.Trans(self.spline(s))
        sample.R = self.orient_s.interp(self.orient_e, s).matrix
        return sample

    def length(self) -> float:
        raise NotImplementedError
