from __future__ import annotations
from typing import Sequence
import numpy as np
from scipy import interpolate

from pyrobopath.tools.types import ArrayLike, NDArray


class BSpline:
    def __init__(self, control_points: Sequence[ArrayLike], knot_vector, degree):
        self.spline = interpolate.BSpline(
            knot_vector, control_points, degree, extrapolate=False
        )

    def __call__(self, ui) -> NDArray:
        return self.spline(ui)


class CubicBSpline(BSpline):
    KNOTS = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)

    def __init__(
        self, control_points: Sequence[ArrayLike], knots: ArrayLike | None=None
    ):
        self.knots = CubicBSpline.KNOTS if knots is None else knots
        super(CubicBSpline, self).__init__(control_points, self.knots, 3)

class CubicBSpline2(BSpline):
    # fmt: off
    BASIS = np.array([[ 1, 4, 1, 0],
                      [-3, 0, 3, 0],
                      [ 3,-6, 3, 0],
                      [-1, 3,-3, 1]])
    # fmt: on

    def __init__(self, control_points: Sequence[ArrayLike]):
        self.c = np.asarray(control_points)

    def __call__(self, ui):
        return self.__basis(ui) @ self.c

    def __basis(self, u):
        v_t = np.array([1, u, u * u, u**3])
        return 1 / 6 * v_t @ CubicBSpline2.BASIS
