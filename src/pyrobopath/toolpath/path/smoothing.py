from typing import List, Sequence
import numpy as np

from pyrobopath.tools.types import ArrayLike3, NDArray
from pyrobopath.tools.linalg import unit_vector3, angle_between
from pyrobopath.toolpath.path import Path, CubicBSplineSegment
from pyrobopath.toolpath.path.transform import Rotation, Transform


def _continuous_cubic_splines(
    control_points: List[ArrayLike3],
) -> List[CubicBSplineSegment]:
    v = control_points
    return [
        CubicBSplineSegment(v[i : (i + 4)], Rotation(), Rotation())
        for i in range(len(v) - 3)
    ]


def _entry_length(p0: NDArray, p1: NDArray, p2: NDArray, tol: float) -> float:
    return 3 * tol / np.cos(angle_between(p0 - p1, p2 - p1) / 2)


def smooth_cubic_bspline(path: List[Transform], tol: float) -> Sequence[Path]:
    """Smooth a piecewise linear path within tolerance `tol` using cubic splines

    :param path: a list of SE(3) poses representing piecewise linear paths
    :type path: List[SE3]
    :param tol: the maximum distance allowed between the generated curves and
        the corner points of the path
    :type tol: float
    :return: a list of G2 continuous B-spline paths
    :rtype: Sequence[Path]
    """
    # blend first segment
    p0 = path[0].t
    p1 = path[1].t
    p2 = path[2].t

    ls_p1 = _entry_length(p0, p1, p2, tol)

    if np.linalg.norm(p1 - p0) > ls_p1:
        v1 = p0
        v3 = p1
        v2 = p1 + ls_p1 * unit_vector3(p0 - p1)
        v0 = 2 * v1 - v2
        v = [v0, v1, v2, v3]
    else:
        v1 = p0
        v2 = p1
        v0 = 2 * v1 - v2
        v = [v0, v1, v2]

    # blend corners
    for i in range(1, len(path) - 2):
        pk_m1 = path[i - 1].t
        pk = path[i].t
        pk_p1 = path[i + 1].t
        pk_p2 = path[i + 2].t

        dist = np.linalg.norm(pk_p1 - pk)

        l_pk = _entry_length(pk_m1, pk, pk_p1, tol)
        l_pk_p1 = _entry_length(pk, pk_p1, pk_p2, tol)
        uk_kp1 = unit_vector3(pk_p1 - pk)

        if dist > (l_pk + l_pk_p1):
            vi = pk + uk_kp1 * l_pk
            vi_p1 = pk_p1 - uk_kp1 * l_pk_p1
            vi_p2 = pk_p1
            v.extend(([vi, vi_p1, vi_p2]))
        elif min(l_pk, l_pk_p1) < dist <= (l_pk + l_pk_p1):
            le_pk = l_pk / (l_pk + l_pk_p1) * dist
            vi = pk + uk_kp1 * le_pk
            vi_p1 = pk_p1
            v.extend([vi, vi_p1])
        else:
            v.append(pk_p1)

    # blend last segment
    l_p1 = _entry_length(path[-3].t, path[-2].t, path[-1].t, tol)

    if np.linalg.norm(path[-1].t - path[-2].t) > l_p1:
        v2 = path[-2].t + l_p1 * unit_vector3(path[-1].t - path[-2].t)
        v.append(v2)
    else:
        v2 = path[-2].t
    v1 = path[-1].t
    v0 = 2 * v1 - v2
    v.extend([v1, v0])

    return _continuous_cubic_splines(v)
