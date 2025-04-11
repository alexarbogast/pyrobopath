from __future__ import annotations
import numpy as np
import math

from pyrobopath.tools.types import *


def unit_vector(vec: NDArray) -> NDArray:
    """Returns the vector of unit magnitude in the direction of vec

    :param vec: the vector to normalize
    :type vec: NDArray
    :return: vector in the direction of `vec` with magnitude 1
    :rtype: NDArray
    :raises ValueError: if the input vector has zero magnitude
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector")
    return vec / norm


def unit_vector3(vec: NDArray) -> NDArray:
    """A simple unit vector for vectors of length 3"""
    return vec / norm3(vec)


def unit_vector2(vec: NDArray) -> NDArray:
    """A simple unit vector for vectors of length 3"""
    return vec / norm2(vec)


def norm3(v: ArrayLike3) -> float:
    """A simple vector norm for vectors of length 3

    ~2x as fast as np.linalg.norm()

    :param v: a three dimensional vector
    :type vec: ArrayLike3
    :return: the magnitude of the vector v
    :rtype: float
    """
    radicand = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    if radicand == 0:
        raise ValueError("Cannot normalize a zero vector")
    return math.sqrt(radicand)


def norm2(v: ArrayLike2) -> float:  # 2x as fast as np.linalg.norm()
    """A simple vector norm for vectors of length 3

    ~2x as fast as np.linalg.norm()

    :param v: a two dimensional vector
    :type vec: ArrayLike2
    :return: the magnitude of the vector v
    :rtype: float
    """
    radicand = v[0] * v[0] + v[1] * v[1]
    if radicand == 0:
        raise ValueError("Cannot normalize a zero vector")
    return math.sqrt(radicand)


def angle_between(v1: NDArray, v2: NDArray) -> float:
    """Compute the angle between two 3D vectors

    Uses the arccosine of the dot product of unit vectors

    :param v1: the first 3D vector
    :type v1: NDArray
    :param v2: the second 3D vector
    :type v2: NDArray
    :return: the angle between v1 and v2 in radians
    :rtype: float
    """
    v1_u = unit_vector3(v1)
    v2_u = unit_vector3(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
