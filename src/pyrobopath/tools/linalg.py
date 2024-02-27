import numpy as np

from pyrobopath.tools.types import R3x3, R4x4, R3

def unit_vector(vec):
    return vec / np.linalg.norm(vec)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


class SO3:
    def __init__(self, matrix: R3x3=np.eye(3)):
        self.data = matrix


class SE3:
    def __init__(self, matrix: R4x4=np.eye(4)):
        self.data = matrix

    @property
    def matrix(self) -> R4x4:
        """The matrix property."""
        return self.data
    
    @matrix.setter
    def matrix(self, value: R4x4):
        self._matrix = value

    @property
    def t(self) -> R3:
        """Translation component of SE(3)"""
        return self.data[:3, 3]
    
    @t.setter
    def t(self, value: R3):
        self.data[:3, 3] = value

