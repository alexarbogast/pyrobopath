from __future__ import annotations
import numpy as np
import math

from colorama import Fore, Style
from scipy.spatial.transform import Rotation, Slerp

from pyrobopath.tools.types import *


def unit_vector(vec: NDArray) -> NDArray:
    """Returns the vector of unit magnitude in the direction of vec

    :param vec: the vector to normalize
    :type vec: NDArray
    :return: vector in the direction of `vec` with magnitude 1
    :rtype: NDArray
    """
    return vec / np.linalg.norm(vec)


def unit_vector3(vec: NDArray) -> NDArray:
    """A simple unit vector for vectors of length 3

    """
    return vec / norm3(vec)


def norm3(v: ArrayLike3) -> float:
    """A simple vector norm for vectors of length 3

    ~2x as fast as np.linalg.norm()

    :param v: a three dimensional vector
    :type vec: ArrayLike3
    :return: the magnitude of the vector v
    :rtype: float
    """
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def norm2(v: ArrayLike2):  # 2x as fast as np.linalg.norm()
    """A simple vector norm for vectors of length 3

    ~2x as fast as np.linalg.norm()

    :param v: a two dimensional vector
    :type vec: ArrayLike2
    :return: the magnitude of the vector v
    :rtype: float
    """
    return math.sqrt(v[0] * v[0] + v[1] * v[1])


def angle_between(v1: NDArray, v2: NDArray):
    v1_u = unit_vector3(v1)
    v2_u = unit_vector3(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def matrix_slerp(r1: R3x3, r2: R3x3, s: float) -> R3x3:
    """Slerp between two rotation matrices

    :param r1: the start rotation at s = 0
    :type r1: R3x3
    :param r2: the end rotation at s = 1
    :type r2: R3x3
    :param s: The sample parameter in range [0, 1]
    :type s: float
    :return:
    :rtype:
    """
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix([r1, r2]))
    return slerp(s).as_matrix()


def h2e(v: NDArray) -> NDArray:
    """Convert from homogeneous to Euclidean form"""
    if isinstance(v, np.ndarray):
        if len(v.shape) == 2:  # column vector
            return v[:-1, :] / v[-1, :][np.newaxis, :]

        return v[:-1] / v[-1]
    else:
        raise ValueError("bad type")


def e2h(v: NDArray) -> NDArray:
    """Convert from Euclidean to homogeneous form"""
    if isinstance(v, np.ndarray):
        if len(v.shape) == 2:  # column vector
            return np.vstack([v, np.ones((1, v.shape[1]))])

        return np.hstack((v, 1))
    else:
        raise ValueError("bad type")


def rotx(theta: float) -> R3x3:
    """Construct a new SO(3) matrix from a pure X-axis rotation

    :param θ: rotation angle about the X-axis
    :type θ: float
    :return: SO(3) rotation
    :rtype: R3x3
    """
    ct, st = np.cos(theta), np.sin(theta)
    # fmt: off
    R = np.array([[1., 0.,  0.],
                  [0., ct, -st],
                  [0., st,  ct]])
    # fmt: on
    return R


def roty(theta: float) -> R3x3:
    """Construct a new SO(3) matrix from a pure Y-axis rotation

    :param θ: rotation angle about the Y-axis
    :type θ: float
    :return: SO(3) rotation
    :rtype: R3x3
    """
    ct, st = np.cos(theta), np.sin(theta)
    # fmt: off
    R = np.array([[ ct, 0., st],
                  [ 0., 1., 0.],
                  [-st, 0., ct]])
    # fmt: on
    return R


def rotz(theta: float) -> R3x3:
    """Construct a new SO(3) matrix from a pure Z-axis rotation

    :param θ: rotation angle about the Z-axis
    :type θ: float
    :return: SO(3) rotation
    :rtype: R3x3
    """
    ct, st = np.cos(theta), np.sin(theta)
    # fmt: off
    R = np.array([[ct, -st, 0.],
                  [st,  ct, 0.],
                  [0.,  0., 1.]])
    # fmt: on
    return R


class SO3:
    def __init__(self, matrix: R3x3 | None = None):
        if matrix is None:
            matrix = np.eye(3, dtype=np.float64)
        elif matrix.shape != (3, 3):
            raise ValueError("Matrix shape is not 3x3")

        self.data: R3x3 = np.asarray(matrix, dtype=np.float64)

    @property
    def matrix(self) -> R3x3:
        """The matrix property."""
        return self.data

    @matrix.setter
    def matrix(self, value: R3x3):
        self.data = value

    @property
    def N(self) -> int:
        """Dimension of the object's group (superclass property)

        :return: dimension
        :rtype: int

        This corresponds to the dimension of the space, 2D or 3D, to which these
        rotations or rigid-body motions apply.
        """
        return 3

    def __str__(self) -> str:
        out = f"{Style.BRIGHT}{Fore.RED}"
        col_maxes = [max([len(("{:g}").format(x)) for x in col]) for col in self.data.T]
        out_fmt = lambda i, j: ("{:" + str(col_maxes[j]) + "g}").format(self.data[i, j])
        for i in range(3):
            out += "".join([out_fmt(i, j) + "  " for j in range(3)]) + "\n"
        out += f"{Style.RESET_ALL}"
        return out

    def __eq__(self, other) -> bool:
        return bool(np.all(self.data == other.data))

    def __mul__(self, other):
        if type(self) == type(other):
            return self.__class__(self.matrix @ other.matrix)

        if isinstance(other, np.ndarray):
            if other.shape[-1] == self.N:
                return self.data @ other
            raise ValueError("bad operands")
        else:
            return NotImplemented

    def __matmul__(self, other):
        if type(self) == type(other):
            return self.__class__(self.matrix @ other.matrix)
        else:
            raise ValueError("@ only applies to rotation composition")

    def inv(self) -> SO3:
        """
        Find the inverse transformation of the SE(3) instance

        :return: SE(3) inverse
        :rtype: SE3
        """
        inv = self.copy()
        inv.matrix = inv.matrix.T
        return inv

    def interp(self, other: SO3, s: float) -> SO3:
        """Interpolate between SO(3) instances

        :param other: the ending orientation at s = 1.0
        :type other: SO3
        :param s: the interpolation variable s in [0, 1]
        :type s: float
        :returns: the SO(3) instances interpolated at value s
        :rtype: SO(3)
        """
        return self.__class__(matrix_slerp(self.matrix, other.matrix, s))

    def copy(self) -> SO3:
        return SO3(self.data.copy())

    def almost_equal(self, other: SO3, rtol=1e-05, atol=1e-08) -> bool:
        return bool(np.all(np.isclose(self.matrix, other.matrix, rtol=rtol, atol=atol)))

    @classmethod
    def Rx(cls, theta: float) -> SO3:
        """Construct a new SO(3) from X-axis rotation

        :param θ: rotation angle about the X-axis
        :type θ: float or array_like
        :return: SO(3) rotation
        :rtype: SO3 instance
        """
        return cls(rotx(theta))

    @classmethod
    def Ry(cls, theta: float) -> SO3:
        """Construct a new SO(3) from Y-axis rotation

        :param θ: rotation angle about the Y-axis
        :type θ: float or array_like
        :return: SO(3) rotation
        :rtype: SO3 instance
        """
        return cls(roty(theta))

    @classmethod
    def Rz(cls, theta: float) -> SO3:
        """Construct a new SO(3) from Z-axis rotation

        :param θ: rotation angle about the Z-axis
        :type θ: float or array_like
        :return: SO(3) rotation
        :rtype: SO3 instance
        """
        return cls(rotz(theta))

    @classmethod
    def Quaternion(cls, w: float, x: float, y: float, z: float) -> SO3:
        """Construct a new SO(3) instance from Quaternion

        q = [w, x, y, z]

        :param w: the real component
        :type w: float
        :param x: the x component of the imaginary vector
        :type x: float
        :param y: the y component of the imaginary vector
        :type y: float
        :param z: the z component of the imaginary vector
        :type z: float
        :return: SO(3) rotation
        :rtype: SO3 instance
        """
        return cls(Rotation.from_quat([x, y, z, w]).as_matrix())


class SE3:
    """
    The SE3 (Special Euclidean Group) class representes three dimensional poses
    with 4x4 homogeneous transformation matrices belonging to the group SE(3).

    """

    def __init__(self, matrix: R4x4 | None = None):
        if matrix is None:
            matrix = np.eye(4, dtype=np.float64)
        elif matrix.shape != (4, 4):
            raise ValueError("Matrix shape is not 4x4")

        self.data: R4x4 = np.asarray(matrix, dtype=np.float64)

    @property
    def matrix(self) -> R4x4:
        """The homogeneous transformation matrix

        :return: An array view of the 4x4 homogeneous transformation
        :rtype: R4x4
        """
        return self.data

    @matrix.setter
    def matrix(self, value: R4x4):
        self.data = value

    @property
    def t(self) -> R3:
        """Translation component of SE(3)

        :return: An array view of the R3 translation
        :rtype: R3
        """
        return self.data[:3, 3]

    @t.setter
    def t(self, value: R3):
        self.data[:3, 3] = value

    @property
    def R(self) -> R3x3:
        """Rotational component of SE(3)

        :return: A matrix view of the SO(3) rotation
        :rtype: R3x3
        """
        return self.data[:3, :3]

    @R.setter
    def R(self, value: R3x3):
        self.data[:3, :3] = value

    @property
    def N(self) -> int:
        """Dimension of the object's group (superclass property)

        :return: dimension
        :rtype: int

        This corresponds to the dimension of the space, 2D or 3D, to which these
        rotations or rigid-body motions apply.
        """
        return 3

    def __str__(self) -> str:
        out = f"{Style.BRIGHT}"
        col_maxes = [max([len(("{:g}").format(x)) for x in col]) for col in self.data.T]
        out_fmt = lambda i, j: ("{:" + str(col_maxes[j]) + "g}").format(self.data[i, j])
        for i in range(3):
            out += f"{Fore.RED}" + "".join([out_fmt(i, j) + "  " for j in range(3)])
            out += f"{Fore.BLUE}{out_fmt(i, 3)}\n"
        out += (
            f"{Fore.RESET}" + "".join([out_fmt(3, j) + "  " for j in range(4)]) + "\n"
        )
        out += f"{Style.RESET_ALL}"
        return out

    def __eq__(self, other) -> bool:
        return bool(np.all(self.data == other.data))

    def __mul__(self, other):
        if type(self) == type(other):
            return self.__class__(self.matrix @ other.matrix)

        if isinstance(other, np.ndarray):
            if other.shape[-1] == self.N:
                return h2e(self.data @ e2h(other))
            elif other.shape[-1] == self.N + 1:
                return self.data @ other
            raise ValueError("bad operands")
        else:
            return NotImplemented

    def __matmul__(self, other):
        if type(self) == type(other):
            return self.__class__(self.matrix @ other.matrix)
        else:
            raise ValueError("@ only applies to pose composition")

    def almost_equal(self, other: SE3, rtol=1e-05, atol=1e-08) -> bool:
        return bool(np.all(np.isclose(self.matrix, other.matrix, rtol=rtol, atol=atol)))

    def copy(self) -> SE3:
        return SE3(self.data.copy())

    def inv(self) -> SE3:
        """Find the inverse transformation of the SE(3) instance

        :return: SE(3) inverse
        :rtype: SE3
        """
        inv = self.copy()
        inv.R = inv.R.T
        inv.t = -inv.R @ inv.t
        return inv

    def interp(self, other: SE3, s: float) -> SE3:
        """Interpolate between SE(3) instances

        :param other: the ending orientation at s = 1.0
        :type other: SE3
        :param s: the interpolation variable s in [0, 1]
        :type s: float
        :returns: the SE(3) instances interpolated at value s
        :rtype: SE(3)
        """
        new = SE3()
        new.R = matrix_slerp(self.R, other.R, s)
        new.t = self.t * (1 - s) + other.t * s
        return new

    @classmethod
    def Trans(
        cls, x: float | ArrayLike3, y: float | None = None, z: float | None = None
    ) -> SE3:
        """Construct a new SE(3) instance from a pure translation

        :param x: the x value of the translation (if float) or a 3 vector position (if ArrayLike3)
        :type x: float or ArrayLike3
        :param y: the y value of the translation
        :type y: float
        :param z: the z value of the translation
        :type z: float
        :return: SE(3) translation
        :rtype: SE3
        """
        new = cls()
        if y is None and z is None:
            new.t = np.array(x)
            return new

        new.t = np.array([x, y, z])
        return new

    @classmethod
    def Rx(cls, theta: float) -> SE3:
        """Construct a new SE(3) from X-axis rotation

        :param θ: rotation angle about the X-axis
        :type θ: float or array_like
        :return: SE(3) rotation
        :rtype: SE3 instance
        """
        new = cls()
        new.R = rotx(theta)
        return new

    @classmethod
    def Ry(cls, theta: float) -> SE3:
        """Construct a new SE(3) from Y-axis rotation

        :param θ: rotation angle about the Y-axis
        :type θ: float or array_like
        :return: SE(3) rotation
        :rtype: SE3 instance
        """
        new = cls()
        new.R = roty(theta)
        return new

    @classmethod
    def Rz(cls, theta: float) -> SE3:
        """Construct a new SO(3) from Z-axis rotation

        :param θ: rotation angle about the Z-axis
        :type θ: float or array_like
        :return: SE(3) rotation
        :rtype: SE3 instance
        """
        new = cls()
        new.R = rotz(theta)
        return new

    @classmethod
    def Quaternion(cls, w: float, x: float, y: float, z: float) -> SE3:
        """Construct a new SE(3) instance from Quaternion

        q = [w, x, y, z]

        :param w: the real component
        :type w: float
        :param x: the x component of the imaginary vector
        :type x: float
        :param y: the y component of the imaginary vector
        :type y: float
        :param z: the z component of the imaginary vector
        :type z: float
        :return: SE(3) rotation
        :rtype: SE3 instance
        """
        new = cls()
        new.R = Rotation.from_quat([x, y, z, w]).as_matrix()
        return new
