from __future__ import annotations
import numpy as np
import quaternion as quat
from colorama import Fore, Style

from pyrobopath.tools.types import *


def rotate_vec(vec: np.ndarray, rot: quat.quaternion):
    vec4 = np.zeros(4)
    vec4[1:] = vec[0:3]
    rot_vec = rot * quat.as_quat_array(vec4) * rot.conjugate()
    return quat.as_vector_part(rot_vec)


class Rotation:
    """A rotation class that implements spatial rotations on members
    of the group SO3 (Special Orthogonal)

    This class internally represents a rotation as a numpy-quaternion. The aim
    is to make interpolation as fast as possible -- as this is used a lot
    throughout pyrobopath

    Class method constructors facilitate constructing rotations with common SO3
    parameterizations (i.e. axis rotations, quaternions, homogeneous
    transformations, Euler angles.)

    :param orient: The rotational component as {w, x, y, z} (default: [1, 0, 0, 0])
    :type orient: Quat
    """

    def __init__(self, orient: Quat | None = None):
        if orient is None:
            self.quat = quat.one
        else:
            self.quat = orient

    @property
    def matrix(self) -> R3x3:
        """The 3x3 rotation matrix

        :return: An array view of the 3x3 rotation matrix
        :rtype: R3x3
        """
        return quat.as_rotation_matrix(self._orient)

    @matrix.setter
    def matrix(self, value: R3x3):
        """Set the transform from a 4x4 homogeneous transformation

        :param value: A 4x4 homogeneous transformation matrix
        :type param: R4x4
        """
        self._orient = quat.from_rotation_matrix(value[:3, :3])

    @property
    def quat(self) -> quat.quaternion:
        """Get the rotation as a quaternion

        :return: The rotation as a quaternion
        :rtype: quat.quaternion
        """
        return self._orient

    @quat.setter
    def quat(self, value: Quat):
        if isinstance(value, quat.quaternion):
            self._orient = value
        elif isinstance(value, np.ndarray) or isinstance(value, list):
            self._orient = quat.as_quat_array(value)
        else:
            raise TypeError(
                "orient should be a numpy quaternion"
                "or an array of length 4 {w, x, y, z}"
            )

    def __str__(self) -> str:
        # fmt: off
        out = f"{Style.BRIGHT}{Fore.RED}"
        col_maxes = [max([len(("{:g}").format(x)) for x in col]) for col in self.matrix.T]
        out_fmt = lambda i, j: ("{:" + str(col_maxes[j]) + "g}").format(self.matrix[i, j])
        # fmt: on
        for i in range(3):
            out += "".join([out_fmt(i, j) + "  " for j in range(3)]) + "\n"
        out += f"{Style.RESET_ALL}"
        return out

    def __eq__(self, other) -> bool:
        return self.quat == other.quat

    def __mul__(self, other):
        if type(self) == type(other):
            return self @ other

        if isinstance(other, np.ndarray):
            return rotate_vec(other, self.quat)
        else:
            return NotImplemented

    def __matmul__(self, other):
        if type(self) == type(other):
            return self.__class__(self.quat * other.quat)
        else:
            raise ValueError("@ only applies to rotation composition")

    def inv(self) -> Rotation:
        """
        Find the inverse transformation of the SE(3) instance

        :return: SE(3) inverse
        :rtype: SE3
        """
        return self.__class__(self.quat.conjugate())

    def interp(self, other: Rotation, s: float) -> Rotation:
        """Interpolate between SO(3) instances

        :param other: the ending orientation at s = 1.0
        :type other: SO3
        :param s: the interpolation variable s in [0, 1]
        :type s: float
        :returns: the SO(3) instances interpolated at value s
        :rtype: SO(3)
        """
        return self.__class__(quat.slerp(self.quat, other.quat, 0.0, 1.0, s))

    def copy(self) -> Rotation:
        return Rotation(self.quat.copy())

    def almost_equal(self, other: Rotation, rtol=1e-05, atol=1e-08) -> bool:
        return quat.isclose(self.quat, other.quat, rtol=rtol, atol=atol)  # type: ignore

    @classmethod
    def Rx(cls, theta: float) -> Rotation:
        """Construct a new SO(3) rotation from X-axis rotation

        :param θ: rotation angle about the X-axis
        :type θ: float or array_like
        :return: SE(3) rotation
        :rtype: SE3 instance
        """
        return cls(quat.from_rotation_vector([theta, 0.0, 0.0]))

    @classmethod
    def Ry(cls, theta: float) -> Rotation:
        """Construct a new SO(3) rotation from Y-axis rotation

        :param θ: rotation angle about the Y-axis
        :type θ: float or array_like
        :return: SE(3) rotation
        :rtype: SE3 instance
        """
        return cls(quat.from_rotation_vector([0.0, theta, 0.0]))

    @classmethod
    def Rz(cls, theta: float) -> Rotation:
        """Construct a new SO(3) rotation from Z-axis rotation

        :param θ: rotation angle about the Z-axis
        :type θ: float or array_like
        :return: SE(3) rotation
        :rtype: SE3 instance
        """
        return cls(quat.from_rotation_vector([0.0, 0.0, theta]))

    @classmethod
    def Quaternion(cls, w: float, x: float, y: float, z: float) -> Rotation:
        """Construct a new SO(3) rotation instance from Quaternion

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
        :rtype: Pose instance
        """
        return cls(quat.as_quat_array([w, x, y, z]))


class Transform:
    """A transformation class that implements spatial transformations on members
    of the group SE3 (Special Euclidean)

    This class internally represents a 3D pose (or transformation) as a
    translation (3D numpy array) and a rotation (numpy-quaternion).

    It provides class methods that facilitate constructing transformation with
    common SE3 parameterizations (i.e. axis rotations, quaternions, homogeneous
    transformations, Euler angles., translations)

    :param trans: The translational component (default: [0, 0, 0])
    :type trans: ArrayLike3
    :param orient: The rotational component as {w, x, y, z} (default: [1, 0, 0, 0])
    :type orient: Quat
    """

    def __init__(self, trans: ArrayLike3 | None = None, orient: Quat | None = None):
        self._trans = np.zeros(3) if trans is None else np.asarray(trans)

        if orient is None:
            self.quat = quat.one
        else:
            self.quat = orient

    @property
    def matrix(self) -> R4x4:
        """The homogeneous transformation matrix

        :return: An array view of the 4x4 homogeneous transformation
        :rtype: R4x4
        """
        matrix = np.eye(4)
        matrix[:3, :3] = quat.as_rotation_matrix(self._orient)
        matrix[:3, 3] = self._trans
        return matrix

    @matrix.setter
    def matrix(self, value: R4x4):
        """Set the transform from a 4x4 homogeneous transformation

        :param value: A 4x4 homogeneous transformation matrix
        :type param: R4x4
        """
        self._orient = quat.from_rotation_matrix(value[:3, :3])
        self._trans = value[:3, 3]

    @property
    def t(self) -> R3:
        """Translation component of SE(3)

        :return: An array view of the R3 translation
        :rtype: R3
        """
        return self._trans

    @t.setter
    def t(self, value: R3):
        """Set the translation of the transformation

        :param value: A numpy array of length 3
        :type param: R3
        """
        self._trans = value

    @property
    def R(self) -> R3x3:
        """Rotational component of SE(3)

        :return: A readonly matrix view of the SO(3) rotation
        :rtype: R3x3
        """
        return quat.as_rotation_matrix(self._orient)

    @R.setter
    def R(self, value: R3x3):
        self._orient = quat.from_rotation_matrix(value)

    @property
    def quat(self) -> quat.quaternion:
        """Get the rotational component of the tranformation as a quaternion

        :return: The rotation of the transformation as a quaternion
        :rtype: Quat
        """
        return self._orient

    @quat.setter
    def quat(self, value: Quat):
        if isinstance(value, quat.quaternion):
            self._orient = value
        elif isinstance(value, np.ndarray) or isinstance(value, list):
            self._orient = quat.as_quat_array(value)
        else:
            raise TypeError(
                "orient should be a numpy quaternion"
                "or an array of length 4 {w, x, y, z}"
            )

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
        # fmt: off
        out = f"{Style.BRIGHT}"
        col_maxes = [max([len(("{:g}").format(x)) for x in col]) for col in self.matrix.T]
        out_fmt = lambda i, j: ("{:" + str(col_maxes[j]) + "g}").format(self.matrix[i, j])
        for i in range(3):
            out += f"{Fore.RED}" + "".join([out_fmt(i, j) + "  " for j in range(3)])
            out += f"{Fore.BLUE}{out_fmt(i, 3)}\n"
        out += (f"{Fore.RESET}" + "".join([out_fmt(3, j) + "  " for j in range(4)]) + "\n")
        out += f"{Style.RESET_ALL}"
        # fmt: on
        return out

    def __eq__(self, other) -> bool:
        return np.all(self.t == other.t) and self.quat == other.quat  # type: ignore

    def __mul__(self, other):
        if type(self) == type(other):
            return self @ other

        if isinstance(other, np.ndarray):
            return rotate_vec(other, self.quat) + self.t
        else:
            return NotImplemented

    def __matmul__(self, other):
        if type(self) == type(other):
            trans = rotate_vec(other.t, self.quat) + self.t
            orient = self.quat * other.quat
            return self.__class__(trans, orient)
        else:
            raise ValueError("@ only applies to transform composition")

    def almost_equal(self, other: Transform, rtol=1e-05, atol=1e-08) -> bool:
        same_trans = np.isclose(self.t, other.t, rtol=rtol, atol=atol)
        same_orient = quat.isclose(self.quat, other.quat, rtol=rtol, atol=atol)  # type: ignore
        return bool(np.all(same_trans) and np.all(same_orient))

    def copy(self) -> Transform:
        return Transform(self._trans.copy(), self._orient.copy())

    def inv(self) -> Transform:
        """Find the inverse transformation of the SE(3) instance

        :return: SE(3) inverse
        :rtype: SE3
        """
        inv = self.copy()
        inv.quat = self.quat.conjugate()
        inv.t = rotate_vec(-self.t, inv.quat)
        return inv

    def interp(self, other: Transform, s: float) -> Transform:
        """Interpolate between SE(3) instances

        :param other: the ending orientation at s = 1.0
        :type other: SE3
        :param s: the interpolation variable s in [0, 1]
        :type s: float
        :returns: the SE(3) instances interpolated at value s
        :rtype: SE(3)
        """
        trans = self.t * (1 - s) + other.t * s
        orient = quat.slerp(self._orient, other._orient, 0, 1, s)
        return Transform(trans, orient)

    @classmethod
    def Trans(
        cls, x: float | ArrayLike3, y: float | None = None, z: float | None = None
    ) -> Transform:
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
        if y is None and z is None:
            return cls(np.array(x))
        return cls(np.array([x, y, z]))

    @classmethod
    def Rx(cls, theta: float) -> Transform:
        """Construct a new SE(3) transformation from X-axis rotation

        :param θ: rotation angle about the X-axis
        :type θ: float or array_like
        :return: SE(3) rotation
        :rtype: SE3 instance
        """
        return cls(orient=quat.from_rotation_vector([theta, 0.0, 0.0]))

    @classmethod
    def Ry(cls, theta: float) -> Transform:
        """Construct a new SE(3) transformation from Y-axis rotation

        :param θ: rotation angle about the Y-axis
        :type θ: float or array_like
        :return: SE(3) rotation
        :rtype: SE3 instance
        """
        return cls(orient=quat.from_rotation_vector([0.0, theta, 0.0]))

    @classmethod
    def Rz(cls, theta: float) -> Transform:
        """Construct a new SE(3) transformation from Z-axis rotation

        :param θ: rotation angle about the Z-axis
        :type θ: float or array_like
        :return: SE(3) rotation
        :rtype: SE3 instance
        """
        return cls(orient=quat.from_rotation_vector([0.0, 0.0, theta]))

    @classmethod
    def Quaternion(cls, w: float, x: float, y: float, z: float) -> Transform:
        """Construct a new SE(3) transformation instance from Quaternion

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
        :rtype: Pose instance
        """
        return cls(orient=[w, x, y, z])
