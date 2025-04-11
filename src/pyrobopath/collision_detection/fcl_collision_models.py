from __future__ import annotations
import numpy as np
import quaternion
import fcl

from pyrobopath.tools.types import ArrayLike3, R3
from pyrobopath.tools.linalg import unit_vector2
from pyrobopath.toolpath.path.transform import Transform
from pyrobopath.collision_detection.collision_model import CollisionModel


class FCLCollisionModel(CollisionModel):
    """A collision model using the `python-fcl <pythonfcl>`_ library

    The in_collision function checks model collisions against other
    FCLCollisionModel objects.

    .. _pythonfcl: https://pypi.org/project/python-fcl/
    """

    def __init__(self):
        super(FCLCollisionModel, self).__init__()
        self.obj: fcl.CollisionObject = None
        self._collision_req = fcl.CollisionRequest(enable_contact=True)

    def in_collision(self, other: CollisionModel) -> bool:
        if not isinstance(other, FCLCollisionModel):
            raise NotImplementedError

        q1 = quaternion.as_float_array(self._transform.quat)
        q2 = quaternion.as_float_array(other._transform.quat)

        this_tf = fcl.Transform(q1, self._transform.t)
        other_tf = fcl.Transform(q2, other._transform.t)

        self.obj.setTransform(this_tf)
        other.obj.setTransform(other_tf)

        res = fcl.CollisionResult()
        fcl.collide(self.obj, other.obj, self._collision_req, res)
        return res.is_collision


class FCLBoxCollisionModel(FCLCollisionModel):
    """An fcl box collision model.

    :param x: length of the box
    :type x: float
    :param y: width of the box
    :type y: float
    :param z: height of the box
    :type z: float
    """

    def __init__(self, x: float, y: float, z: float):
        super(FCLBoxCollisionModel, self).__init__()
        self.box = fcl.Box(x, y, z)
        self.obj = fcl.CollisionObject(self.box, fcl.Transform())


class FCLRobotBBCollisionModel(FCLBoxCollisionModel):
    """An FCL collision model for a robot's bounding box with rotation
    about a planar anchor point and customizable center offset.

    This model approximates a robot using a box aligned along the XY plane
    that rotates around a vertical axis passing through `anchor`. The box
    is positioned relative to an end-effector translation, and offset by a
    specified vector to better fit the robot’s actual bounding shape.

    See the pyrobopath documentation for further details.

    :param dims: The dimensions of the box [length (X), width (Y), height (Z)].
    :type dims: ArrayLike3
    :param anchor: The 3D anchor point about which the box rotates. Typically
                   the robot base in world coordinates.
    :type anchor: ArrayLike3
    :param offset: An offset from the end-effector (tip) position to the center
                   of the box in the robot's frame. Allows fine-tuning of the
                   bounding volume shape.
    :type offset: ArrayLike3
    """

    def __init__(
        self,
        dims: ArrayLike3,
        anchor: ArrayLike3 = np.zeros(3),
        offset: ArrayLike3 = np.zeros(3),
    ):
        super().__init__(*dims)
        self._anchor = np.array(anchor)
        self._eef_transform = Transform()

        self._offset = np.array(offset)
        self._box_center_in_eef = offset + np.array([-dims[0] * 0.5, 0.0, 0.0])

    @property
    def translation(self) -> R3:
        return self._eef_transform.t

    @translation.setter
    def translation(self, value: R3):
        p_eef_anchor = value[:2] - self.anchor[:2]
        dir = unit_vector2(p_eef_anchor)

        self._eef_transform = Transform.Rz(np.arctan2(dir[1], dir[0]))
        self._eef_transform.t = value

        self._transform.quat = self._eef_transform.quat
        self._transform.t = self._eef_transform * self._box_center_in_eef

    @property
    def anchor(self):
        return self._anchor

    @anchor.setter
    def anchor(self, value: ArrayLike3):
        self._anchor = value

    @property
    def offset(self):
        return self._offset


def _continuous_collision_check(
    model1: FCLCollisionModel,
    trans1_final: np.ndarray,
    model2: FCLCollisionModel,
    trans2_final: np.ndarray,
) -> bool:
    t1_initial = fcl.Transform(model1.rotation, model1.translation)
    t2_initial = fcl.Transform(model2.rotation, model2.translation)

    model1.translation = trans1_final
    model2.translation = trans2_final
    t1_final = fcl.Transform(model1.rotation, model1.translation)
    t2_final = fcl.Transform(model2.rotation, model2.translation)

    model1.obj.setTransform(t1_initial)
    model2.obj.setTransform(t2_initial)

    request = fcl.ContinuousCollisionRequest()
    request.ccd_motion_type = fcl.CCDMotionType.CCDM_LINEAR
    request.ccd_solver_type = fcl.CCDSolverType.CCDC_CONSERVATIVE_ADVANCEMENT
    request.gjk_solver_type = fcl.GJKSolverType.GST_LIBCCD

    result = fcl.ContinuousCollisionResult()
    ret = fcl.continuousCollide(
        model1.obj, t1_final, model2.obj, t2_final, request, result
    )
    return result.is_collide
