from __future__ import annotations
import numpy as np
import fcl

from pyrobopath.tools.types import ArrayLike3
from pyrobopath.tools.linalg import SE3, R3, unit_vector
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

    def in_collision(self, other: CollisionModel) -> bool:
        if not isinstance(other, FCLCollisionModel):
            raise NotImplementedError

        this_tf = fcl.Transform(self._transform.R, self._transform.t)
        other_tf = fcl.Transform(other._transform.R, other._transform.t)

        self.obj.setTransform(this_tf)
        other.obj.setTransform(other_tf)

        req = fcl.CollisionRequest(enable_contact=True)
        res = fcl.CollisionResult()
        fcl.collide(self.obj, other.obj, req, res)
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
    """This model rotates about an axis orthogonal to the xy-plane located
    at anchor.

    The anchor point is the base of the robot, and the `translation`
    property is used to set the radial distance of the farthest face.

    :param x: The length of the box. The radial extension of the box towards
              the end-effector from any given position.
    :type x: float
    :param y: The width of the box. The approximate bounding width of the robot
              as viewed from above.
    :type y: float
    :param z: The height of the box. The total height of the box extruded in
              both directions in z from `anchor`.
    :type z: float
    :param anchor: the anchor point about which the box rotates
    :type anchor: np.ndarray
    """

    def __init__(self, x: float, y: float, z: float, anchor: ArrayLike3):
        super().__init__(x, y, z)
        self._anchor = np.array(anchor)
        self._eef_transform = SE3()

    @property
    def translation(self):
        return self._eef_transform.t

    @translation.setter
    def translation(self, value: R3):
        self._eef_transform.t = value

        # find box center location (z_height matches anchor)
        p_tip_anchor = (value - self.anchor)[:2]
        dir = unit_vector(p_tip_anchor)
        box_origin = value[:2] - self.box.side[0] * 0.5 * dir

        # set rotation
        y_axis = np.array([-dir[1], dir[0]])
        self._eef_transform.matrix[:2, 0] = dir
        self._eef_transform.matrix[:2, 1] = y_axis
        self._transform.matrix[:2, 0] = dir
        self._transform.matrix[:2, 1] = y_axis

        self._transform.matrix[:2, 3] = box_origin
        self._transform.matrix[2, 3] = self.anchor[2]

    @property
    def anchor(self):
        return self._anchor

    @anchor.setter
    def anchor(self, value: ArrayLike3):
        self._anchor = value


def _continuous_collision_check(
    model1: FCLCollisionModel,
    trans1_final: np.ndarray,
    model2: FCLCollisionModel,
    trans2_final: np.ndarray,
):
    t1_initial = fcl.Transform(model1._transform[:3, :3], model1._transform[:3, 3])
    t2_initial = fcl.Transform(model2._transform[:3, :3], model2._transform[:3, 3])

    model1.translation = trans1_final
    model2.translation = trans2_final
    t1_final = fcl.Transform(model1._transform[:3, :3], model1._transform[:3, 3])
    t2_final = fcl.Transform(model2._transform[:3, :3], model2._transform[:3, 3])

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
