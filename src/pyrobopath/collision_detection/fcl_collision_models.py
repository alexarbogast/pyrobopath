from __future__ import annotations
import numpy as np
import fcl

from .collision_model import CollisionModel


class FCLCollisionModel(CollisionModel):
    def __init__(self):
        super().__init__()
        self.obj = None

    def in_collision(self, other: FCLCollisionModel):
        this_tf = fcl.Transform(self._transform[:3, :3], self._transform[:3, 3])
        other_tf = fcl.Transform(other._transform[:3, :3], other._transform[:3, 3])

        self.obj.setTransform(this_tf)
        other.obj.setTransform(other_tf)

        req = fcl.CollisionRequest(enable_contact=True)
        res = fcl.CollisionResult()
        n_contacts = fcl.collide(self.obj, other.obj, req, res)
        return res.is_collision


class FCLBoxCollisionModel(FCLCollisionModel):
    def __init__(self, x: float, y: float, z: float):
        super().__init__()
        self.box = fcl.Box(x, y, z)
        self.obj = fcl.CollisionObject(self.box, fcl.Transform())


class FCLRobotBBCollisionModel(FCLBoxCollisionModel):
    """This model rotates about an axis orthogonal to the xy-plane.

    The anchor point is the base of the robot, and the `translation`
    property is used to set the radial distance of the farthest face.

                               <----- l ------>
    axis of rotation -> |      +--------------+
        (anchor)        |     /           w  /|
                        |    /              / |
                        |   +--------------+  |
                        |-- |              |. |<-- translation point (p)
                        |   |              |  +    end effector position
                        |   |            h | /
                        |   |              |/
                        |   +--------------+
                        |
    """

    def __init__(self, x: float, y: float, z: float, anchor: np.ndarray):
        super().__init__(x, y, z)
        self._anchor = np.array(anchor)
        self._eef_transform = np.identity(4)

    @property
    def translation(self):
        return self._eef_transform[:3, 3]

    @translation.setter
    def translation(self, value):
        value = np.array(value)
        self._eef_transform[:3, 3] = value

        # find box center location (z_height matches anchor)
        p_tip_anchor = value - self.anchor
        dir = p_tip_anchor[:2] / np.linalg.norm(p_tip_anchor[:2])
        box_origin = value[:2] - self.box.side[0] * 0.5 * dir

        x = dir
        y = np.array([-dir[1], dir[0]])
        R = np.array([x, y]).T
        self._eef_transform[:2, :2] = R
        self._transform[:2, :2] = R

        box_origin = np.concatenate((box_origin, [self.anchor[2]]))
        self._transform[:3, 3] = box_origin


    @property
    def anchor(self):
        return self._anchor

    @anchor.setter
    def anchor(self, value):
        self._anchor = value


def continuous_collision_check(
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
