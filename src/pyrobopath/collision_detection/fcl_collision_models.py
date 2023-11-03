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

                          <-----f(p)----->
    axis of rotation -> | +--------------+
                        |/           w  /|
                        /              / |
                       +--------------+  |
                       |:             |. |<-- translation point (p)
                       |:             |  +    end effector position
                       |:           h | /
                       |:             |/
                       +--------------+
                        |
    """

    def __init__(self, width: float, height: float, anchor: np.ndarray):
        super().__init__(0.0, width, height)
        self._anchor = np.array(anchor)
        self._eef_transform = np.identity(4)

    @property
    def translation(self):
        return self._eef_transform[:3, 3]

    @translation.setter
    def translation(self, value):
        self._eef_transform[:3, 3] = value
        # change the length of the box to match the end effectors position
        v = self.translation - self._anchor
        self.box.side = np.array(
            [np.linalg.norm(v), self.box.side[1], self.box.side[2]]
        )

        # find box center location (z_height matches anchor)
        box_origin = self._anchor[:2] + 0.5 * v[:2]
        box_origin = np.concatenate((box_origin, [self.anchor[2]]))
        self._transform[:3, 3] = box_origin.T

        theta = np.arctan2(v[1], v[0])
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        self._transform[:2, :2] = R
        self._eef_transform[:2, :2] = R

    @property
    def anchor(self):
        return self._anchor

    @anchor.setter
    def anchor(self, value):
        self._anchor = value
