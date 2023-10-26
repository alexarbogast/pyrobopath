from __future__ import annotations
from typing import List
import numpy as np

from .utilities import orientation, on_segment


class CollisionModel(object):
    "A collision model is used by a collision checker"

    def __init__(self):
        raise NotImplementedError

    def set_position(self, value):
        raise NotImplementedError

    def in_collision(self, other: CollisionModel) -> bool:
        raise NotImplementedError


class LineCollisionModel(CollisionModel):
    def __init__(self, start, end):
        self._start = start
        self._end = end

    def set_position(self, value):
        self._end = value

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        self._start = value

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        self._end = value

    def in_collision(self, other: LineCollisionModel) -> bool:
        o1 = orientation(self.start, self.end, other.start)
        o2 = orientation(self.start, self.end, other.end)
        o3 = orientation(other.start, other.end, self.start)
        o4 = orientation(other.start, other.end, self.end)

        # General case
        if (o1 != o2) and (o3 != o4):
            return True

        # Special Cases 
        # p1, q1 and p2 are collinear and p2 lies on segment p1q1 
        if ((o1 == 0) and on_segment(self.start, other.start, self.end)): 
            return True
    
        # p1, q1 and q2 are collinear and q2 lies on segment p1q1 
        if ((o2 == 0) and on_segment(self.start, other.end, self.end)): 
            return True
    
        # p2, q2 and p1 are collinear and p1 lies on segment p2q2 
        if ((o3 == 0) and on_segment(other.start, self.start, other.end)): 
            return True
    
        # p2, q2 and q1 are collinear and q1 lies on segment p2q2 
        if ((o4 == 0) and on_segment(other.start, self.end, other.end)): 
            return True
        return False


class LollipopCollisionModel(LineCollisionModel):
    def __init__(self, start, end, radius):
        super().__init__(start, end)
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value

    def in_collision(self, other: LollipopCollisionModel) -> bool:
        if super().in_collision(other):
            return True

        tip_to_tip = np.linalg.norm(self.end - other.end)
        return tip_to_tip < (self.radius + other.radius)


class CollisionGroup(object):
    def __init__(self, models: List[CollisionModel]):
        self.models = models

    def in_collision(self) -> bool:
        for i in range(len(self.models) - 1):
            for j in range(i + 1, len(self.models)):
                col = self.models[i].in_collision(self.models[j])
                if col:
                    return True
        return False


if __name__ == "__main__":
    p1 = np.array([0.0, 0.0])
    q1 = np.array([0.0, 2.0])
    p2 = np.array([1.0, 0.0])
    q2 = np.array([1.0, 2.0])

    model_a = LollipopCollisionModel(p1, q1, 0.5)
    model_b = LollipopCollisionModel(p2, q2, 0.6)

    cg = CollisionGroup([model_a, model_b])
    print(cg.in_collision())

    base_A = np.array([2.0, 0.0])
    base_B = np.array([-2.0, 0.0])

    model_rob1 = LineCollisionModel(base_A, np.zeros(2))
    model_rob2 = LineCollisionModel(base_B, np.zeros(2))
    collision_group = CollisionGroup([model_rob1, model_rob2])
