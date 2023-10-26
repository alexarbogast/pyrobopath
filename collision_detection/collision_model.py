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
        p1, q1 = np.round(self.start, 5), np.round(self.end, 5)
        p2, q2 = np.round(other.start, 5), np.round(other.end, 5)

        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        # General case
        if (o1 != o2) and (o3 != o4):
            return True

        # Special Cases
        # p1, q1 and p2 are collinear and p2 lies on segment p1q1
        if (o1 == 0) and on_segment(p1, p2, q1):
            return True

        # p1, q1 and q2 are collinear and q2 lies on segment p1q1
        if (o2 == 0) and on_segment(p1, q2, q1):
            return True

        # p2, q2 and p1 are collinear and p1 lies on segment p2q2
        if (o3 == 0) and on_segment(p2, p1, q2):
            return True

        # p2, q2 and q1 are collinear and q1 lies on segment p2q2
        if (o4 == 0) and on_segment(p2, q1, q2):
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
