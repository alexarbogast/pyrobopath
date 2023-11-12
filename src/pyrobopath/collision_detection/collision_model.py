from __future__ import annotations
from typing import List
import numpy as np

from .utilities import orientation, on_segment


class CollisionModel(object):
    "A collision model is used by a collision checker"

    def __init__(self):
        self._transform = np.identity(4)

    @property
    def translation(self):
        return self._transform[:3, 3]
    
    @translation.setter
    def translation(self, value):
        self._transform[:3, 3] = value

    @property
    def rotation(self):
        return self._transform[:3, :3]

    @rotation.setter
    def rotation(self, value):
        self._transform[:3, :3] = value

    @property
    def transform(self):
        return self._transform
    
    @transform.setter
    def transform(self, value):
        self._transform = value

    def in_collision(self, other: CollisionModel) -> bool:
        raise NotImplementedError


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
