from __future__ import annotations
from typing import List
from abc import ABC, abstractmethod

from pyrobopath.tools.types import R3, R3x3
from pyrobopath.toolpath.path.transform import Transform


class CollisionModel(ABC):
    "A collision model is used by a collision checker"

    def __init__(self):
        self._transform = Transform()

    @property
    def translation(self) -> R3:
        return self._transform.t

    @translation.setter
    def translation(self, value: R3):
        self._transform.t = value

    @property
    def rotation(self) -> R3x3:
        return self._transform.R

    @rotation.setter
    def rotation(self, value: R3x3):
        self._transform.R = value

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value: Transform):
        self._transform = value

    @abstractmethod
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
