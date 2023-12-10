from __future__ import annotations
from typing import List, Hashable
import numpy as np
from dataclasses import dataclass

from pyrobopath.collision_detection import FCLCollisionModel

@dataclass
class AgentModel(object):
    capabilities: List[Hashable] = None
    collision_model : FCLCollisionModel = None
    
    base_frame_position: np.ndarray = np.zeros(3)
    home_position: np.ndarray = np.zeros(3)

    velocity: float = None
    travel_velocity: float = None

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = [0]
        self.capabilities = list(self.capabilities)
