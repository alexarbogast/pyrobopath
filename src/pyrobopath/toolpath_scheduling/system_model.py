from __future__ import annotations
from typing import Hashable
import numpy as np

from pyrobopath.collision_detection import FCLCollisionModel


class AgentModel(object):
    capabilities: Hashable
    collision_model : FCLCollisionModel
    
    base_frame_position: np.ndarray = np.zeros(3)
    home_position: np.ndarray = np.zeros(3)
