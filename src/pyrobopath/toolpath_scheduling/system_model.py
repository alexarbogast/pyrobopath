from __future__ import annotations
from typing import List, Hashable
from dataclasses import dataclass

from pyrobopath.collision_detection import FCLCollisionModel
from pyrobopath.tools.types import ArrayLike


@dataclass
class AgentModel(object):
    """A data class for storing an agent's (robot's) configuration for toolpath scheduling

    :param capabilities: A list of hashable objects defining the `capabilities`
                         of the agent (e.g. [0, 1] or ["cap1", "cap2"])
    :type capabilities: List[Hashable]
    :param collision_model: The collision model of the agent. (Only supports
                            FCLCollisionModel for now)
    :type collision_model: FCLCollisionModel
    :param base_frame_position: A 3-vector representing the base frame position of the agent
    :type base_frame_position: ArrayLike
    :param home_position: A 3-vector representing the home position of the agent
    :type home_position: ArrayLike
    :param velocity: The velocity for linear movements
    :type velocity: float
    :param travel_velocity: The velocity for rapid linear movements
    :type travel_velocity: float
    """

    capabilities: List[Hashable]
    collision_model: FCLCollisionModel

    base_frame_position: ArrayLike
    home_position: ArrayLike

    velocity: float
    travel_velocity: float
