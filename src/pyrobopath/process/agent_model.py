from __future__ import annotations
from typing import List, Hashable
from dataclasses import dataclass

from pyrobopath.collision_detection import FCLCollisionModel
from pyrobopath.tools.types import ArrayLike


@dataclass
class AgentModel(object):
    """
    A data class for storing an agent's configuration for toolpath scheduling

    Parameters
    ----------
    capabilities : list of Hashable
        Capabilities the agent supports (e.g. tools indices)
    collision_model : FCLCollisionModel
        The agent's spatial collision model. Currently, only FCL-based models
        are supported.
    base_frame_position : ArrayLike
        The origin of the agent's frame.
    home_position : ArrayLike
         Resting or return-to (default) position of the agent.
    velocity : float
        Standard linear motion speed used to execute tasks.
    travel_velocity : float
        Rapid (non-operational) motion speed used to travel between tasks.
    """

    capabilities: List[Hashable]
    collision_model: FCLCollisionModel

    base_frame_position: ArrayLike
    home_position: ArrayLike

    velocity: float
    travel_velocity: float
