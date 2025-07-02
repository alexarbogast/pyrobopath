from __future__ import annotations
from typing import List, Dict, Hashable, Optional
import collections

from pyrobopath.scheduling import Event, Schedule, MultiAgentSchedule
from pyrobopath.collision_detection import Trajectory
from pyrobopath.tools.types import NDArray


class MoveEvent(Event):
    def __init__(self, start, path, velocity):
        self.traj = Trajectory.from_const_vel_path(path, velocity, start)
        self.velocity = velocity
        super(MoveEvent, self).__init__(start, self.traj.end_time(), path)

    def offset(self, t):
        super().offset(t)
        self.traj.offset(t)


class ContourEvent(MoveEvent):
    def __init__(self, start, contour, velocity):
        self.contour = contour
        super(ContourEvent, self).__init__(start, contour.path, velocity)


class ToolpathSchedule(Schedule):
    def __init__(self):
        super().__init__()
        self._events: List[MoveEvent] = []

    # schedule sampling
    def get_state(self, time, default: Optional[NDArray] = None) -> Optional[NDArray]:
        """Samples the state (position) in the toolpath schedule at time"""
        state = default
        if time < self.start_time():
            return state

        for e in self._events:
            if e.end < time:
                state = e.data[-1]  # keep track of current state
                continue

            if e.start <= time:
                return e.traj.get_point_at_time(time).data
            break

        return state


class MultiAgentToolpathSchedule(MultiAgentSchedule):
    def __init__(self):
        super().__init__()
        self.schedules: Dict[Hashable, ToolpathSchedule] = collections.defaultdict(
            ToolpathSchedule
        )

    def add_agent(self, agent: Hashable):
        self.schedules[agent] = ToolpathSchedule()
