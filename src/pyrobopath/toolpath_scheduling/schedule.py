from __future__ import annotations
from typing import List, Hashable
import collections

from ..scheduling import Event, Schedule, MultiAgentSchedule
from ..collision_detection import Trajectory
from ..toolpath import Contour


class ContourEvent(Event):
    def __init__(self, contour: Contour, start: float, velocity: float):
        duration = contour.path_length() / velocity
        super().__init__(contour, start, duration)
        self.velocity = velocity
        self.traj = Trajectory.from_const_vel_path(contour.path, velocity, start)


class ToolpathSchedule(Schedule):
    def __init__(self):
        super().__init__()
        self._events: List[ContourEvent] = []

    # schedule sampling
    def get_state(self, time, default=None):
        """Samples the state (position) in the toolpath schedule at time"""
        state = default
        if time < self.start_time():
            return state
        
        for e in self._events:
            event_end = e.start + e.duration
            if event_end < time:
                state = e.data.path[-1] # keep track of current state
                continue

            if e.start <= time:
                return e.traj.get_point_at_time(time).data
            break

        return state
    
class MultiAgentToolpathSchedule(MultiAgentSchedule):
    def __init__(self):
        super().__init__()
        self.schedules = collections.defaultdict(ToolpathSchedule)

    def add_agent(self, agent: Hashable):
        self.schedules[agent] = ToolpathSchedule()
