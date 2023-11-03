from __future__ import annotations
from typing import List
import numpy as np

from .collision_model import LollipopCollisionModel, CollisionGroup
from .trajectory import Trajectory, TrajectoryPoint


class _TrajectoryStateInterpolator(object):
    """Interpolate a trajectory by stepping the state from start_time
    to the latest time in the trajectory."""

    def __init__(self, traj: Trajectory, delta_t, start_time=0.0):
        self.traj = traj
        self.delta_t = delta_t
        self.start_time = start_time
        self.reset()

    def reset(self):
        self.current_start_point = self.traj[0]
        self.current_end_point = self.traj[1]
        self.time = self.start_time
        self.segment_idx = 1
        self.complete = False

    def step_state(self):
        # return last state after reaching final time
        if self.complete:
            return self.traj[-1]

        self.time += self.delta_t

        # return start state before reaching initial time
        if self.current_start_point.time > self.time:
            return self.current_start_point

        # update current segment
        if self.current_end_point.time <= self.time:
            self.segment_idx += 1
            if self.segment_idx + 1 > len(self.traj.points):
                self.complete = True
                return self.traj[-1]
            self.current_start_point = self.current_end_point
            self.current_end_point = self.traj[self.segment_idx]

        elapsed = self.time - self.current_start_point.time
        s = elapsed / (self.current_end_point.time - self.current_start_point.time)
        state = self.current_start_point.interp(self.current_end_point, s)
        return state

    def interp_state(self, t):
        """Find the trajectory state at time t"""

        # use interp functions in trajectory points
        pass


def check_trajectory_collision(
    group: CollisionGroup, trajectories: List[Trajectory], threshold: float
):
    start_time, end_time = 0.0, 0.0
    for traj in trajectories:
        start_time = min(start_time, traj.start_time())
        end_time = max(end_time, traj.end_time())

    # find trajectory with the fastest velocity
    # this trajectory defines the time step to ensure distance 'threshold'
    max_vel_idx = np.argmin([t.distance() / t.elapsed() for t in trajectories])
    n_steps = trajectories[max_vel_idx].distance() / threshold
    delta_t = trajectories[max_vel_idx].elapsed() / n_steps

    traj_interps = [
        _TrajectoryStateInterpolator(t, delta_t, start_time) for t in trajectories
    ]

    completed = False
    while not completed:
        completed = True
        for model_id, traj_interp in enumerate(traj_interps):
            completed = traj_interp.complete and completed
            # step trajectory
            state = traj_interp.step_state()
            group.models[model_id].translation = state.data
        if group.in_collision():
            return True
    return False
