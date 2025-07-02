from __future__ import annotations
from typing import List
import numpy as np

from pyrobopath.tools.types import NDArray
from .collision_model import (
    CollisionGroup,
    CollisionModel,
)
from .trajectory import Trajectory


def continuous_collide(
    model1: CollisionModel,
    trans1_final: NDArray,
    model2: CollisionModel,
    trans2_final: NDArray,
    threshold,
) -> bool:
    start1 = model1.translation.copy()
    start2 = model2.translation.copy()

    dir1 = trans1_final - start1
    dir2 = trans2_final - start2
    len1 = np.linalg.norm(dir1)
    len2 = np.linalg.norm(dir2)

    n = int(np.ceil(max(len1, len2) / threshold))
    collision_result = False
    for s in np.linspace(0.0, 1.0, n):
        model1.translation = start1 + dir1 * s
        model2.translation = start2 + dir2 * s
        if model1.in_collision(model2):
            collision_result = True
            break

    # reset model position
    model1.translation = start1
    model2.translation = start2
    return collision_result


class _ConcurrentSegmentIterator:
    """An iterator to loop over concurrent sections of trajectory segments."""

    def __init__(self, trajs: List[Trajectory]):
        unique_times = set()
        for t in trajs:
            times = [p.time for p in t]
            unique_times = unique_times.union(times)
        self.unique_times = sorted(list(unique_times))

        self.trajs = trajs
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx + 1 == len(self.unique_times):
            raise StopIteration

        t0 = self.unique_times[self.idx]
        t1 = self.unique_times[self.idx + 1]
        slices = [t.slice(t0, t1) for t in self.trajs]

        self.idx += 1
        return slices


def trajectory_collision_query(
    model1: CollisionModel,
    traj1: Trajectory,
    model2: CollisionModel,
    traj2: Trajectory,
    threshold: float,
) -> bool:
    """Determine if two models collide along trajectories"""
    for traj_pair in _ConcurrentSegmentIterator([traj1, traj2]):
        model1.translation = traj_pair[0][0].data  # first point
        model2.translation = traj_pair[1][0].data

        p1_final = traj_pair[0][-1].data  # last point
        p2_final = traj_pair[1][-1].data

        if continuous_collide(model1, p1_final, model2, p2_final, threshold):
            return True
    return False


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
) -> bool:
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
