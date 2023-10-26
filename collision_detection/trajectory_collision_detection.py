from __future__ import annotations
from typing import List
import numpy as np

from .collision_model import LollipopCollisionModel, CollisionGroup
from .trajectory import Trajectory, TrajectoryPoint


class _TrajectoryStateInterpolator(object):
    def __init__(self, traj: Trajectory, delta_t):
        self.traj = traj
        self.delta_t = delta_t
        self.v = traj.distance() / traj.elapsed()

        # segment specific info
        self.dir = traj.points[1].data - traj.points[0].data
        self.distance = np.linalg.norm(self.dir)
        self.seg_start_point = traj.points[0].data
        self.seg_start_time = traj.start_time()
        self.seg_end_time = traj.points[1].time
        self.segment_idx = 0

        # state
        self.position = traj.points[0].data
        self.time = 0.0
        self.complete = False
        self.started = False

    def step_state(self):
        if self.complete:
            return self.position

        self.time += self.delta_t

        # return start point before reaching first time
        if self.seg_start_time > self.time:
            return self.seg_start_point

        # update when the next segment is reached
        if self.seg_end_time <= self.time:
            self.segment_idx += 1
            # When we reach the last segment, return the final position
            if self.segment_idx + 1 > len(self.traj.points):
                self.complete = True
                self.position = self.traj[-1].data
                return self.position
            self.dir = (
                self.traj[self.segment_idx].data - self.traj[self.segment_idx - 1].data
            )
            self.distance = np.linalg.norm(self.dir)
            self.seg_start_point = self.traj[self.segment_idx - 1].data
            self.seg_start_time = self.traj[self.segment_idx - 1].time
            self.seg_end_time = self.traj[self.segment_idx].time

        elapsed = self.time - self.seg_start_time
        perct = elapsed / (self.seg_end_time - self.seg_start_time)
        self.position = self.seg_start_point + self.dir * perct
        return self.position


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

    traj_interps = [_TrajectoryStateInterpolator(t, delta_t) for t in trajectories]

    completed = False
    while not completed:
        completed = True
        for model_id, traj_interp in enumerate(traj_interps):
            completed = traj_interp.complete and completed
            # step trajectory
            position = traj_interp.step_state()
            group.models[model_id].set_position(position)
        if group.in_collision():
            return True
    return False


if __name__ == "__main__":
    p1 = np.array([-2.0, 0.0])
    q1 = np.array([-1.0, 0.0])
    p2 = np.array([2.0, 0.0])
    q2 = np.array([1.0, 0.0])

    model_a = LollipopCollisionModel(p1, q1, 0.3)
    model_b = LollipopCollisionModel(p2, q2, 0.3)

    collision_group = CollisionGroup([model_a, model_b])

    # collision
    traj_a = Trajectory()
    traj_a.add_traj_point(TrajectoryPoint([-1, 1], 0.0))
    traj_a.add_traj_point(TrajectoryPoint([1, -1], 1.0))

    traj_b = Trajectory()
    traj_b.add_traj_point(TrajectoryPoint([1, 1], 0.5))
    traj_b.add_traj_point(TrajectoryPoint([-1, -1], 1.0))

    trajectories = [traj_a, traj_b]

    threshold = 0.1
    collision = check_trajectory_collision(collision_group, trajectories, threshold)
    print(f"Collision: {collision}")

    # no collision
    traj_a = Trajectory()
    traj_a.add_traj_point(TrajectoryPoint([-1.0, 1.0], 0.0))
    traj_a.add_traj_point(TrajectoryPoint([-1.0, -1.0], 1.0))

    traj_b = Trajectory()
    traj_b.add_traj_point(TrajectoryPoint([1.0, 1.0], 0.5))
    traj_b.add_traj_point(TrajectoryPoint([1.0, -1.0], 1.0))

    trajectories = [traj_a, traj_b]

    threshold = 0.1
    collision = check_trajectory_collision(collision_group, trajectories, threshold)
    print(f"Collision: {collision}")

    # path1 = [[1, 1], [1, -1]]
    # path2 = [[-1, -1], [-1, 1]]
    #
    ## collision-free
    # trajA = Trajectory.from_const_vel_path(path1, 1.0)
    # trajB = Trajectory.from_const_vel_path(path2, 1.0)
    # trajs = [trajA, trajB]
    # collision = check_trajectory_collision(collision_group, trajs, 0.1)
    # self.assertFalse(collision, "Collision-free trajectory returned with collision")
    #
    ## collision
    # trajs = [trajB, trajA]
    # collision = check_trajectory_collision(collision_group, trajs, 0.1)
    # self.assertTrue(collision, "Colliding trajectory returned with collision-free")
