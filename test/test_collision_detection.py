import unittest
import numpy as np
from copy import copy

from pyrobopath.collision_detection import (
    LineCollisionModel,
    LollipopCollisionModel,
    CollisionGroup,
)
from pyrobopath.collision_detection import Trajectory, TrajectoryPoint, check_trajectory_collision


class TestCollisionDetection(unittest.TestCase):
    def test_const_vel_traj(self):
        vel = 1.0
        start_time = 1

        path = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
        traj = Trajectory.from_const_vel_path(path, vel, start_time)
        for i, point in enumerate(traj):
            self.assertEqual(
                point.time,
                i * vel + start_time,
                "Const velocity trajectory was created with incorrect time",
            )
        self.assertEqual(traj.distance(), 3.0)

    def test_trajectory(self):
        traj = Trajectory()
        traj.add_traj_point(TrajectoryPoint([-1.0, 1.0, 0.0], -20.0))
        traj.add_traj_point(TrajectoryPoint([1.0, 0.0, -1.0], 40.0))

        self.assertEqual(traj.start_time(), -20.0, "Start time != -20.0")
        self.assertEqual(traj.end_time(), 40.0, "End time != 40.0")
        self.assertEqual(traj.elapsed(), 60.0, "Elapsed != 60.0")

    def test_collision_group(self):
        base_A = np.array([-1.0, -1.0])
        base_B = np.array([1.0, -1.0])
        base_C = np.array([0.0, 1.0])

        model_A = LollipopCollisionModel(base_A, np.zeros(2), 0.1)
        model_B = LollipopCollisionModel(base_B, np.zeros(2), 0.1)
        model_C = LollipopCollisionModel(base_C, np.zeros(2), 0.1)
        collision_group = CollisionGroup([model_A, model_B, model_C])

        # no collisions
        model_A.set_position(np.array([-1.0, 0.0]))
        model_B.set_position(np.array([1.0, 0.0]))
        model_C.set_position(np.array([1.0, 1.0]))
        self.assertFalse(
            collision_group.in_collision(),
            "Collision-free state returned with collision",
        )

        # one collision (A & B)
        model_A.set_position(np.array([0.0, 0.0]))
        model_B.set_position(np.array([0.0, 0.0]))
        model_C.set_position(np.array([1.0, 1.0]))
        self.assertTrue(
            collision_group.in_collision(),
            "Collision state returned with collision-free",
        )

        # all colliding
        model_A.set_position(np.array([1.0, 1.0]))
        model_B.set_position(np.array([-1.0, -1.0]))
        model_C.set_position(np.array([0.0, -1.0]))
        self.assertTrue(
            collision_group.in_collision(),
            "Collision state returned with collision-free",
        )

    def test_line_collision_model(self):
        base_A = np.array([-2.0, 0.0])
        base_B = np.array([2.0, 0.0])

        model_A = LineCollisionModel(base_A, np.zeros(2))
        model_B = LineCollisionModel(base_B, np.zeros(2))

        # no collision
        model_A.set_position([-1.0, 0.0])
        model_B.set_position([1.0, 0.0])
        self.assertFalse(
            model_A.in_collision(model_B),
            "Collision-free state returned with collision",
        )

        # collision
        model_A.set_position([1.0, 1.0])
        model_B.set_position([-1.0, 1.0])
        self.assertTrue(
            model_A.in_collision(model_B),
            "Collision state returned with collision-free",
        )

        # colinear collision
        model_A.set_position([1.0, 0.0])
        model_B.set_position([-1.0, 0.0])
        self.assertTrue(
            model_A.in_collision(model_B),
            "Collision state returned with collision-free",
        )

    def test_lollipop_collision_model(self):
        base_A = np.array([-2.0, 0.0])
        base_B = np.array([2.0, 0.0])

        model_A = LollipopCollisionModel(base_A, np.zeros(2), 0.25)
        model_B = LollipopCollisionModel(base_B, np.zeros(2), 0.25)

        # collision
        model_A.set_position(np.array([1.0, 1.0]))
        model_B.set_position(np.array([-1.0, 1.0]))
        self.assertTrue(
            model_A.in_collision(model_B),
            "Collision state returned with collision-free",
        )

        # no collision
        model_A.set_position(np.array([-0.5, 0.0]))
        model_B.set_position(np.array([0.5, 0.0]))
        self.assertFalse(
            model_A.in_collision(model_B),
            "Collision-free state returned with collision",
        )

        # collision
        model_A.radius = 0.5
        model_B.radius = 0.6
        self.assertTrue(
            model_A.in_collision(model_B),
            "Collision state returned with collision-free",
        )

    def test_trajectory_collision(self):
        base_A = np.array([-2.0, 0.0])
        base_B = np.array([2.0, 0.0])

        model_A = LineCollisionModel(base_A, np.zeros(2))
        model_B = LineCollisionModel(base_B, np.zeros(2))
        collision_group = CollisionGroup([model_A, model_B])

        # trajectories with the same start time
        path1 = [[-1, 1], [-1, -1]]
        path2 = [[1, -1], [1, 1]]
        trajA = Trajectory.from_const_vel_path(path1, 1.0)
        trajB = Trajectory.from_const_vel_path(path2, 1.0)

        # collision-free
        trajs = [trajA, trajB]
        collision = check_trajectory_collision(collision_group, trajs, 0.1)
        self.assertFalse(collision, "Collision-free trajectory returned with collision")

        # collision
        trajs = [trajB, trajA]
        collision = check_trajectory_collision(collision_group, trajs, 0.1)
        self.assertTrue(collision, "Colliding trajectory returned with collision-free")

        # trajectories with different start times
        path1 = [[0, 1], [0, -1]]
        path2 = [[1.5, 0], [-0.5, 0]]
        trajA = Trajectory.from_const_vel_path(path1, 1.0)
        trajB = Trajectory.from_const_vel_path(path2, 1.0)

        # collision-free
        trajs = [trajA, trajB]
        collision = check_trajectory_collision(collision_group, trajs, 0.1)
        self.assertFalse(collision, "Collision-free trajectory returned with collision")

        # collision
        trajA = Trajectory.from_const_vel_path(path1, 1.0, start_time=0.5)
        trajB = Trajectory.from_const_vel_path(path2, 1.0, start_time=-0.5)
        trajs = [trajA, trajB]
        collision = check_trajectory_collision(collision_group, trajs, 0.1)
        print(collision)
        self.assertTrue(collision, "Colliding trajectory returned with collision-free")
