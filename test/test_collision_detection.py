import unittest
import numpy as np
from copy import copy

from collision_detection import (
    LineCollisionModel,
    LollipopCollisionModel,
    CollisionGroup,
)
from collision_detection import Trajectory, check_trajectory_collision


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


    def test_line_collision_model(self):
        base_A = np.array([-2.0, 0.0])
        base_B = np.array([2.0, 0.0])

        model_rob1 = LineCollisionModel(base_A, np.zeros(2))
        model_rob2 = LineCollisionModel(base_B, np.zeros(2))
        collision_group = CollisionGroup([model_rob1, model_rob2])

        # no collision
        model_rob1.set_position([-1.0, 0.0])
        model_rob2.set_position([1.0, 0.0])
        self.assertFalse(
            collision_group.in_collision(),
            "Collision-free state returned with collision",
        )

        # collision
        model_rob1.set_position([1.0, 1.0])
        model_rob2.set_position([-1.0, 1.0])
        self.assertTrue(
            collision_group.in_collision(),
            "Collision state returned with collision-free",
        )

        # colinear collision
        model_rob1.set_position([1.0, 0.0])
        model_rob2.set_position([-1.0, 0.0])
        self.assertTrue(
            collision_group.in_collision(),
            "Collision state returned with collision-free",
        )

    def test_lollipop_collision_model(self):
        base_A = np.array([-2.0, 0.0])
        base_B = np.array([2.0, 0.0])

        model_rob1 = LollipopCollisionModel(base_A, np.zeros(2), 0.25)
        model_rob2 = LollipopCollisionModel(base_B, np.zeros(2), 0.25)
        collision_group = CollisionGroup([model_rob1, model_rob2])

        # collision
        model_rob1.set_position(np.array([1.0, 1.0]))
        model_rob2.set_position(np.array([-1.0, 1.0]))
        self.assertTrue(
            collision_group.in_collision(),
            "Collision state returned with collision-free",
        )

                # no collision
        model_rob1.set_position(np.array([-0.5, 0.0]))
        model_rob2.set_position(np.array([0.5, 0.0]))
        self.assertFalse(
            collision_group.in_collision(),
            "Collision-free state returned with collision",
        )

        model_rob1.radius = 0.5
        model_rob2.radius = 0.6
        self.assertTrue(
            collision_group.in_collision(),
            "Collision state returned with collision-free",
        )


    def test_trajectory_collision(self):
        base_A = np.array([2.0, 0.0])
        base_B = np.array([-2.0, 0.0])

        model_rob1 = LineCollisionModel(base_A, np.zeros(2))
        model_rob2 = LineCollisionModel(base_B, np.zeros(2))
        collision_group = CollisionGroup([model_rob1, model_rob2])
        path1 = [[1, 1], [1, -1]]
        path2 = [[-1, 1], [-1, -1]]

        # collision-free
        trajA = Trajectory.from_const_vel_path(path1, 1.0)
        trajB = Trajectory.from_const_vel_path(path2, 1.0)
        trajs = [trajA, trajB]
        collision = check_trajectory_collision(collision_group, trajs, 0.1)
        self.assertFalse(collision, "Collision-free trajectory returned with collision")

        # collision
        trajs = [trajB, trajA]
        collision = check_trajectory_collision(collision_group, trajs, 0.1)
        self.assertTrue(collision, "Colliding trajectory returned with collision-free")
