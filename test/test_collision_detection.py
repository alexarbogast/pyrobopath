import unittest
import numpy as np

from pyrobopath.collision_detection import *
from pyrobopath.collision_detection import _ConcurrentSegmentIterator


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

        self.assertEqual(traj.start_time(), 0.0)
        self.assertEqual(traj.end_time(), 0.0)

        traj.add_traj_point(TrajectoryPoint([-1.0, 1.0, 0.0], -20.0))
        traj.add_traj_point(TrajectoryPoint([1.0, 0.0, -1.0], 40.0))

        self.assertEqual(traj.start_time(), -20.0, "Start time != -20.0")
        self.assertEqual(traj.end_time(), 40.0, "End time != 40.0")
        self.assertEqual(traj.elapsed(), 60.0, "Elapsed != 60.0")

        traj = Trajectory()
        pt1 = TrajectoryPoint([-1.0, 0.0, 0.0], 0.0)
        pt2 = TrajectoryPoint([0.0, 0.0, 0.0], 1.0)
        pt3 = TrajectoryPoint([1.0, 0.0, 0.0], 2.0)
        traj.add_traj_point(pt1)
        traj.add_traj_point(pt2)
        traj.add_traj_point(pt3)

        # point interpolation
        point = traj.get_point_at_time(-1.0)
        self.assertIsNone(point)

        point = traj.get_point_at_time(0.0)
        self.assertEqual(point, pt1)

        point = traj.get_point_at_time(0.5)
        self.assertEqual(point, TrajectoryPoint([-0.5, 0.0, 0.0], 0.5))

        point = traj.get_point_at_time(2.0)
        self.assertEqual(point, pt3)

        point = traj.get_point_at_time(3.0)
        self.assertIsNone(point)

        # trajectory slicing
        # fmt: off
        # before start time
        sliced = traj.slice(-2.0, -1.0)
        self.assertEqual(sliced, Trajectory(), "Sliced is not empty")

        # up to start time
        sliced = traj.slice(-2.0, 0.0)
        self.assertEqual(len(sliced.points), 1, "Sliced trajectory does not have 1 point")
        self.assertEqual(sliced.start_time(), pt1.time, "Sliced trajectory has wrong start time")
        self.assertEqual(sliced.end_time(), pt1.time, "Sliced trajectory has wrong end time")

        # original slice
        sliced = traj.slice(0.0, 2.0)
        self.assertEqual(len(sliced.points), 3, "Sliced trajectory does not have 3 points")
        self.assertEqual(sliced.start_time(), pt1.time, "Sliced trajectory has wrong start time")
        self.assertEqual(sliced.end_time(), pt3.time, "Sliced trajectory has wrong end time")
        for a, b in zip(sliced.points, traj.points):
            self.assertTrue(np.allclose(a.data, b.data),
                            "Sliced trajectory points are incorrect")

        # sandwiching start time
        sliced = traj.slice(-1.0, 0.5)
        self.assertEqual(len(sliced.points), 2, "Sliced trajectory does not have 2 points")
        self.assertEqual(sliced.start_time(), 0.0)
        self.assertEqual(sliced.end_time(), 0.5)
        points = [TrajectoryPoint([-1.0, 0.0, 0.0], 0.0), TrajectoryPoint([-0.5, 0.0, 0.0], 0.5)]
        for a, b in zip(sliced.points, points):
            self.assertEqual(a, b, "Sliced trajectory points are incorrect")

        # in the middle
        sliced = traj.slice(0.5, 1.5)
        self.assertEqual(sliced.start_time(), 0.5, "Sliced trajectory has wrong start time")
        self.assertEqual(sliced.end_time(), 1.5, "Sliced trajectory has wrong end time")
        points = [TrajectoryPoint([-0.5, 0.0, 0.0], 0.5), pt2, TrajectoryPoint([0.5, 0.0, 0.0], 1.5)]
        for a, b in zip(sliced.points, points):
            self.assertTrue(np.allclose(a.data, b.data),
                            "Sliced trajectory points are incorrect")

        # one segment
        sliced = traj.slice(0.25, 0.75)
        self.assertEqual(len(sliced.points), 2, "Sliced trajectory does not have 2 point")
        self.assertEqual(sliced.start_time(), 0.25, "Sliced trajectory has wrong start time")
        self.assertEqual(sliced.end_time(), 0.75, "Sliced trajectory has wrong end time")

        # at end time
        sliced = traj.slice(2.0, 3.0)
        self.assertEqual(len(sliced.points), 1, "Sliced trajectory does not have 1 point")
        self.assertEqual(sliced.start_time(), pt3.time, "Sliced trajectory has wrong start time")
        self.assertEqual(sliced.end_time(), pt3.time, "Sliced trajectory has wrong end time")

        # after end time
        sliced = traj.slice(3.0, 4.0)
        self.assertEqual(sliced, Trajectory(), "Sliced is not empty")
        # fmt: on

    def test_collision_group(self):
        base_A = np.array([-1.0, -1.0, 0.0])
        base_B = np.array([1.0, -1.0, 0.0])
        base_C = np.array([0.0, 1.0, 0.0])

        model_A = LollipopCollisionModel(base_A, 0.1)
        model_B = LollipopCollisionModel(base_B, 0.1)
        model_C = LollipopCollisionModel(base_C, 0.1)
        collision_group = CollisionGroup([model_A, model_B, model_C])

        # no collisions
        model_A.translation = np.array([-1.0, 0.0, 0.0])
        model_B.translation = np.array([1.0, 0.0, 0.0])
        model_C.translation = np.array([1.0, 1.0, 0.0])
        self.assertFalse(
            collision_group.in_collision(),
            "Collision-free state returned with collision",
        )

        # one collision (A & B)
        model_A.translation = np.array([0.0, 0.0, 0.0])
        model_B.translation = np.array([0.0, 0.0, 0.0])
        model_C.translation = np.array([1.0, 1.0, 0.0])
        self.assertTrue(
            collision_group.in_collision(),
            "Collision state returned with collision-free",
        )

        # all colliding
        model_A.translation = np.array([1.0, 1.0, 0.0])
        model_B.translation = np.array([-1.0, -1.0, 0.0])
        model_C.translation = np.array([0.0, -1.0, 0.0])
        self.assertTrue(
            collision_group.in_collision(),
            "Collision state returned with collision-free",
        )

    def test_line_collision_model(self):
        base_A = np.array([-2.0, 0.0, 0.0])
        base_B = np.array([2.0, 0.0, 0.0])

        model_A = LineCollisionModel(base_A)
        model_B = LineCollisionModel(base_B)

        # no collision
        model_A.translation = np.array([-1.0, 0.0, 0.0])
        model_B.translation = np.array([1.0, 0.0, 0.0])
        self.assertFalse(
            model_A.in_collision(model_B),
            "Collision-free state returned with collision",
        )

        # collision
        model_A.translation = np.array([1.0, 1.0, 0.0])
        model_B.translation = np.array([-1.0, 1.0, 0.0])
        self.assertTrue(
            model_A.in_collision(model_B),
            "Collision state returned with collision-free",
        )

        # colinear collision
        model_A.translation = np.array([1.0, 0.0, 0.0])
        model_B.translation = np.array([-1.0, 0.0, 0.0])
        self.assertTrue(
            model_A.in_collision(model_B),
            "Collision state returned with collision-free",
        )

    def test_lollipop_collision_model(self):
        base_A = np.array([-2.0, 0.0, 0.0])
        base_B = np.array([2.0, 0.0, 0.0])

        model_A = LollipopCollisionModel(base_A, 0.25)
        model_B = LollipopCollisionModel(base_B, 0.25)

        # collision
        model_A.translation = np.array([1.0, 1.0, 0.0])
        model_B.translation = np.array([-1.0, 1.0, 0.0])
        self.assertTrue(
            model_A.in_collision(model_B),
            "Collision state returned with collision-free",
        )

        # no collision
        model_A.translation = np.array([-0.5, 0.0, 0.0])
        model_B.translation = np.array([0.5, 0.0, 0.0])
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


class TestFCLCollisionDetection(unittest.TestCase):
    def test_fcl_box_collision_models(self):
        box_model_1 = FCLBoxCollisionModel(1, 1, 1)
        box_model_2 = FCLBoxCollisionModel(1, 1, 1)

        # collision
        self.assertTrue(
            box_model_1.in_collision(box_model_2),
            "Collision state returned with collision-free",
        )

        # collision-free
        box_model_1.translation = np.array([-1, 0, 0])
        box_model_2.translation = np.array([1, 0, 0])
        self.assertFalse(
            box_model_1.in_collision(box_model_2),
            "Collision-free state returned with collision",
        )

    def test_fcl_robot_bb_collision_model(self):
        # models with no offset
        # robot_bb_1 = FCLRobotBBCollisionModel(
        #     dims=(3.0, 1.0, 0.5), anchor=(-5.0, 0.0, 0.0)
        # )
        # robot_bb_1.translation = np.array([0.1, 2.0, 0.0])
        #
        # robot_bb_2 = FCLRobotBBCollisionModel(
        #     dims=(3.0, 1.0, 0.5), anchor=(5.0, 0.0, 0.0)
        # )
        # robot_bb_2.translation = np.array([-0.1, 2.0, 0.0])
        #
        # # collision
        # self.assertTrue(
        #     robot_bb_1.in_collision(robot_bb_2),
        #     "Collision state returned with collision-free",
        # )
        #
        # # collision-free
        # robot_bb_1.translation = np.array([-1.0, 0.0, 0.0])
        # robot_bb_2.translation = np.array([1.0, 0.0, 0.0])
        # self.assertFalse(
        #     robot_bb_1.in_collision(robot_bb_2),
        #     "Collision-free state returned with collision",
        # )

        ## models with no offset
        robot_bb_1 = FCLRobotBBCollisionModel(
            dims=(3.0, 2.0, 2.0), anchor=(-7.0, 0.0, 0.0)
        )
        robot_bb_1.translation = np.array([0.0, 0.0, 0.0])

        robot_bb_2 = FCLRobotBBCollisionModel(
            dims=(3.0, 2.0, 2.0), anchor=(7.0, 0.0, 0.0)
        )
        robot_bb_2.translation = np.array([0.0, 0.0, 0.0])

        # edge collision
        self.assertTrue(
            robot_bb_1.in_collision(robot_bb_2),
            "Collision state returned with collision-free",
        )

        # collision
        robot_bb_1.translation = np.array([1.0, 0.0, 0.0])
        robot_bb_2.translation = np.array([-1.0, 0.0, 0.0])
        self.assertTrue(
            robot_bb_1.in_collision(robot_bb_2),
            "Collision state returned with collision-free",
        )

        # collision-free
        robot_bb_1.translation = np.array([-1.0, 0.0, 0.0])
        robot_bb_2.translation = np.array([1.0, 0.0, 0.0])
        self.assertFalse(
            robot_bb_1.in_collision(robot_bb_2),
            "Collision-free state returned with collision",
        )

        ## models with offset
        robot_bb_1 = FCLRobotBBCollisionModel(
            dims=(3.0, 2.0, 2.0), anchor=(-7.0, 0.0, 0.0), offset=(1.0, 0.0, 0.0)
        )

        robot_bb_2 = FCLRobotBBCollisionModel(
            dims=(3.0, 2.0, 2.0), anchor=(7.0, 0.0, 0.0), offset=(1.0, 0.0, 0.0)
        )

        # edge-collision
        robot_bb_1.translation = np.array([-1.0, 0.0, 0.0])
        robot_bb_2.translation = np.array([1.0, 0.0, 0.0])
        self.assertTrue(
            robot_bb_1.in_collision(robot_bb_2),
            "Collision state returned with collision-free",
        )

        robot_bb_1 = FCLRobotBBCollisionModel(
            dims=(3.0, 2.0, 2.0), anchor=(-5.0, 1.0, 0.0), offset=(1.0, 1.0, 0.0)
        )

        robot_bb_2 = FCLRobotBBCollisionModel(
            dims=(3.0, 2.0, 2.0), anchor=(5.0, -1.0, 0.0), offset=(1.0, 1.0, 0.0)
        )

        # collision-free
        robot_bb_1.translation = np.array([0.0, 1.0, 0.0])
        robot_bb_2.translation = np.array([0.0, -1.0, 0.0])
        self.assertFalse(
            robot_bb_1.in_collision(robot_bb_2),
            "Collision-free state returned with collision",
        )

    def test_trajectory_collision_query(self):
        robot_bb_1 = FCLRobotBBCollisionModel(
            dims=(3.0, 0.2, 1.0), anchor=(-5.0, 0.0, 0.0)
        )
        robot_bb_2 = FCLRobotBBCollisionModel(
            dims=(3.0, 0.2, 1.0), anchor=(5.0, 0.0, 0.0)
        )
        threshold = 0.1

        # collision-free
        path1 = [[-3.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
        path2 = [[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        traj1 = Trajectory.from_const_vel_path(path1, 1.0)
        traj2 = Trajectory.from_const_vel_path(path2, 1.0)

        res = trajectory_collision_query(
            robot_bb_1, traj1, robot_bb_2, traj2, threshold
        )
        self.assertFalse(res, "Collision-free trajectory returned with collision")

        # collision
        path1 = [[-3.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        path2 = [[3.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
        traj1 = Trajectory.from_const_vel_path(path1, 1.0)
        traj2 = Trajectory.from_const_vel_path(path2, 1.0)

        res = trajectory_collision_query(
            robot_bb_1, traj1, robot_bb_2, traj2, threshold
        )
        self.assertTrue(res, "Collision trajectory returned with collision-free")

        # collision
        path1 = [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
        path2 = [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]
        traj1 = Trajectory.from_const_vel_path(path1, 1.0, 0.5)
        traj2 = Trajectory.from_const_vel_path(path2, 1.0)

        res = trajectory_collision_query(
            robot_bb_1, traj1, robot_bb_2, traj2, threshold
        )
        self.assertTrue(res, "Collision trajectory returned with collision-free")


class TestConcurrentSegmentIterator(unittest.TestCase):
    def test_concurrent_segment_iterator(self):
        path1 = [[0.0, 2.0, 0.0], [0.0, -2.0, 0.0]]
        path2 = [[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]
        traj1 = Trajectory().from_const_vel_path(path1, 1.0, 0.0)
        traj2 = Trajectory().from_const_vel_path(path2, 1.0, 0.0)

        traj_pair = list(_ConcurrentSegmentIterator([traj1, traj2]))
        self.assertEqual(traj_pair[0][0].start_time(), traj1.start_time())
        self.assertEqual(traj_pair[0][1].start_time(), traj2.start_time())
        self.assertEqual(traj_pair[0][0].end_time(), traj1.end_time())
        self.assertEqual(traj_pair[0][1].end_time(), traj2.end_time())

        traj2 = Trajectory.from_const_vel_path(path2, 1.0, 2.0)
        traj_pair = list(_ConcurrentSegmentIterator([traj1, traj2]))

        self.assertEqual(len(traj_pair), 3)
        self.assertEqual(traj_pair[0][0].start_time(), traj1.start_time())
        self.assertEqual(traj_pair[0][1].start_time(), traj2.start_time())
        self.assertEqual(traj_pair[0][0].end_time(), 2.0)
        self.assertEqual(traj_pair[0][1].end_time(), 2.0)

        self.assertEqual(traj_pair[1][0].start_time(), 2.0)
        self.assertEqual(traj_pair[1][1].start_time(), 2.0)
        self.assertEqual(traj_pair[1][0].end_time(), 4.0)
        self.assertEqual(traj_pair[1][1].end_time(), 4.0)

        self.assertEqual(traj_pair[2][0].start_time(), 4.0)
        self.assertEqual(traj_pair[2][1].start_time(), 4.0)
        self.assertEqual(traj_pair[2][0].end_time(), traj1.end_time())
        self.assertEqual(traj_pair[2][1].end_time(), traj2.end_time())


class TestTrajectoryCollision(unittest.TestCase):
    def test_continuous_collide(self):
        model1 = FCLBoxCollisionModel(1.0, 1.0, 1.0)
        model2 = FCLBoxCollisionModel(1.0, 1.0, 1.0)

        model1.translation = np.array([-2.0, 0.0, 0.0])
        model2.translation = np.array([2.0, 0.0, 0.0])

        # collide
        origin = np.array([0.0, 0.0, 0.0])
        threshold = 0.01
        ret = continuous_collide(model1, origin, model2, origin, threshold)
        self.assertTrue(ret)

        # no collide
        pm1 = np.array([-0.6, 0.0, 0.0])
        pm2 = np.array([0.6, 0.0, 0.0])
        threshold = 0.01
        ret = continuous_collide(model1, pm1, model2, pm2, threshold)
        self.assertFalse(ret)

        # collide
        pm1 = np.array([0.6, 0.0, 0.0])
        pm2 = np.array([-0.6, 0.0, 0.0])
        threshold = 0.01
        ret = continuous_collide(model1, pm1, model2, pm2, threshold)
        self.assertTrue(ret)

    def test_trajectory_collision(self):
        base_A = np.array([-2.0, 0.0, 0.0])
        base_B = np.array([2.0, 0.0, 0.0])

        model_A = LineCollisionModel(base_A)
        model_B = LineCollisionModel(base_B)
        collision_group = CollisionGroup([model_A, model_B])

        # trajectories with the same start time
        path1 = [np.array([-1.0, 1.0, 0.0]), np.array([-1.0, -1.0, 0.0])]
        path2 = [np.array([1.0, -1.0, 0.0]), np.array([1.0, 1.0, 0.0])]
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
        path1 = [np.array([0, 1, 0]), np.array([0, -1, 0])]
        path2 = [np.array([1.5, 0, 0]), np.array([-0.5, 0, 0])]
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
        self.assertTrue(collision, "Colliding trajectory returned with collision-free")

    def test_trajectory_collision_query(self):
        model1 = FCLBoxCollisionModel(1.0, 1.0, 1.0)
        model2 = FCLBoxCollisionModel(1.0, 1.0, 1.0)
        threshold = 0.01

        path1 = [np.array([-3.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0])]
        path2 = [np.array([0.0, -3.0, 0.0]), np.array([0.0, 3.0, 0.0])]
        traj1 = Trajectory.from_const_vel_path(path1, 1.0, 0.0)
        traj2 = Trajectory.from_const_vel_path(path2, 1.0, 0.0)
        ret = trajectory_collision_query(model1, traj1, model2, traj2, threshold)
        self.assertTrue(ret)

        path1 = [[-3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
        path2 = [[3.0, 0.0, 0.0], [0.0, -3.0, 0.0]]
        traj1 = Trajectory.from_const_vel_path(path1, 1.0, 0.0)
        traj2 = Trajectory.from_const_vel_path(path2, 1.0, 0.0)
        ret = trajectory_collision_query(model1, traj1, model2, traj2, threshold)
        self.assertFalse(ret)

        model1 = FCLRobotBBCollisionModel((3.0, 0.2, 1.0), (-6.0, 0.0, 0.0))
        model2 = FCLRobotBBCollisionModel((3.0, 0.2, 1.0), (6.0, 0.0, 0.0))

        path1 = [[0.0, 3.0, 0.0], [0.0, -3.0, 0.0]]
        path2 = [[0.0, -3.0, 0.0], [0.0, 3.0, 0.0]]
        traj1 = Trajectory.from_const_vel_path(path1, 1.0, 0.0)
        traj2 = Trajectory.from_const_vel_path(path2, 1.0, 0.0)
        ret = trajectory_collision_query(model1, traj1, model2, traj2, threshold)
        self.assertTrue(ret)

        path1 = [[-3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
        path2 = [[3.0, 0.0, 0.0], [0.0, -3.0, 0.0]]
        traj1 = Trajectory.from_const_vel_path(path1, 1.0, 0.0)
        traj2 = Trajectory.from_const_vel_path(path2, 1.0, 0.0)
        ret = trajectory_collision_query(model1, traj1, model2, traj2, threshold)
        self.assertFalse(ret)


if __name__ == "__main__":
    unittest.main()
