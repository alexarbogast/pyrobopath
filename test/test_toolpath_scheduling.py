import numpy as np
import numpy.testing as nt

from pyrobopath.process import AgentModel
from pyrobopath.toolpath import Contour
from pyrobopath.collision_detection import (
    FCLRobotBBCollisionModel,
    TrajectoryPoint,
    Trajectory,
)
from pyrobopath.toolpath_scheduling import *


class TestToolpathSchedule:
    def test_toolpath_schedule(self):
        contour1 = Contour(
            [
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, -1.0, 0.0]),
            ],
        )

        contour2 = Contour(
            [
                np.array([0.0, -1.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
            ],
        )

        schedule = ToolpathSchedule()
        schedule.add_event(ContourEvent(0.0, contour1, 1.0))
        schedule.add_event(ContourEvent(5.0, contour2, 1.0))

        assert schedule.duration() == 7.0

        default = "default"
        state = schedule.get_state(-1.0, default)
        assert state == default

        state = schedule.get_state(0.0, default)
        nt.assert_array_equal(state, np.array([0.0, 1.0, 0.0]))

        state = schedule.get_state(0.5, default)
        nt.assert_array_equal(state, np.array([0.0, 0.5, 0.0]))

        state = schedule.get_state(2.0, default)
        nt.assert_array_equal(state, np.array([0.0, -1.0, 0.0]))

        state = schedule.get_state(3.0, default)
        nt.assert_array_equal(state, np.array([0.0, -1.0, 0.0]))

        state = schedule.get_state(5.0, default)
        nt.assert_array_equal(state, np.array([0.0, -1.0, 0.0]))

        state = schedule.get_state(7.0, default)
        nt.assert_array_equal(state, np.array([1.0, 0.0, 0.0]))

        state = schedule.get_state(8.0, default)
        nt.assert_array_equal(state, np.array([1.0, 0.0, 0.0]))


class TestToolpathCollision:
    def test_event_causes_collision_two(self):
        bf1 = np.array([-5.0, 0.0, 0.0])
        bf2 = np.array([5.0, 0.0, 0.0])

        agent1 = AgentModel(
            capabilities=[0],
            collision_model=FCLRobotBBCollisionModel((3.0, 0.2, 1.0), bf1),
            base_frame_position=bf1,
            home_position=np.array([-3.0, 0.0, 0.0]),
            velocity=10.0,
            travel_velocity=10.0,
        )
        agent2 = AgentModel(
            capabilities=[0],
            collision_model=FCLRobotBBCollisionModel((3.0, 0.2, 1.0), bf2),
            base_frame_position=bf2,
            home_position=np.array([3.0, 0.0, 0.0]),
            velocity=10.0,
            travel_velocity=10.0,
        )
        agent_models = {"agent1": agent1, "agent2": agent2}
        threshold = 0.05

        c1 = Contour([np.array([0.0, 2.0, 0.0]), np.array([0.0, -2.0, 0.0])])
        c2 = Contour([np.array([2.0, 0.0, 0.0]), np.array([-2.0, 0.0, 0.0])])

        schedule = MultiAgentToolpathSchedule()

        # no collision
        event1 = ContourEvent(0.0, c1, 1.0)
        collide = event_causes_collision(
            event1, "agent1", schedule, agent_models, threshold
        )
        assert not collide
        schedule.add_event(event1, "agent1")

        # collision
        event2 = ContourEvent(0.0, c2, 1.0)
        collide = event_causes_collision(
            event2, "agent2", schedule, agent_models, threshold
        )
        assert collide

        # no collision
        event3 = ContourEvent(2.0, c2, 1.0)
        collide = event_causes_collision(
            event3, "agent2", schedule, agent_models, threshold
        )
        assert not collide
        schedule.add_event(event3, "agent2")

        # no collision
        c3 = Contour([np.array([-2.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])])
        event4 = ContourEvent(6.0, c3, 1.0)
        collide = event_causes_collision(
            event4, "agent2", schedule, agent_models, threshold
        )
        assert not collide

        # collision
        c4 = Contour([np.array([-2.0, 0.0, 0.0]), np.array([0.0, -2.0, 0.0])])
        event5 = ContourEvent(6.0, c4, 1.0)
        collide = event_causes_collision(
            event5, "agent2", schedule, agent_models, threshold
        )
        assert collide

    def test_event_causes_collision_three(self):
        agent1 = AgentModel(
            capabilities=[0],
            base_frame_position=np.array([-5.0, 0.0, 0.0]),
            home_position=np.array([-3.0, 0.0, 0.0]),
            velocity=10.0,
            travel_velocity=10.0,
            collision_model=FCLRobotBBCollisionModel(
                dims=(3.0, 0.2, 1.0), anchor=(-5.0, 0.0, 0.0)
            ),
        )
        agent2 = AgentModel(
            capabilities=[0],
            base_frame_position=np.array([5.0, 0.0, 0.0]),
            home_position=np.array([3.0, 0.0, 0.0]),
            velocity=10.0,
            travel_velocity=10.0,
            collision_model=FCLRobotBBCollisionModel(
                dims=(3.0, 0.2, 1.0), anchor=(5.0, 0.0, 0.0)
            ),
        )
        agent3 = AgentModel(
            capabilities=[0],
            base_frame_position=np.array([0.0, 5.0, 0.0]),
            home_position=np.array([0.0, 3.0, 0.0]),
            velocity=10.0,
            travel_velocity=10.0,
            collision_model=FCLRobotBBCollisionModel(
                dims=(3.0, 0.2, 1.0), anchor=(0.0, 5.0, 0.0)
            ),
        )
        agent_models = {"agent1": agent1, "agent2": agent2, "agent3": agent3}
        threshold = 0.05

        c1 = Contour([np.array([-4.0, 0.0, 0.0]), np.array([-2.0, 0.0, 0.0])])
        c2 = Contour([np.array([4.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])])
        c3 = Contour([np.array([0.0, 4.0, 0.0]), np.array([0.0, 2.0, 0.0])])

        schedule = MultiAgentToolpathSchedule()

        # no collision
        event1 = ContourEvent(0.0, c1, 1.0)
        collide = event_causes_collision(
            event1, "agent1", schedule, agent_models, threshold
        )
        assert not collide
        schedule.add_event(event1, "agent1")

        event2 = ContourEvent(0.0, c2, 1.0)
        collide = event_causes_collision(
            event2, "agent2", schedule, agent_models, threshold
        )
        assert not collide
        schedule.add_event(event2, "agent2")

        event3 = ContourEvent(0.0, c3, 1.0)
        collide = event_causes_collision(
            event3, "agent3", schedule, agent_models, threshold
        )
        assert not collide
        schedule.add_event(event3, "agent3")

        # collision (agent1 - agent2)
        c4 = Contour([np.array([-2.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0])])
        event4 = ContourEvent(2.0, c4, 1.0)
        collide = event_causes_collision(
            event4, "agent1", schedule, agent_models, threshold
        )
        assert collide

        event5 = ContourEvent(2.1, c4, 1.0)
        collide = event_causes_collision(
            event5, "agent1", schedule, agent_models, threshold
        )
        assert collide

        # no collision (agent1 - agent3)
        c5 = Contour([np.array([-2.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])])
        event6 = ContourEvent(3.0, c5, 1.0)
        collide = event_causes_collision(
            event6, "agent1", schedule, agent_models, threshold
        )
        assert not collide
        schedule.add_event(event6, "agent1")

        # collision (agent1 - agent3)
        c6 = Contour([np.array([0.0, 2.0, 0.0]), np.array([0.0, 0.0, 0.0])])
        event7 = ContourEvent(3.0, c6, 1.0)
        collide = event_causes_collision(
            event7, "agent3", schedule, agent_models, threshold
        )
        assert collide

        c7 = Contour(
            [
                np.array([-1.0, 1.0, 0.0]),
                np.array([-1.0, -1.0, 0.0]),
                np.array([-2.0, -1.0, 0.0]),
                np.array([-2.0, 1.0, 0.0]),
            ]
        )
        event8 = ContourEvent(6.0, c7, 2.0)
        collide = event_causes_collision(
            event8, "agent1", schedule, agent_models, threshold
        )
        assert not collide
        schedule.add_event(event8, "agent1")

        c8 = Contour([np.array([0.0, 1.0, 0.0]), np.array([0.0, -0.4, 0.0])])
        event9 = ContourEvent(6.0, c8, 1.0)
        collide = event_causes_collision(
            event9, "agent2", schedule, agent_models, threshold
        )
        assert not collide
        schedule.add_event(event9, "agent2")

        c9 = Contour([np.array([0.0, 2.0, 0.0]), np.array([-1.0, 1.0, 0.0])])
        event10 = ContourEvent(5.0, c9, 5.0)
        collide = event_causes_collision(
            event10, "agent3", schedule, agent_models, threshold
        )
        schedule.add_event(event10, "agent3")

        assert collide

    def test_events_cause_collision_two(self):
        agent1 = AgentModel(
            capabilities=[0],
            base_frame_position=np.array([-5.0, 0.0, 0.0]),
            home_position=np.array([-3.0, 0.0, 0.0]),
            velocity=10.0,
            travel_velocity=10.0,
            collision_model=FCLRobotBBCollisionModel(
                dims=(3.0, 0.2, 1.0), anchor=(-5.0, 0.0, 0.0)
            ),
        )
        agent2 = AgentModel(
            capabilities=[0],
            base_frame_position=np.array([5.0, 0.0, 0.0]),
            home_position=np.array([3.0, 0.0, 0.0]),
            velocity=10.0,
            travel_velocity=10.0,
            collision_model=FCLRobotBBCollisionModel(
                dims=(3.0, 0.2, 1.0), anchor=(5.0, 0.0, 0.0)
            ),
        )
        agent_models = {"agent1": agent1, "agent2": agent2}
        threshold = 0.05

        schedule = MultiAgentToolpathSchedule()
        schedule.add_agent("agent1")
        schedule.add_agent("agent2")

        c1s = [
            Contour([np.array([-3.0, 0.0, 0.0]), np.array([-2.0, 0.0, 0.0])], tool=-1),
            Contour([np.array([-2.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])], tool=0),
            Contour(
                [
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 0.0, 0.0]),
                    np.array([-3.0, 0.0, 0.0]),
                ],
                tool=-1,
            ),
        ]
        ec1_0 = ContourEvent(0.0, c1s[0], 1.0)
        ec1_1 = ContourEvent(ec1_0.end, c1s[1], 1.0)
        ec1_2 = ContourEvent(ec1_1.end, c1s[2], 1.0)

        ret = events_cause_collision(
            [ec1_0, ec1_1, ec1_2], "agent1", schedule, agent_models, threshold
        )
        schedule.add_event(ec1_0, "agent1")
        schedule.add_event(ec1_1, "agent1")
        schedule.add_event(ec1_2, "agent1")
        assert not ret

        c2s = [
            Contour([np.array([3.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])], tool=-1),
            Contour(
                [
                    np.array([0.0, 1.0, 0.0]),
                    np.array([0.0, 0.0, 0.0]),
                    np.array([0.0, -1.0, 0.0]),
                ],
                tool=0,
            ),
            Contour([np.array([0.0, -1.0, 0.0]), np.array([3.0, 0.0, 0.0])], tool=-1),
        ]
        ec2_0 = ContourEvent(0.0, c2s[0], 1.0)
        ec2_1 = ContourEvent(ec2_0.end, c2s[1], 1.0)
        ec2_2 = ContourEvent(ec2_1.end, c2s[2], 1.0)

        ret = events_cause_collision(
            [ec2_0, ec2_1, ec2_2], "agent2", schedule, agent_models, threshold
        )
        assert ret

        # shift start time
        ec2_0 = ContourEvent(2.0, c2s[0], 1.0)
        ec2_1 = ContourEvent(ec2_0.end, c2s[1], 1.0)
        ec2_2 = ContourEvent(ec2_1.end, c2s[2], 1.0)
        ret = events_cause_collision(
            [ec2_0, ec2_1, ec2_2], "agent2", schedule, agent_models, threshold
        )
        assert not ret
        schedule.add_event(ec2_0, "agent2")
        schedule.add_event(ec2_1, "agent2")
        schedule.add_event(ec2_2, "agent2")

    def test_schedule_to_trajectory(self):
        c1 = Contour([np.array([0.0, 2.0, 0.0]), np.array([0.0, -2.0, 0.0])])
        c2 = Contour([np.array([2.0, 0.0, 0.0]), np.array([-2.0, 0.0, 0.0])])

        event1 = ContourEvent(0.0, c1, 1.0)
        event2 = ContourEvent(6.0, c2, 1.0)

        schedule = ToolpathSchedule()
        schedule.add_event(event1)
        schedule.add_event(event2)

        default = "default"
        traj = schedule_to_trajectory(
            schedule, t_start=-0.2, t_end=-0.1, default_state=default
        )
        assert traj.n_points() == 2
        assert traj.start_time() == -0.2
        assert traj.end_time() == -0.1
        assert traj.points[0].data == default
        assert traj.points[-1].data == default

        traj = schedule_to_trajectory(
            schedule, t_start=-0.2, t_end=0.0, default_state=default
        )
        assert traj.n_points() == 2
        assert traj.start_time() == -0.2
        assert traj.end_time() == 0.0
        assert traj.points[0].data == default
        nt.assert_array_equal(traj.points[-1].data, c1.path[0])

        traj = schedule_to_trajectory(
            schedule, t_start=0.0, t_end=2.0, default_state=default
        )
        assert traj.n_points() == 2
        assert traj.start_time() == 0.0
        assert traj.end_time() == 2.0
        nt.assert_array_equal(traj.points[0].data, c1.path[0])
        nt.assert_array_equal(traj.points[-1].data, np.array([0.0, 0.0, 0.0]))

        traj = schedule_to_trajectory(
            schedule, t_start=2.0, t_end=5.0, default_state=default
        )
        assert traj.n_points() == 3
        assert traj.start_time() == 2.0
        assert traj.end_time() == 5.0
        nt.assert_array_equal(traj.points[0].data, np.array([0.0, 0.0, 0.0]))
        nt.assert_array_equal(traj.points[1].data, c1.path[1])
        nt.assert_array_equal(traj.points[2].data, c1.path[1])

        traj = schedule_to_trajectory(
            schedule, t_start=5.0, t_end=11.0, default_state=default
        )
        assert traj.n_points() == 4
        assert traj.start_time() == 5.0
        assert traj.end_time() == 11.0
        nt.assert_array_equal(traj.points[0].data, c1.path[1])
        nt.assert_array_equal(traj.points[1].data, c2.path[0])
        nt.assert_array_equal(traj.points[2].data, c2.path[1])
        nt.assert_array_equal(traj.points[3].data, c2.path[1])

        traj = schedule_to_trajectory(
            schedule, t_start=10.0, t_end=11.0, default_state=default
        )
        assert traj.n_points() == 2
        assert traj.start_time() == 10.0
        assert traj.end_time() == 11.0
        nt.assert_array_equal(traj.points[0].data, c2.path[1])
        nt.assert_array_equal(traj.points[1].data, c2.path[1])

        traj = schedule_to_trajectory(
            schedule, t_start=6.0, t_end=10.0, default_state=default
        )
        assert traj.n_points() == 2
        assert traj.start_time() == 6.0
        assert traj.end_time() == 10.0
        nt.assert_array_equal(traj.points[0].data, c2.path[0])
        nt.assert_array_equal(traj.points[1].data, c2.path[1])

    def test_schedule_to_trajectories(self):
        p1 = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
        p2 = [np.array([2.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0])]

        e1 = MoveEvent(0.0, p1, 1.0)
        e2 = MoveEvent(3.0, p2, 1.0)

        s = ToolpathSchedule()
        s.add_event(e1)
        s.add_event(e2)

        # full schedule interval
        trajs = schedule_to_trajectories(s, s.start_time(), s.end_time())
        assert len(trajs) == 2
        assert trajs[0].start_time() == e1.start
        assert trajs[0].end_time() == e1.end
        assert trajs[1].start_time() == e2.start
        assert trajs[1].end_time() == e2.end

        trajs = schedule_to_trajectories(s, e1.start, e1.end)
        assert len(trajs) == 1
        assert trajs[0].start_time() == e1.start
        assert trajs[0].end_time() == e1.end

        # partial schedule interval
        trajs = schedule_to_trajectories(s, 0.5, 3.5)
        t1 = Trajectory()
        t1.add_traj_point(TrajectoryPoint(np.array([0.5, 0.0, 0.0]), 0.5))
        t1.add_traj_point(TrajectoryPoint(np.array([1.0, 0.0, 0.0]), 1.0))
        t2 = Trajectory()
        t2.add_traj_point(TrajectoryPoint(np.array([2.0, 0.0, 0.0]), 3.0))
        t2.add_traj_point(TrajectoryPoint(np.array([2.5, 0.0, 0.0]), 3.5))
        assert trajs == [t1, t2]

        # add new event
        p3 = [np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])]
        e3 = MoveEvent(1.0, p3, 1.0)
        s.add_event(e3)

        trajs = schedule_to_trajectories(s, 0.5, 3.5)
        t3 = Trajectory()
        t3.add_traj_point(TrajectoryPoint(np.array([1.0, 0.0, 0.0]), 1.0))
        t3.add_traj_point(TrajectoryPoint(np.array([2.0, 0.0, 0.0]), 2.0))
        assert trajs == [t1, t2, t3]

        # test empty interval
        trajs = schedule_to_trajectories(s, 6.0, 7.0)
        assert trajs == []

    def test_chop_concurrent_trajectories(self):
        # list 1
        t11 = Trajectory()
        t11.add_traj_point(TrajectoryPoint(np.array([0.0, 0.0, 0.0]), 0.0))
        t11.add_traj_point(TrajectoryPoint(np.array([1.0, 0.0, 0.0]), 1.0))

        t12 = Trajectory()
        t12.add_traj_point(TrajectoryPoint(np.array([5.0, 0.0, 0.0]), 5.0))
        t12.add_traj_point(TrajectoryPoint(np.array([8.0, 0.0, 0.0]), 8.0))

        t13 = Trajectory()
        t13.add_traj_point(TrajectoryPoint(np.array([8.0, 0.0, 0.0]), 8.0))
        t13.add_traj_point(TrajectoryPoint(np.array([9.0, 0.0, 0.0]), 9.0))
        list1 = [t11, t12, t13]

        # list 2
        t21 = Trajectory()
        t21.add_traj_point(TrajectoryPoint(np.array([0.0, 0.0, 0.0]), 0.0))
        t21.add_traj_point(TrajectoryPoint(np.array([6.0, 0.0, 0.0]), 6.0))

        t22 = Trajectory()
        t22.add_traj_point(TrajectoryPoint(np.array([7.0, 0.0, 0.0]), 7.0))
        t22.add_traj_point(TrajectoryPoint(np.array([10.0, 0.0, 0.0]), 10.0))
        list2 = [t21, t22]

        concurrent_pairs = concurrent_trajectory_pairs(list1, list2)
        assert len(concurrent_pairs) == 4

        pair0 = concurrent_pairs[0]
        assert pair0[0] == t11
        assert pair0[1] == t11

        pair1 = concurrent_pairs[1]
        sliced_traj = t12.slice(5, 6)
        assert pair1[0] == sliced_traj
        assert pair1[1] == sliced_traj

        pair2 = concurrent_pairs[2]
        sliced_traj = t12.slice(7, 8)
        assert pair2[0] == sliced_traj
        assert pair2[1] == sliced_traj

        pair3 = concurrent_pairs[3]
        assert pair3[0] == t13
        assert pair3[1] == t13

        # single point duplicate trajectory (edge case)
        t11 = Trajectory()
        t11.add_traj_point(TrajectoryPoint(np.array([2.0, 0.0, 0.0]), 2.0))

        t12 = Trajectory()
        t12.add_traj_point(TrajectoryPoint(np.array([2.0, 0.0, 0.0]), 2.0))
        t12.add_traj_point(TrajectoryPoint(np.array([3.0, 0.0, 0.0]), 3.0))
        list1 = [t11, t12]

        t21 = Trajectory()
        t21.add_traj_point(TrajectoryPoint(np.array([0.0, 0.0, 0.0]), 0.0))
        t21.add_traj_point(TrajectoryPoint(np.array([5.0, 0.0, 0.0]), 5.0))
        list2 = [t21]

        concurrent_pairs = concurrent_trajectory_pairs(list1, list2)
        assert len(concurrent_pairs) == 2

        pair0 = concurrent_pairs[0]
        assert pair0[0] == t11
        assert pair0[1] == t11

        pair1 = concurrent_pairs[1]
        assert pair1[0] == t12
        assert pair1[1] == t12
