import unittest
import numpy as np

from pyrobopath.toolpath import Contour, Toolpath
from pyrobopath.scheduling import DependencyGraph
from pyrobopath.collision_detection import FCLRobotBBCollisionModel
from pyrobopath.toolpath_scheduling import *


class TestToolpathSchedule(unittest.TestCase):
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
        schedule.add_event(ContourEvent(contour1, 0.0, 1.0))
        schedule.add_event(ContourEvent(contour2, 5.0, 1.0))

        self.assertEqual(schedule.duration(), 7)

        default = "default"
        state = schedule.get_state(-1.0, default)
        self.assertEqual(state, default)

        state = schedule.get_state(0.0, default)
        self.assertTrue(np.all(state == np.array([0.0, 1.0, 0.0])))

        state = schedule.get_state(0.5, default)
        self.assertTrue(np.all(state == np.array([0.0, 0.5, 0.0])))

        state = schedule.get_state(2.0, default)
        self.assertTrue(np.all(state == np.array([0.0, -1.0, 0.0])))

        state = schedule.get_state(3.0, default)
        self.assertTrue(np.all(state == np.array([0.0, -1.0, 0.0])))

        state = schedule.get_state(5.0, default)
        self.assertTrue(np.all(state == np.array([0.0, -1.0, 0.0])))

        state = schedule.get_state(7.0, default)
        self.assertTrue(np.all(state == np.array([1.0, 0.0, 0.0])))

        state = schedule.get_state(8.0, default)
        self.assertTrue(np.all(state == np.array([1.0, 0.0, 0.0])))


class TestToolpathScheduling(unittest.TestCase):
    def setUp(self):
        self.capabilities = {
            "agent1": 0,
            "agent2": 1,
            "agent3": 1,
        }
        self.ts = ToolpathScheduler(self.capabilities)

    def test_toolpath_scheduler_single_agent(self):
        c0 = Contour(
            [
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([1.0, 1.0, 0.0]),
            ],
            tool=0,
        )

        toolpath = Toolpath()
        toolpath.contours.append(c0)
        dg = DependencyGraph()
        dg.add_node(0, ["start"])

        capabilities = {"agent1": [0]}
        ts = ToolpathScheduler(capabilities)

        options = PlanningOptions(velocity=1.0, retract_height=0.0)
        schedule = ts.schedule(toolpath, dg, options)

        self.assertEqual(schedule.duration(), 2.0, "Schedule duration is not 2.0")


class TestToolpathCollision(unittest.TestCase):
    def test_event_causes_collision_two(self):
        agent1 = AgentModel()
        agent1.base_frame_position = [-5.0, 0.0, 0.0]
        agent1.home_position = [-3.0, 0.0, 0.0]
        agent1.collision_model = FCLRobotBBCollisionModel(
            3.0, 0.2, 1.0, agent1.base_frame_position
        )

        agent2 = AgentModel()
        agent2.base_frame_position = [5.0, 0.0, 0.0]
        agent2.home_position = [3.0, 0.0, 0.0]
        agent2.collision_model = FCLRobotBBCollisionModel(
            3.0, 0.2, 1.0, agent2.base_frame_position
        )

        agent_models = {"agent1": agent1, "agent2": agent2}

        c1 = Contour([np.array([0.0, 2.0, 0.0]), np.array([0.0, -2.0, 0.0])])
        c2 = Contour([np.array([2.0, 0.0, 0.0]), np.array([-2.0, 0.0, 0.0])])

        schedule = MultiAgentToolpathSchedule()

        # no collision
        event1 = ContourEvent(c1, 0.0, 1.0)
        collide = event_causes_collision(event1, "agent1", schedule, agent_models)
        self.assertFalse(collide)
        schedule.add_event(event1, "agent1")

        # collision
        event2 = ContourEvent(c2, 0.0, 1.0)
        collide = event_causes_collision(event2, "agent2", schedule, agent_models)
        self.assertTrue(collide)

        # no collision
        event3 = ContourEvent(c2, 2.0, 1.0)
        collide = event_causes_collision(event3, "agent2", schedule, agent_models)
        self.assertFalse(collide)
        schedule.add_event(event3, "agent2")

        # no collision
        c3 = Contour([np.array([-2.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])])
        event4 = ContourEvent(c3, 6.0, 1.0)
        collide = event_causes_collision(event4, "agent2", schedule, agent_models)
        self.assertFalse(collide)

        # collision
        c4 = Contour([np.array([-2.0, 0.0, 0.0]), np.array([0.0, -2.0, 0.0])])
        event5 = ContourEvent(c4, 6.0, 1.0)
        collide = event_causes_collision(event5, "agent2", schedule, agent_models)        
        self.assertTrue(collide)

    def test_event_causes_collision_three(self):
        agent1 = AgentModel()
        agent1.base_frame_position = [-5.0, 0.0, 0.0]
        agent1.home_position = [-3.0, 0.0, 0.0]
        agent1.collision_model = FCLRobotBBCollisionModel(
            3.0, 0.2, 1.0, agent1.base_frame_position
        )

        agent2 = AgentModel()
        agent2.base_frame_position = [5.0, 0.0, 0.0]
        agent2.home_position = [3.0, 0.0, 0.0]
        agent2.collision_model = FCLRobotBBCollisionModel(
            3.0, 0.2, 1.0, agent2.base_frame_position
        )

        agent3 = AgentModel()
        agent3.base_frame_position = [0.0, 5.0, 0.0]
        agent3.home_position = [0.0, 3.0, 0.0]
        agent3.collision_model = FCLRobotBBCollisionModel(
            3.0, 0.2, 1.0, agent3.base_frame_position
        )

        agent_models = {"agent1": agent1, "agent2": agent2, "agent3": agent3}

        c1 = Contour([np.array([-4.0, 0.0, 0.0]), np.array([-2.0, 0.0, 0.0])])
        c2 = Contour([np.array([4.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])])
        c3 = Contour([np.array([0.0, 4.0, 0.0]), np.array([0.0, 2.0, 0.0])])

        schedule = MultiAgentToolpathSchedule()

        # no collision
        event1 = ContourEvent(c1, 0.0, 1.0)
        collide = event_causes_collision(event1, "agent1", schedule, agent_models)
        self.assertFalse(collide)
        schedule.add_event(event1, "agent1")

        event2 = ContourEvent(c2, 0.0, 1.0)
        collide = event_causes_collision(event2, "agent2", schedule, agent_models)
        self.assertFalse(collide)
        schedule.add_event(event2, "agent2")

        event3 = ContourEvent(c3, 0.0, 1.0)
        collide = event_causes_collision(event3, "agent3", schedule, agent_models)
        self.assertFalse(collide)
        schedule.add_event(event3, "agent3")

        # collision (agent1 - agent2)
        c4 = Contour([np.array([-2.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0])])
        event4 = ContourEvent(c4, 2.0, 1.0)
        collide = event_causes_collision(event4, "agent1", schedule, agent_models)
        self.assertTrue(collide, "Colliding event returned False")

        event5 = ContourEvent(c4, 2.1, 1.0)
        collide = event_causes_collision(event5, "agent1", schedule, agent_models)
        self.assertTrue(collide, "Colliding event returned False")

        # no collision (agent1 - agent3)
        c5 = Contour([np.array([-2.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])])
        event6 = ContourEvent(c5, 3.0, 1.0)
        collide = event_causes_collision(event6, "agent1", schedule, agent_models)
        self.assertFalse(collide, "Non-colliding event returned True")
        schedule.add_event(event6, "agent1")

        # collision (agent1 - agent3)
        c6 = Contour([np.array([0.0, 2.0, 0.0]), np.array([0.0, 0.0, 0.0])])
        event7 = ContourEvent(c6, 3.0, 1.0)
        collide = event_causes_collision(event7, "agent3", schedule, agent_models)
        self.assertTrue(collide, "Colliding event returned False")

        c7 = Contour(
            [
                np.array([-1.0, 1.0, 0.0]),
                np.array([-1.0, -1.0, 0.0]),
                np.array([-2.0, -1.0, 0.0]),
                np.array([-2.0, 1.0, 0.0]),
            ]
        )
        event8 = ContourEvent(c7, 6.0, 2.0)
        collide = event_causes_collision(event8, "agent1", schedule, agent_models)
        self.assertFalse(collide, "Non-colliding event returned True")
        schedule.add_event(event8, "agent1")

        c8 = Contour([np.array([0.0, 1.0, 0.0]), np.array([0.0, -0.4, 0.0])])
        event9 = ContourEvent(c8, 6.0, 1.0)
        collide = event_causes_collision(event9, "agent2", schedule, agent_models)
        self.assertFalse(collide, "Non-colliding event returned True")
        schedule.add_event(event9, "agent2")

        c9 = Contour([np.array([0.0, 2.0, 0.0]), np.array([-1.0, 1.0, 0.0])])
        event10 = ContourEvent(c9, 5.0, 5.0)
        collide = event_causes_collision(event10, "agent3", schedule, agent_models)
        schedule.add_event(event10, "agent3")

        self.assertTrue(collide, "Colliding event returned False")

    def test_schedule_to_trajectory(self):
        c1 = Contour([np.array([0.0, 2.0, 0.0]), np.array([0.0, -2.0, 0.0])])
        c2 = Contour([np.array([2.0, 0.0, 0.0]), np.array([-2.0, 0.0, 0.0])])

        event1 = ContourEvent(c1, 0.0, 1.0)
        event2 = ContourEvent(c2, 6.0, 1.0)

        schedule = ToolpathSchedule()
        schedule.add_event(event1)
        schedule.add_event(event2)

        default = "default"
        traj = schedule_to_trajectory(
            schedule, t_start=-0.2, t_end=-0.1, default_state=default
        )
        self.assertEqual(traj.n_points(), 2)
        self.assertEqual(traj.start_time(), -0.2)
        self.assertEqual(traj.end_time(), -0.1)
        self.assertEqual(traj.points[0].data, default)
        self.assertEqual(traj.points[-1].data, default)

        traj = schedule_to_trajectory(
            schedule, t_start=-0.2, t_end=0.0, default_state=default
        )
        self.assertEqual(traj.n_points(), 2)
        self.assertEqual(traj.start_time(), -0.2)
        self.assertEqual(traj.end_time(), 0.0)
        self.assertEqual(traj.points[0].data, default)
        self.assertTrue(np.all(traj.points[-1].data == c1.path[0]))

        traj = schedule_to_trajectory(
            schedule, t_start=0.0, t_end=2.0, default_state=default
        )
        self.assertEqual(traj.n_points(), 2)
        self.assertEqual(traj.start_time(), 0.0)
        self.assertEqual(traj.end_time(), 2.0)
        self.assertTrue(np.all(traj.points[0].data == c1.path[0]))
        self.assertTrue(np.all(traj.points[-1].data == np.array([0.0, 0.0, 0.0])))

        traj = schedule_to_trajectory(
            schedule, t_start=2.0, t_end=5.0, default_state=default
        )
        self.assertEqual(traj.n_points(), 3)
        self.assertEqual(traj.start_time(), 2.0)
        self.assertEqual(traj.end_time(), 5.0)
        self.assertTrue(np.all(traj.points[0].data == np.array([0.0, 0.0, 0.0])))
        self.assertTrue(np.all(traj.points[1].data == c1.path[1]))
        self.assertTrue(np.all(traj.points[2].data == c1.path[1]))

        traj = schedule_to_trajectory(
            schedule, t_start=5.0, t_end=11.0, default_state=default
        )
        self.assertEqual(traj.n_points(), 4)
        self.assertEqual(traj.start_time(), 5.0)
        self.assertEqual(traj.end_time(), 11.0)
        self.assertTrue(np.all(traj.points[0].data == c1.path[1]))
        self.assertTrue(np.all(traj.points[1].data == c2.path[0]))
        self.assertTrue(np.all(traj.points[2].data == c2.path[1]))
        self.assertTrue(np.all(traj.points[3].data == c2.path[1]))

        traj = schedule_to_trajectory(
            schedule, t_start=10.0, t_end=11.0, default_state=default
        )
        self.assertEqual(traj.n_points(), 2)
        self.assertEqual(traj.start_time(), 10.0)
        self.assertEqual(traj.end_time(), 11.0)
        self.assertTrue(np.all(traj.points[0].data == c2.path[1]))
        self.assertTrue(np.all(traj.points[1].data == c2.path[1]))

        traj = schedule_to_trajectory(
            schedule, t_start=6.0, t_end=10.0, default_state=default
        )
        self.assertEqual(traj.n_points(), 2)
        self.assertEqual(traj.start_time(), 6.0)
        self.assertEqual(traj.end_time(), 10.0)
        self.assertTrue(np.all(traj.points[0].data == c2.path[0]))
        self.assertTrue(np.all(traj.points[1].data == c2.path[1]))


if __name__ == "__main__":
    unittest.main()
