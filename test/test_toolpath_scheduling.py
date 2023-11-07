import unittest
import numpy as np

from pyrobopath.toolpath import Contour, Toolpath
from pyrobopath.scheduling import DependencyGraph, MultiAgentSchedule
from pyrobopath.collision_detection import FCLRobotBBCollisionModel
from pyrobopath.toolpath_scheduling import *


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
            [np.array([0.0, 0.0, 0.0]),
             np.array([0.0, 1.0, 0.0]),
             np.array([1.0, 1.0, 0.0])], 
             tool=0
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
    def setUp(self):
        self.collision_models = {
            "agent1": FCLRobotBBCollisionModel(3.0, 0.2, 1.0, [-5.0, 0.0, 0.0]),
            "agent2": FCLRobotBBCollisionModel(3.0, 0.2, 1.0, [5.0, 0.0, 0.0]),
        }

    def test_event_causes_collision(self):
        c1 = Contour([np.array([0.0, 2.0, 0.0]), np.array([0.0, -2.0, 0.0])])
        c2 = Contour([np.array([2.0, 0.0, 0.0]), np.array([-2.0, 0.0, 0.0])])

        schedule = MultiAgentSchedule()

        # no collision
        event1 = ContourEvent(c1, 0.0, 1.0)
        collide = event_causes_collision(event1, "agent1", schedule, self.collision_models)        
        self.assertFalse(collide)
        schedule.add_event(event1, "agent1")

        # collision
        event2 = ContourEvent(c2, 0.0, 1.0)
        collide = event_causes_collision(event1, "agent2", schedule, self.collision_models)        
        self.assertTrue(collide)

        # no collision
        event3 = ContourEvent(c2, 2.0, 1.0)
        collide = event_causes_collision(event3, "agent2", schedule, self.collision_models)
        self.assertFalse(collide)
        schedule.add_event(event3, "agent2")

        # no collision
        c3 = Contour([np.array([-2.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])])
        event4 = ContourEvent(c3, 4.0, 1.0)
        collide = event_causes_collision(event4, "agent2", schedule, self.collision_models)
        self.assertFalse(collide)
        
        # collision
        c4 = Contour([np.array([-2.0, 0.0, 0.0]), np.array([0.0, -2.0, 0.0])])
        event5 = ContourEvent(c4, 4.0, 1.0)
        collide = event_causes_collision(event5, "agent2", schedule, self.collision_models)
        self.assertTrue(collide)


if __name__ == "__main__":
    unittest.main()
