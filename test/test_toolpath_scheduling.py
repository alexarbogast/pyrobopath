import unittest
import numpy as np

from pyrobopath.toolpath import Contour, Toolpath
from pyrobopath.toolpath_scheduling import ToolpathScheduler, PlanningOptions
from pyrobopath.scheduling import DependencyGraph


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
