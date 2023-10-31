import unittest
import numpy as np

from pyrobopath.toolpath import Contour
from pyrobopath.toolpath_scheduling import ToolpathScheduler

class TestToolpathScheduling(unittest.TestCase):
    def setUp(self):
        self.capabilities = {
            "agent1": 0,
            "agent2": 1,
            "agent3": 1,
        }
        self.ts = ToolpathScheduler(self.capabilities)


    def test_toolpath_scheduler_single_agent(self):
        c0 = Contour(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        c0.tool = 0

        capabilities = {"agent1": 0}
        ts = ToolpathScheduler(capabilities)
