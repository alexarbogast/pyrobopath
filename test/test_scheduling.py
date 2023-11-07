import unittest
from pyrobopath.scheduling import Event, Schedule, MultiAgentSchedule
from pyrobopath.scheduling import draw_schedule, draw_multi_agent_schedule
from pyrobopath.scheduling import DependencyGraph


class TestSchedule(unittest.TestCase):
    def setUp(self):
        self.schedule = Schedule()
        self.schedule.add_event(Event("eventA", 0.0, 5.0))
        self.schedule.add_event(Event("eventB", 5.0, 2.0))
        self.schedule.add_event(Event("eventC", 7.0, 5.0))
        self.schedule.add_event(Event("eventD", 12.0, 10.0))
        self.schedule.add_event(Event("eventE", 22.0, 45.0))
        self.schedule.add_event(Event("eventF", 67.0, 15.0))

    def test_schedule(self):
        self.assertEqual(self.schedule.start_time(), 0.0, "Start time != 0.0")
        self.assertEqual(self.schedule.end_time(), 82.0, "End time != 82.0")
        self.assertEqual(self.schedule.duration(), 82.0, "Duration != 82.0")
        self.assertEqual(self.schedule.n_events(), 6, "Number of events != 6")

        sliced = self.schedule.slice(0.0, 82.0)
        self.assertEqual(sliced.start_time(), 0.0, "Start time != 0.0")
        self.assertEqual(sliced.end_time(), 82.0, "End time != 82.0")
        self.assertEqual(sliced.duration(), 82.0, "Duration != 82.0")
        self.assertEqual(sliced.n_events(), 6, "Number of events != 6")

        sliced = self.schedule.slice(3.0, 11.0)
        self.assertEqual(sliced.start_time(), 0.0, "Start time != 0.0")
        self.assertEqual(sliced.end_time(), 12.0, "End time != 12.0")
        self.assertEqual(sliced.duration(), 12.0, "Duration != 12.0")
        self.assertEqual(sliced.n_events(), 3, "Number of events != 3")

        sliced = self.schedule.slice(-1.0, -0.1)
        self.assertEqual(sliced.n_events(), 0, "Number of events != 0")
        self.assertEqual(sliced.start_time(), -1.0)
        self.assertEqual(sliced.end_time(), -0.1)

        sliced = self.schedule.slice(-1.0, 0.0)
        self.assertEqual(sliced.n_events(), 1, "Number of events != 1")
        self.assertEqual(sliced.start_time(), 0.0, "Start time != 0.0")
        self.assertEqual(sliced.end_time(), 5.0, "End time != 5.0")

        sliced = self.schedule.slice(82.0, 83.0)
        self.assertEqual(sliced.n_events(), 1, "Number of events != 1")
        self.assertEqual(sliced.start_time(), 67.0)
        self.assertEqual(sliced.end_time(), 82.0)


class TestMultiAgentSchedule(unittest.TestCase):
    def test_schedule(self):
        schedule = MultiAgentSchedule()
        schedule.add_event(Event("eventA1", -1.0, 5.0), "agent1")
        schedule.add_event(Event("eventB1", 5.0, 2.0), "agent1")
        schedule.add_event(Event("eventC1", 7.0, 5.0), "agent1")
        schedule.add_event(Event("eventD1", 12.0, 10.0), "agent1")
        schedule.add_event(Event("eventE1", 22.0, 45.0), "agent1")
        schedule.add_event(Event("eventF1", 67.0, 15.0), "agent1")

        schedule.add_event(Event("eventA2", 0.0, 5.0), "agent2")
        schedule.add_event(Event("eventB2", 5.0, 4.0), "agent2")
        schedule.add_event(Event("eventC2", 9.0, 10.0), "agent2")
        schedule.add_event(Event("eventD2", 19.0, 10.0), "agent2")
        schedule.add_event(Event("eventE2", 67.0, 16.0), "agent2")

        self.assertEqual(schedule.start_time(), -1.0)
        self.assertEqual(schedule.end_time(), 83.0)
        self.assertEqual(schedule.duration(), 84.0)
        self.assertEqual(schedule.n_agents(), 2)

        other = Schedule()
        other.add_event(Event("eventA3", -2.0, 5.0))
        other.add_event(Event("eventB3", 70.0, 20.0))
        schedule.add_schedule(other, "agent3")

        self.assertEqual(schedule.start_time(), -2.0)
        self.assertEqual(schedule.end_time(), 90.0)
        self.assertEqual(schedule.duration(), 92.0)
        self.assertEqual(schedule.n_agents(), 3)

        # test other sequence functions
        self.assertEqual(schedule.first_started(), "agent3")
        self.assertEqual(schedule.last_started(), "agent2")
        self.assertEqual(schedule.first_finished(), "agent1")
        self.assertEqual(schedule.last_finished(), "agent3")

        schedule.add_agent("agent4")
        self.assertEqual(schedule.n_agents(), 4, "Number of agents != 4")
        self.assertEqual(schedule["agent3"], other)

        schedule.add_agents(["agent5", "agent6", "agent7"])
        self.assertEqual(schedule.n_agents(), 7, "Number of agents != 7")

    def test_slicing(self):
        schedule = MultiAgentSchedule()
        schedule.add_event(Event(None, start=-2.0, duration=2.0), "agent1")
        schedule.add_event(Event(None, start=1.0, duration=2.0), "agent1")

        schedule.add_event(Event(None, start=-1.0, duration=1.0), "agent2")
        schedule.add_event(Event(None, start=1.0, duration=3.0), "agent2")

        schedule.add_event(Event(None, start=-3.0, duration=3.0), "agent3")
        schedule.add_event(Event(None, start=2.0, duration=1.0), "agent3")

        sliced = schedule.slice(-3.0, 4.0)
        self.assertEqual(sliced.start_time(), -3.0)
        self.assertEqual(sliced.end_time(), 4.0)
        self.assertEqual(schedule.duration(), 7)
        self.assertEqual(schedule.n_agents(), 3)
        self.assertEqual(schedule.schedules["agent1"].n_events(), 2)
        self.assertEqual(schedule.schedules["agent2"].n_events(), 2)
        self.assertEqual(schedule.schedules["agent3"].n_events(), 2)

        sliced = schedule.slice(-4.0, -3.0)
        self.assertEqual(sliced.n_events(), 1)


class TestVisualization(unittest.TestCase):
    def test_visualization_api(self):
        self.schedule = Schedule()
        self.schedule.add_event(Event("eventA", 0.0, 5.0))
        self.schedule.add_event(Event("eventB", 5.0, 2.0))
        draw_schedule(self.schedule, show=False)

        self.multi_schedule = MultiAgentSchedule()
        self.multi_schedule.add_event(Event("eventA1", -1.0, 5.0), "agent1")
        self.multi_schedule.add_event(Event("eventB1", 5.0, 2.0), "agent1")
        self.multi_schedule.add_event(Event("eventA2", 0.0, 5.0), "agent2")
        self.multi_schedule.add_event(Event("eventB2", 5.0, 4.0), "agent2")
        draw_multi_agent_schedule(self.multi_schedule, show=False)


class TestDependencyGraph(unittest.TestCase):
    def test_create_dependency_graph(self):
        dg = DependencyGraph()
        dg.add_node(0, ["start"])

        can_start = dg.can_start(0)
        self.assertFalse(can_start, "node zero cannot start before start is completed")

        dg.set_complete("start")
        complete = dg._graph.nodes["start"]["complete"]
        self.assertTrue(complete, "'start' was not marked complete")

        dg.reset()
        can_start = dg.can_start(0)
        self.assertFalse(can_start, "node zero cannot start before start is completed")

        dg.set_complete("start")
        dg.set_complete(0)
        dg.add_node(1, [0])
        can_start = dg.can_start(1)
        self.assertTrue(can_start)


if __name__ == "__main__":
    unittest.main()