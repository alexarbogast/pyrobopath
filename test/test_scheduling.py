import unittest

from pyrobopath.process import DependencyGraph
from pyrobopath.scheduling import Interval, Event, Schedule, MultiAgentSchedule


class TestSchedule(unittest.TestCase):
    def setUp(self):
        self.schedule = Schedule()
        self.schedule.add_event(Event(0.0, 5.0, "eventA"))
        self.schedule.add_event(Event(5.0, 7.0, "eventB"))
        self.schedule.add_event(Event(7.0, 12.0, "eventC"))
        self.schedule.add_event(Event(12.0, 22.0, "eventD"))
        self.schedule.add_event(Event(22.0, 67.0, "eventE"))
        self.schedule.add_event(Event(67.0, 82.0, "eventF"))

    def test_interval(self):
        def test_all_relations(i1, i2):
            ans = []
            ans.append(i1.precedes(i2))
            ans.append(i1.meets(i2))
            ans.append(i1.overlaps(i2))
            ans.append(i1.starts(i2))
            ans.append(i1.during(i2))
            ans.append(i1.finishes(i2))
            ans.append(i1.equals(i2))
            ans.append(i1.finished_by(i2))
            ans.append(i1.contains(i2))
            ans.append(i1.started_by(i2))
            ans.append(i1.overlapped_by(i2))
            ans.append(i1.met_by(i2))
            ans.append(i1.preceded_by(i2))
            return ans

        # preceedes
        i1 = Interval(1, 2)
        i2 = Interval(3, 4)
        ans = test_all_relations(i1, i2)
        self.assertListEqual(ans, [True] + [False] * 12)

        # meets
        i1 = Interval(1, 2)
        i2 = Interval(2, 3)
        ans = test_all_relations(i1, i2)
        self.assertListEqual(ans, [False] + [True] + [False] * 11)

        # overlaps
        i1 = Interval(1, 3)
        i2 = Interval(2, 4)
        ans = test_all_relations(i1, i2)
        self.assertListEqual(ans, [False] * 2 + [True] + [False] * 10)

        # starts
        i1 = Interval(2, 3)
        i2 = Interval(2, 4)
        ans = test_all_relations(i1, i2)
        self.assertListEqual(ans, [False] * 3 + [True] + [False] * 9)

        # during
        i1 = Interval(2.5, 3)
        i2 = Interval(2, 4)
        ans = test_all_relations(i1, i2)
        self.assertListEqual(ans, [False] * 4 + [True] + [False] * 8)

        # finishes
        i1 = Interval(3, 4)
        i2 = Interval(2, 4)
        ans = test_all_relations(i1, i2)
        self.assertListEqual(ans, [False] * 5 + [True] + [False] * 7)

        # equals
        i1 = Interval(1, 3)
        i2 = Interval(1, 3)
        ans = test_all_relations(i1, i2)
        self.assertListEqual(ans, [False] * 6 + [True] + [False] * 6)

        # finished_by
        i1 = Interval(1, 3)
        i2 = Interval(2, 3)
        ans = test_all_relations(i1, i2)
        self.assertListEqual(ans, [False] * 7 + [True] + [False] * 5)

        # contains
        i1 = Interval(0, 3)
        i2 = Interval(1, 2)
        ans = test_all_relations(i1, i2)
        self.assertListEqual(ans, [False] * 8 + [True] + [False] * 4)

        # started_by
        i1 = Interval(1, 3)
        i2 = Interval(1, 2)
        ans = test_all_relations(i1, i2)
        self.assertListEqual(ans, [False] * 9 + [True] + [False] * 3)

        # overlapped_by
        i1 = Interval(2, 4)
        i2 = Interval(1, 3)
        ans = test_all_relations(i1, i2)
        self.assertListEqual(ans, [False] * 10 + [True] + [False] * 2)

        # met_by
        i1 = Interval(3, 4)
        i2 = Interval(1, 3)
        ans = test_all_relations(i1, i2)
        self.assertListEqual(ans, [False] * 11 + [True] + [False])

        # preceded_by
        i1 = Interval(5, 6)
        i2 = Interval(1, 4)
        ans = test_all_relations(i1, i2)
        self.assertListEqual(ans, [False] * 12 + [True])

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

        ind_sliced = self.schedule.slice_ind(0.0, 82.0)
        self.assertListEqual(ind_sliced, [0, 1, 2, 3, 4, 5])

        ind_sliced = self.schedule.slice_ind(3.0, 11.0)
        self.assertListEqual(ind_sliced, [0, 1, 2])


class TestMultiAgentSchedule(unittest.TestCase):
    def test_schedule(self):
        schedule = MultiAgentSchedule()
        schedule.add_event(Event(-1.0, 4.0, "eventA1"), "agent1")
        schedule.add_event(Event(4.0, 6.0, "eventB1"), "agent1")
        schedule.add_event(Event(6.0, 11.0, "eventC1"), "agent1")
        schedule.add_event(Event(11.0, 21.0, "eventD1"), "agent1")
        schedule.add_event(Event(21.0, 66.0, "eventE1"), "agent1")
        schedule.add_event(Event(66.0, 81.0, "eventF1"), "agent1")

        schedule.add_event(Event(0.0, 5.0, "eventA2"), "agent2")
        schedule.add_event(Event(5.0, 9.0, "eventB2"), "agent2")
        schedule.add_event(Event(9.0, 19.0, "eventC2"), "agent2")
        schedule.add_event(Event(19.0, 29.0, "eventD2"), "agent2")
        schedule.add_event(Event(67.0, 82.0, "eventE2"), "agent2")

        self.assertEqual(schedule.start_time(), -1.0)
        self.assertEqual(schedule.end_time(), 82.0)
        self.assertEqual(schedule.duration(), 83.0)
        self.assertEqual(schedule.n_agents(), 2)

        other = Schedule()
        other.add_event(Event(-2.0, 3.0, "eventA3"))
        other.add_event(Event(70.0, 90.0, "eventB3"))
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
        schedule.add_event(Event(start=-2.0, end=0.0), "agent1")
        schedule.add_event(Event(start=1.0, end=3.0), "agent1")

        schedule.add_event(Event(start=-1.0, end=0.0), "agent2")
        schedule.add_event(Event(start=1.0, end=4.0), "agent2")

        schedule.add_event(Event(start=-3.0, end=0.0), "agent3")
        schedule.add_event(Event(start=2.0, end=3.0), "agent3")

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
        self.schedule.add_event(Event(0.0, 5.0))
        self.schedule.add_event(Event(5.0, 2.0))
        # draw_schedule(self.schedule, show=False)

        self.multi_schedule = MultiAgentSchedule()
        self.multi_schedule.add_event(Event(-1.0, 5.0), "agent1")
        self.multi_schedule.add_event(Event(5.0, 2.0), "agent1")
        self.multi_schedule.add_event(Event(0.0, 5.0), "agent2")
        self.multi_schedule.add_event(Event(5.0, 4.0), "agent2")
        # draw_multi_agent_schedule(self.multi_schedule, show=False)


class TestDependencyGraph(unittest.TestCase):
    def test_create_dependency_graph(self):
        dg = DependencyGraph()
        dg.add_node(0, ["start"])

        can_start = dg.can_start(0)
        self.assertFalse(can_start, "node zero cannot start before start is completed")

        dg.mark_complete("start")
        can_start = dg.can_start(0)
        self.assertTrue(can_start, "node 0 cannot start after 'start' marked complete")

        dg.reset()
        can_start = dg.can_start(0)
        self.assertFalse(can_start, "dependency graph was not reset")

        roots = dg.roots()
        self.assertEqual(roots, ["start"])

        dg.mark_complete("start")
        dg.mark_complete(0)
        dg.add_node(1, [0])
        dg.add_node(2, ["start", 1])
        can_start = dg.can_start(1)
        self.assertTrue(can_start)
        self.assertEqual(dg.pending_tasks(), [1, 2])


if __name__ == "__main__":
    unittest.main()
