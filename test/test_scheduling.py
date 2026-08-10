import pytest

from pyrobopath.process import DependencyGraph
from pyrobopath.scheduling import Interval, Event, Schedule, MultiAgentSchedule


@pytest.fixture
def schedule():
    schedule = Schedule()
    schedule.add_event(Event(0.0, 5.0, "eventA"))
    schedule.add_event(Event(5.0, 7.0, "eventB"))
    schedule.add_event(Event(7.0, 12.0, "eventC"))
    schedule.add_event(Event(12.0, 22.0, "eventD"))
    schedule.add_event(Event(22.0, 67.0, "eventE"))
    schedule.add_event(Event(67.0, 82.0, "eventF"))
    return schedule


class TestSchedule:
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
        assert ans == [True] + [False] * 12

        # meets
        i1 = Interval(1, 2)
        i2 = Interval(2, 3)
        ans = test_all_relations(i1, i2)
        assert ans == [False] + [True] + [False] * 11

        # overlaps
        i1 = Interval(1, 3)
        i2 = Interval(2, 4)
        ans = test_all_relations(i1, i2)
        assert ans == [False] * 2 + [True] + [False] * 10

        # starts
        i1 = Interval(2, 3)
        i2 = Interval(2, 4)
        ans = test_all_relations(i1, i2)
        assert ans == [False] * 3 + [True] + [False] * 9

        # during
        i1 = Interval(2.5, 3)
        i2 = Interval(2, 4)
        ans = test_all_relations(i1, i2)
        assert ans == [False] * 4 + [True] + [False] * 8

        # finishes
        i1 = Interval(3, 4)
        i2 = Interval(2, 4)
        ans = test_all_relations(i1, i2)
        assert ans == [False] * 5 + [True] + [False] * 7

        # equals
        i1 = Interval(1, 3)
        i2 = Interval(1, 3)
        ans = test_all_relations(i1, i2)
        assert ans == [False] * 6 + [True] + [False] * 6

        # finished_by
        i1 = Interval(1, 3)
        i2 = Interval(2, 3)
        ans = test_all_relations(i1, i2)
        assert ans == [False] * 7 + [True] + [False] * 5

        # contains
        i1 = Interval(0, 3)
        i2 = Interval(1, 2)
        ans = test_all_relations(i1, i2)
        assert ans == [False] * 8 + [True] + [False] * 4

        # started_by
        i1 = Interval(1, 3)
        i2 = Interval(1, 2)
        ans = test_all_relations(i1, i2)
        assert ans == [False] * 9 + [True] + [False] * 3

        # overlapped_by
        i1 = Interval(2, 4)
        i2 = Interval(1, 3)
        ans = test_all_relations(i1, i2)
        assert ans == [False] * 10 + [True] + [False] * 2

        # met_by
        i1 = Interval(3, 4)
        i2 = Interval(1, 3)
        ans = test_all_relations(i1, i2)
        assert ans == [False] * 11 + [True] + [False]

        # preceded_by
        i1 = Interval(5, 6)
        i2 = Interval(1, 4)
        ans = test_all_relations(i1, i2)
        assert ans == [False] * 12 + [True]

    def test_schedule(self, schedule):
        assert schedule.start_time() == 0.0
        assert schedule.end_time() == 82.0
        assert schedule.duration() == 82.0
        assert schedule.n_events() == 6

        sliced = schedule.slice(0.0, 82.0)
        assert sliced.start_time() == 0.0
        assert sliced.end_time() == 82.0
        assert sliced.duration() == 82.0
        assert sliced.n_events() == 6

        sliced = schedule.slice(3.0, 11.0)
        assert sliced.start_time() == 0.0
        assert sliced.end_time() == 12.0
        assert sliced.duration() == 12.0
        assert sliced.n_events() == 3

        sliced = schedule.slice(-1.0, -0.1)
        assert sliced.n_events() == 0
        assert sliced.start_time() == -1.0
        assert sliced.end_time() == -0.1

        sliced = schedule.slice(-1.0, 0.0)
        assert sliced.n_events() == 1
        assert sliced.start_time() == 0.0
        assert sliced.end_time() == 5.0

        sliced = schedule.slice(82.0, 83.0)
        assert sliced.n_events() == 1
        assert sliced.start_time() == 67.0
        assert sliced.end_time() == 82.0

        ind_sliced = schedule.slice_ind(0.0, 82.0)
        assert ind_sliced == [0, 1, 2, 3, 4, 5]

        ind_sliced = schedule.slice_ind(3.0, 11.0)
        assert ind_sliced == [0, 1, 2]


class TestMultiAgentSchedule:
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

        assert schedule.start_time() == -1.0
        assert schedule.end_time() == 82.0
        assert schedule.duration() == 83.0
        assert schedule.n_agents() == 2

        other = Schedule()
        other.add_event(Event(-2.0, 3.0, "eventA3"))
        other.add_event(Event(70.0, 90.0, "eventB3"))
        schedule.add_schedule(other, "agent3")

        assert schedule.start_time() == -2.0
        assert schedule.end_time() == 90.0
        assert schedule.duration() == 92.0
        assert schedule.n_agents() == 3

        # test other sequence functions
        assert schedule.first_started() == "agent3"
        assert schedule.last_started() == "agent2"
        assert schedule.first_finished() == "agent1"
        assert schedule.last_finished() == "agent3"

        schedule.add_agent("agent4")
        assert schedule.n_agents() == 4
        assert schedule["agent3"] == other

        schedule.add_agents(["agent5", "agent6", "agent7"])
        assert schedule.n_agents() == 7

    def test_slicing(self):
        schedule = MultiAgentSchedule()
        schedule.add_event(Event(start=-2.0, end=0.0), "agent1")
        schedule.add_event(Event(start=1.0, end=3.0), "agent1")

        schedule.add_event(Event(start=-1.0, end=0.0), "agent2")
        schedule.add_event(Event(start=1.0, end=4.0), "agent2")

        schedule.add_event(Event(start=-3.0, end=0.0), "agent3")
        schedule.add_event(Event(start=2.0, end=3.0), "agent3")

        sliced = schedule.slice(-3.0, 4.0)
        assert sliced.start_time() == -3.0
        assert sliced.end_time() == 4.0
        assert schedule.duration() == 7.0
        assert schedule.n_agents() == 3
        assert schedule.schedules["agent1"].n_events() == 2
        assert schedule.schedules["agent2"].n_events() == 2
        assert schedule.schedules["agent3"].n_events() == 2

        sliced = schedule.slice(-4.0, -3.0)
        assert sliced.n_events() == 1


class TestVisualization:
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


class TestDependencyGraph:
    def test_create_dependency_graph(self):
        dg = DependencyGraph()
        dg.add_node(0, ["start"])

        can_start = dg.can_start(0)
        assert not can_start

        dg.mark_complete("start")
        can_start = dg.can_start(0)
        assert can_start

        dg.reset()
        can_start = dg.can_start(0)
        assert not can_start

        roots = dg.roots()
        assert roots == ["start"]

        dg.mark_complete("start")
        dg.mark_complete(0)
        dg.add_node(1, [0])
        dg.add_node(2, ["start", 1])
        can_start = dg.can_start(1)
        assert can_start
        assert dg.pending_tasks() == [1, 2]
