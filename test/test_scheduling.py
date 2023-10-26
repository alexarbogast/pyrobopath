import unittest
from scheduling.schedule import Event, Schedule, MultiAgentSchedule


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


class TestMultiAgentSchedule(unittest.TestCase):
    def setUp(self):
        self.schedule = MultiAgentSchedule()

        self.schedule.add_event(Event("eventA1", -1.0, 5.0), "agent1")
        self.schedule.add_event(Event("eventB1", 5.0, 2.0), "agent1")
        self.schedule.add_event(Event("eventC1", 7.0, 5.0), "agent1")
        self.schedule.add_event(Event("eventD1", 12.0, 10.0), "agent1")
        self.schedule.add_event(Event("eventE1", 22.0, 45.0), "agent1")
        self.schedule.add_event(Event("eventF1", 67.0, 15.0), "agent1")

        self.schedule.add_event(Event("eventA2", 0.0, 5.0), "agent2")
        self.schedule.add_event(Event("eventB2", 5.0, 4.0), "agent2")
        self.schedule.add_event(Event("eventC2", 9.0, 10.0), "agent2")
        self.schedule.add_event(Event("eventD2", 19.0, 10.0), "agent2")
        self.schedule.add_event(Event("eventE2", 67.0, 16.0), "agent2")

    def test_schedule(self):
        self.assertEqual(self.schedule.start_time(), -1.0)
        self.assertEqual(self.schedule.end_time(), 83.0)
        self.assertEqual(self.schedule.duration(), 84.0)
        self.assertEqual(self.schedule.n_agents(), 2)


if __name__ == "__main__":
    unittest.main()
