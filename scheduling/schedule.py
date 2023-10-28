from __future__ import annotations
from typing import List
import collections
import matplotlib.pyplot as plt


class Event(object):
    def __init__(self, data, start, duration):
        self.data = data
        self.start = start
        self.duration = duration


class Schedule(object):
    def __init__(self):
        self._events: List[Event] = []
        self._start_time = 0.0
        self._end_time = 0.0

    def add_event(self, event: Event):
        end_time = event.start + event.duration
        if event.start < self._start_time:
            self._start_time = event.start
        if end_time > self._end_time:
            self._end_time = end_time
        self._events.append(event)

    def start_time(self):
        return self._start_time

    def end_time(self):
        return self._end_time

    def duration(self):
        return self.end_time() - self.start_time()

    def n_events(self):
        return len(self._events)


class MultiAgentSchedule(object):
    def __init__(self):
        self.schedules = collections.defaultdict(Schedule)
        self._start_time = 0.0
        self._end_time = 0.0

    def __getitem__(self, agent):
        return self.schedules[agent]

    def add_agent(self, agent):
        self.schedules[agent] = Schedule()

    def add_event(self, event: Event, agent):
        end_time = event.start + event.duration
        if event.start < self._start_time:
            self._start_time = event.start
        if end_time > self._end_time:
            self._end_time = end_time
        self.schedules[agent].add_event(event)

    def add_schedule(self, schedule: Schedule, agent):
        if schedule.start_time() < self._start_time:
            self._start_time = schedule.start_time()
        if schedule.end_time() > self._end_time:
            self._end_time = schedule.end_time()
        self.schedules[agent] = schedule

    def start_time(self):
        return self._start_time

    def end_time(self):
        return self._end_time

    def duration(self):
        """The duration of the combined schedule"""
        return self.end_time() - self.start_time()

    def n_agents(self):
        """The number of agents with schedules"""
        return len(self.schedules)

    def first_started(self):
        """Returns the agent belonging to the schedule that finishes first"""
        return min(self.schedules, key=lambda s: self.schedules[s].start_time())

    def last_started(self):
        """Returns the agent belonging to the schedule that finishes last"""
        return max(self.schedules, key=lambda s: self.schedules[s].start_time())

    def first_finished(self):
        """Returns the agent belonging to the schedule that finishes first"""
        return min(self.schedules, key=lambda s: self.schedules[s].end_time())

    def last_finished(self):
        """Returns the agent belonging to the schedule that finishes last"""
        return max(self.schedules, key=lambda s: self.schedules[s].end_time())
