from __future__ import annotations
from typing import List, Iterable, Hashable
import collections


class Interval(object):
    """
    An implementation of Allen's interval algebra

    See: https://cse.unl.edu/~choueiry/Documents/Allen-CACM1983.pdf
    """

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def precedes(self, other):
        """XXX YYY"""
        return self.end < other.start

    def meets(self, other):
        """
        XXXYYY
        """
        return self.end == other.start

    def overlaps(self, other):
        """
        | XXX
        |  YYY
        """
        return (
            self.start < other.start and self.end > other.start and self.end < other.end
        )

    def starts(self, other):
        """
        | XXX
        | YYYYY
        """
        return self.start == other.start and self.end < other.end

    def during(self, other):
        """
        |  XXX
        | YYYYY
        """
        return self.start > other.start and self.end < other.end

    def finishes(self, other):
        """
        |   XXX
        | YYYYY
        """
        return self.start > other.start and self.end == other.end

    def equals(self, other):
        """
        | XXX
        | YYY
        """
        return self.start == other.start and self.end == other.end

    def finished_by(self, other):
        """
        | XXXXX
        |   YYY
        """
        return self.start < other.start and self.end == other.end

    def contains(self, other):
        """
        | XXXXX
        |  YYY
        """
        return self.start < other.start and self.end > other.end

    def started_by(self, other):
        """
        | XXXXX
        | YYY
        """
        return self.start == other.start and self.end > other.end

    def overlapped_by(self, other):
        """
        |  XXX
        | YYY
        """
        return (
            self.start > other.start and self.start < other.end and self.end > other.end
        )

    def met_by(self, other):
        """
        YYYXXX
        """
        return self.start == other.end

    def preceded_by(self, other):
        """
        YYY XXX
        """
        return self.start > other.end


class Event(Interval):
    def __init__(self, start, end, data=None):
        super(Event, self).__init__(start, end)
        self.data = data

    @property
    def duration(self):
        return self.end - self.start


class Schedule(object):
    def __init__(self):
        self._events: List[Event] = []
        self._start_time = float("inf")
        self._end_time = float("-inf")

    def add_event(self, event: Event):
        if event.start < self._start_time:
            self._start_time = event.start
        if event.end > self._end_time:
            self._end_time = event.end
        self._events.append(event)

    def add_events(self, event: List[Event]):
        for e in event:
            self.add_event(e)

    def start_time(self):
        return self._start_time

    def end_time(self):
        return self._end_time

    def duration(self):
        return self.end_time() - self.start_time()

    def n_events(self):
        return len(self._events)

    def slice(self, t_start, t_end) -> Schedule:
        """
        Returns a new schedule with events that end at/after t_start and
        start at/before t_end

        {e | e.start <= t_end ∧ e.end >= t_start for all e in schedule}

        Note: Events are not sliced. If an event has any time in [t_start, t_end],
        the original event is included in its entirety.
        """

        new_sched = Schedule()
        new_sched._events = [self._events[i] for i in self.slice_ind(t_start, t_end)]
        if not new_sched._events:
            new_sched._start_time = t_start
            new_sched._end_time = t_end
            return new_sched

        new_sched._start_time = min([e.start for e in new_sched._events])
        new_sched._end_time = max([e.end for e in new_sched._events])
        return new_sched

    def slice_ind(self, t_start, t_end) -> List[int]:
        """
        Returns the indices of events that end at/after t_start and
        start at/before t_end

        {i | e[i].start <= t_end ∧ e[i].end >= t_start for all e in schedule}

        Note: Events are not sliced. If an event has any time in [t_start, t_end],
        the original event is included in its entirety.
        """

        filter = lambda e: e.end >= t_start and e.start <= t_end
        ind = [i for i, e in enumerate(self._events) if filter(e)]
        return ind


class MultiAgentSchedule(object):
    def __init__(self):
        self.schedules = collections.defaultdict(Schedule)
        self._start_time = 0.0
        self._end_time = 0.0

    def __getitem__(self, agent: Hashable):
        return self.schedules[agent]

    def add_agent(self, agent: Hashable):
        self.schedules[agent] = Schedule()

    def add_agents(self, agents: Iterable[Hashable]):
        for agent in agents:
            self.add_agent(agent)

    def agents(self):
        return self.schedules.keys()

    def add_event(self, event: Event, agent):
        end_time = event.start + event.duration
        if event.start < self._start_time:
            self._start_time = event.start
        if end_time > self._end_time:
            self._end_time = end_time
        self.schedules[agent].add_event(event)

    def add_events(self, events: List[Event], agent):
        self.schedules[agent].add_events(events)
        self._start_time = min(self._start_time, self.schedules[agent].start_time())
        self._end_time = max(self._end_time, self.schedules[agent].end_time())

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

    def n_events(self):
        """The number of events from all schedules"""
        return sum([s.n_events() for s in self.schedules.values()])

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

    def slice(self, t_start, t_end) -> MultiAgentSchedule:
        """Returns a new multi-agent schedule with the schedules for each agent
        filtered with events that end after t_start and start before t_end"""
        new_mas = MultiAgentSchedule()
        for agent, schedule in self.schedules.items():
            new_mas.add_schedule(schedule.slice(t_start, t_end), agent)

        if new_mas.n_events() == 0:
            new_mas._start_time = t_start
            new_mas._end_time = t_end
            return new_mas

        new_mas._start_time = min([s.start_time() for s in new_mas.schedules.values()])
        new_mas._end_time = max([s.end_time() for s in new_mas.schedules.values()])
        return new_mas
