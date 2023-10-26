from __future__ import annotations
from typing import List
import collections
import matplotlib.pyplot as plt

class Event(object):
    def __init__(self, data, start, duration):
        self.data = data
        self.start = start
        self.duration = duration


""" class Schedule(object):
    def __init__(self):
        self.events = collections.defaultdict(list)

    def add_event(self, event, agent: int = 0):
        self.events[agent].append(event)

    def draw(self):
        fig, ax = plt.subplots()
        for agent in self.events.keys():
            for event in self.events[agent]:
                p = ax.barh(agent, left=event.start, width=event.duration)
                ax.bar_label(p, label_type="center")
        ax.set_yticks(range(len(self.events)))
        plt.show()

    @property 
    def n_agents(self):
        return len(self.events.keys) """
    

class Schedule(object):
    def __init__(self):
        self._events: List[Event] = []
    
    def add_event(self, event: Event):
        self._events.append(event)

    def start_time(self):
        return min([e.start for e in self._events])
    
    def end_time(self):
        return max([e.start + e.duration for e in self._events])

    def duration(self):
        return self.end_time() - self.start_time()
    
    def n_events(self):
        return len(self._events)


class MultiAgentSchedule(object):
    def __init__(self):
        self.schedules = collections.defaultdict(Schedule)

    def add_event(self, event: Event, agent):
        self.schedules[agent].add_event(event)

    def start_time(self):
        return min([s.start_time() for s in self.schedules.values()])
    
    def end_time(self):
        return max([s.end_time() for s in self.schedules.values()])

    def duration(self):
        return self.end_time() - self.start_time()

    def n_agents(self):
        return len(self.schedules)
    
