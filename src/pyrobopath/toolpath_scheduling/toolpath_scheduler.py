from __future__ import annotations
from typing import Dict
import numpy as np
from dataclasses import dataclass

from ..toolpath import Toolpath, Contour
from ..scheduling import DependencyGraph

from .system_model import AgentModel
from .schedule import ContourEvent, MoveEvent, MultiAgentToolpathSchedule
from .toolpath_collision import events_cause_collision


@dataclass
class PlanningOptions:
    contour_velocity: float = 50.0
    travel_velocity: float = 100.0
    retract_height: float = 50.0
    collision_offset: float = 5.0
    collision_gap_threshold: float = 1.0


class SchedulingContext(object):
    def __init__(self, agent_models: Dict[str, AgentModel]):
        self.agent_models = agent_models
        self.reset()

    def reset(self):
        self.start_times = dict.fromkeys(self.agent_models.keys(), 0.0)
        self.positions = dict()
        for agent in self.agent_models:
            self.positions[agent] = self.agent_models[agent].home_position

    def get_agents_with_start_time(self, time):
        min_time_agents = [a for (a, t) in self.start_times.items() if t == time]
        return min_time_agents

    def get_unique_start_times(self):
        return sorted(set(self.start_times.values()))

    def set_agent_start_time(self, agent, time):
        self.start_times[agent] = time

    def get_current_position(self, agent):
        return self.positions[agent]


class TaskManager(object):
    def __init__(self, toolpath: Toolpath, dg: DependencyGraph):
        self.contours = toolpath.contours
        self.dg = dg

        # task sets
        self.frontier = set()
        self.completed_tasks = set()
        self.in_progress: Dict[str, float] = dict()

    def add_inprogress(self, id, t_end):
        self.in_progress[id] = t_end

    def mark_inprogress_complete(self, time):
        complete = [k for (k, v) in self.in_progress.items() if time >= v]
        self.completed_tasks.update(complete)
        for c in complete:
            self.dg.set_complete(c)
            self.in_progress.pop(c)

    def has_frontier(self):
        return bool(self.frontier)

    def get_available_tasks(self, *args):
        available = filter(lambda n: self.dg.can_start(n), self.frontier)
        for arg in args:
            available = filter(arg, available)
        return list(available)


class EventBuilder(object):
    def __init__(self, context: SchedulingContext):
        self.context = context

    def build_move_event(self, t_start, p_start, p_end, options: PlanningOptions):
        path = [p_start, p_end]
        event = MoveEvent(t_start, path, options.travel_velocity)
        return event

    def build_contour_event(self, t_start, contour: Contour, options: PlanningOptions):
        event = ContourEvent(t_start, contour, options.contour_velocity)
        return event


class MultiAgentToolpathPlanner(object):
    def __init__(self, agent_models: Dict[str, AgentModel]):
        self._agent_models = agent_models
        self._agents = agent_models.keys()

    def plan(self, toolpath: Toolpath, dg: DependencyGraph, options: PlanningOptions):
        schedule = MultiAgentToolpathSchedule()
        schedule.add_agents(self._agent_models.keys())

        context = SchedulingContext(self._agent_models)
        tm = TaskManager(toolpath, dg)
        eb = EventBuilder(context)

        tm.add_inprogress("start", 0.0)
        tm.frontier.update(dg._graph.successors("start"))

        while tm.has_frontier():
            sorted_times = context.get_unique_start_times()
            time = sorted_times[0]
            min_time_agents = context.get_agents_with_start_time(time)

            tm.mark_inprogress_complete(time)

            for agent in min_time_agents:
                tools = self._agent_models[agent].capabilities
                available = tm.get_available_tasks(
                    lambda n: tm.contours[n].tool in tools
                )

                if not available:
                    if len(sorted_times) > 1:
                        context.set_agent_start_time(agent, sorted_times[1])
                    continue

                # sort tasks by out-degree
                all_collide_flag = True
                nodes = sorted(available, key=lambda a: dg._graph.out_degree(a))
                for node in nodes:
                    contour = tm.contours[node]
                    p_start = schedule[agent].get_state(
                        time, self._agent_models[agent].home_position
                    )

                    travel_event = eb.build_move_event(
                        time, p_start, contour.path[0], options
                    )
                    contour_event = eb.build_contour_event(
                        travel_event.end, contour, options
                    )
                    home_event = eb.build_move_event(
                        contour_event.end,
                        contour.path[-1],
                        self._agent_models[agent].home_position,
                        options,
                    )

                    events = [travel_event, contour_event, home_event]

                    if events_cause_collision(
                        events,
                        agent,
                        schedule,
                        self._agent_models,
                        options.collision_gap_threshold,
                    ):
                        continue

                    # slice travel event if overlap
                    if schedule[agent].end_time() > travel_event.start:
                        prev_home_event = schedule[agent]._events.pop()
                        sliced_home = self._slice_home_event(
                            prev_home_event, travel_event.start
                        )
                        schedule.add_event(sliced_home, agent)

                    schedule.add_events(events, agent)
                    tm.add_inprogress(node, contour_event.end)
                    tm.frontier.remove(node)
                    tm.frontier.update(dg._graph.successors(node))
                    context.positions[agent] = events[1].data[-1]
                    context.set_agent_start_time(agent, contour_event.end)

                    all_collide_flag = False
                    break

                if all_collide_flag:
                    context.set_agent_start_time(agent, time + options.collision_offset)

        return schedule

    def _slice_home_event(self, home_event: MoveEvent, end_time: float):
        new_traj = home_event.traj.slice(home_event.start, end_time)
        path = [p.data for p in new_traj.points]
        return MoveEvent(home_event.start, path, home_event.velocity)
