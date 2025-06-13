from __future__ import annotations
from typing import Dict
from dataclasses import dataclass

from pyrobopath.toolpath import Toolpath, Contour
from pyrobopath.scheduling import DependencyGraph

from pyrobopath.toolpath_scheduling.system_model import AgentModel
from pyrobopath.toolpath_scheduling.schedule import (
    ContourEvent,
    MoveEvent,
    MultiAgentToolpathSchedule,
)
from pyrobopath.toolpath_scheduling.toolpath_collision import events_cause_collision


@dataclass
class PlanningOptions:
    retract_height: float = 50.0
    collision_offset: float = 5.0
    collision_gap_threshold: float = 1.0


class SchedulingContext(object):
    def __init__(self, agent_models: Dict[str, AgentModel], options: PlanningOptions):
        self.agent_models = agent_models
        self.options = options
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
        available = [n for n in self.frontier if self.dg.can_start(n)]
        return available


def build_event_chain(
    t_start, p_start, contour: Contour, agent, context: SchedulingContext
):
    # travel + approach event
    p_approach = contour.path[0].copy()
    p_approach[2] += context.options.retract_height
    path_travel = [p_start, p_approach, contour.path[0]]
    if (p_start == p_approach).all():
        path_travel.pop(0)
    e_travel = MoveEvent(
        t_start, path_travel, context.agent_models[agent].travel_velocity
    )

    # contour event
    e_contour = ContourEvent(
        e_travel.end, contour, context.agent_models[agent].velocity
    )

    # depart + home events
    p_depart = contour.path[-1].copy()
    p_depart[2] += context.options.retract_height
    e_depart = MoveEvent(
        e_contour.end,
        [contour.path[-1], p_depart],
        context.agent_models[agent].travel_velocity,
    )
    e_home = MoveEvent(
        e_depart.end,
        [p_depart, context.agent_models[agent].home_position],
        context.agent_models[agent].travel_velocity,
    )

    return [e_travel, e_contour, e_depart, e_home]


class MultiAgentToolpathPlanner(object):
    def __init__(self, agent_models: Dict[str, AgentModel]):
        self._agent_models = agent_models
        self._agents = agent_models.keys()

    def plan(self, toolpath: Toolpath, dg: DependencyGraph, options: PlanningOptions):
        schedule = MultiAgentToolpathSchedule()
        schedule.add_agents(self._agent_models.keys())

        context = SchedulingContext(self._agent_models, options)
        tm = TaskManager(toolpath, dg)

        tm.add_inprogress("start", 0.0)
        tm.frontier.update(dg._graph.successors("start"))

        while tm.has_frontier():
            sorted_times = context.get_unique_start_times()
            time = sorted_times[0]
            min_time_agents = context.get_agents_with_start_time(time)

            tm.mark_inprogress_complete(time)

            for agent in min_time_agents:
                tools = self._agent_models[agent].capabilities
                available = tm.get_available_tasks()
                available = [c for c in available if tm.contours[c].tool in tools]

                if not available:
                    if len(sorted_times) > 1:
                        context.set_agent_start_time(agent, sorted_times[1])
                    continue

                # prioritize tasks with highest out-degree
                key = lambda n: dg._graph.out_degree(n)
                nodes = sorted(available, key=key, reverse=True)  # type: ignore

                all_collide_flag = True
                for node in nodes:
                    contour = tm.contours[node]
                    p_start = schedule[agent].get_state(
                        time, self._agent_models[agent].home_position
                    )

                    events = build_event_chain(time, p_start, contour, agent, context)
                    if events_cause_collision(
                        events,
                        agent,
                        schedule,
                        self._agent_models,
                        options.collision_gap_threshold,
                    ):
                        continue

                    # slice home event if overlap
                    if schedule[agent].end_time() > events[0].start:
                        prev_home_event = schedule[agent]._events.pop()
                        if prev_home_event.start != events[0].start:
                            sliced_home = self._slice_home_event(
                                prev_home_event, events[0].start
                            )
                            schedule.add_event(sliced_home, agent)

                    schedule.add_events(events, agent)
                    tm.add_inprogress(node, events[1].end)
                    tm.frontier.remove(node)
                    tm.frontier.update(dg._graph.successors(node))
                    context.positions[agent] = events[2].data[-1]
                    context.set_agent_start_time(agent, events[2].end)

                    all_collide_flag = False
                    break

                if all_collide_flag:
                    context.set_agent_start_time(agent, time + options.collision_offset)

        return schedule

    def _slice_home_event(self, home_event: MoveEvent, end_time: float):
        new_traj = home_event.traj.slice(home_event.start, end_time)
        path = [p.data for p in new_traj.points]
        return MoveEvent(home_event.start, path, home_event.velocity)
