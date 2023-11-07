import numpy as np

from ..toolpath import Toolpath, Contour
from ..scheduling import Event, MultiAgentSchedule
from ..scheduling import DependencyGraph

from dataclasses import dataclass


@dataclass
class PlanningOptions:
    velocity: float = 50.0
    retract_height: float = 50.0
    collision_offset: float = 5.0


class ToolpathScheduler(object):
    def __init__(self, capabilities: dict):
        self._capabilities = capabilities
        self._agents = capabilities.keys()

    def schedule(
        self, toolpath: Toolpath, dg: DependencyGraph, options: PlanningOptions
    ):
        """Create a longest-processing-time-first (LPT) schedule with travel moves"""
        contours = toolpath.contours

        completed_tasks = set()
        in_progress = {"start": 0.0}
        frontier = set(dg._graph.successors("start"))
        dg.set_complete("start")

        current_positions = dict().fromkeys(self._agents, np.array([0.0, 0.0, 0.0]))
        agent_times = dict().fromkeys(self._agents, 0.0)
        agent_schedules = MultiAgentSchedule()
        for agent in self._agents:
            agent_schedules.add_agent(agent)

        while frontier:
            unique_times = sorted(set(agent_times.values()))
            time = unique_times[0]
            min_time_agents = filter(lambda n: n[1] == time, agent_times.items())

            # change in_progress to complete
            complete = [k for (k, v) in in_progress.items() if time >= v]
            completed_tasks.update(complete)
            for c in complete:
                dg.set_complete(c)
                in_progress.pop(c)

            # find assignments for the min time agents
            for agent, _ in min_time_agents:
                tools = self._capabilities[agent]

                available = filter(lambda n: dg.can_start(n), frontier)
                available = filter(lambda n: contours[n].tool in tools, available)
                available = list(available)

                if not available:
                    if len(unique_times) > 1:
                        agent_times[agent] = unique_times[1]
                    else:
                        agent_times[agent] = unique_times[0]
                    continue

                # sort tasks by out-degree
                node = sorted(available, key=lambda a: dg._graph.out_degree(a))[0]

                travel_contour = self.create_travel_contour(
                    current_positions[agent],
                    contours[node].path[0],
                    options.retract_height,
                )
                travel_duration = travel_contour.path_length() / options.velocity
                travel_event = Event(
                    travel_contour, start=time, duration=travel_duration
                )
                event = Event(
                    contours[node],
                    start=time + travel_duration,
                    duration=contours[node].path_length() / options.velocity,
                )
                agent_schedules.add_event(travel_event, agent)
                agent_schedules.add_event(event, agent)

                in_progress[node] = agent_schedules[agent].end_time()
                agent_times[agent] = agent_schedules[agent].end_time()
                frontier.update(dg._graph.successors(node))
                frontier.remove(node)
                current_positions[agent] = contours[node].path[-1]

        return agent_schedules

    def create_travel_contour(self, start, end, retract):
        """Create a linear travel move between a start point and end point"""
        above_s = start + np.array([0, 0, retract])
        above_e = end + np.array([0, 0, retract])
        travel = [start, above_s, above_e, end]
        contour = Contour(travel, tool=-1)
        return contour


class ContourEvent(Event):
    def __init__(self, contour: Contour, start: float, velocity: float):
        duration = contour.path_length() / velocity
        super().__init__(contour, start, duration)
        self.velocity = velocity
