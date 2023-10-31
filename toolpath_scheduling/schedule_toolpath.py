from ..toolpath import Toolpath
from ..scheduling import Event, MultiAgentSchedule
from ..scheduling import DependencyGraph


velocity = 50.0


class ToolpathScheduler(object):
    def __init__(self, capabilities: dict):
        self._capabilities = capabilities
        self._agents = capabilities.keys()

    def schedule(self, toolpath: Toolpath, dg: DependencyGraph):
        """Create a longest-processing-time-first (LPT) schedule"""
        contours = toolpath.contours

        completed_tasks = set()  # all tasks that have finished
        in_progress = {"start": 0.0} # the set of tasks in progress and their expiration date {task: finish_time}
        frontier = set(dg._graph.successors("start"))
        dg.set_complete("start")

        # the minimum starting time for a task on a given agent
        agent_times = dict().fromkeys(self._capabilities.keys(), 0.0)

        agent_schedules = MultiAgentSchedule()
        for agent in self._agents:
            agent_schedules.add_agent(agent)

        while frontier:
            unique_times = sorted(set(agent_times.values()))
            time = unique_times[0]
            min_time_agents = filter(lambda n: n[1] == time, agent_times.items())

            # change in_progress to complete
            complete = [k for (k, v) in in_progress.items() if time >= v]
            completed_tasks = completed_tasks.union(complete)
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

                # sort tasks by out degree
                node = sorted(available, key=lambda a: dg._graph.out_degree(a))[0]
                event = Event(
                    contours[node],
                    start=time,
                    duration=contours[node].path_length() / velocity,
                )
                agent_schedules.add_event(event, agent)
                in_progress[node] = agent_schedules[agent].end_time()
                agent_times[agent] = agent_schedules[agent].end_time()
                frontier = frontier.union(dg._graph.successors(node))
                frontier.remove(node)

        return agent_schedules
