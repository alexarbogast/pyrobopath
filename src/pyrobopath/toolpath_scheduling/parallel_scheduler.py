from copy import deepcopy
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from pyrobopath.toolpath import Toolpath
from pyrobopath.process import AgentModel, DependencyGraph, split_digraph

from .schedule import MultiAgentToolpathSchedule
from .toolpath_scheduler import PlanningOptions, MultiAgentToolpathPlanner


class DepthLimitedParallelPlanner:
    def __init__(self, agent_models: Dict[str, AgentModel], planning_depth: int):
        self._base_planner = MultiAgentToolpathPlanner(agent_models)
        self._planning_depth = planning_depth

    def plan(
        self, toolpath: Toolpath, dg: DependencyGraph, options: PlanningOptions
    ) -> MultiAgentToolpathSchedule:
        # split dependency graph by max depth
        dgs = split_digraph(dg, self._planning_depth)
        n_threads = len(dgs)

        results = []
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = {
                executor.submit(
                    deepcopy(self._base_planner).plan, toolpath, dgs[i], options
                ): i
                for i in range(n_threads)
            }

            for future in as_completed(futures):
                i = futures[future]
                try:
                    result = future.result()
                    results.append((i, result))
                except Exception as e:
                    results.append((i, f"Execption: {e}"))

        schedules = [res for _, res in sorted(results, key=lambda x: x[0])]
        return MultiAgentToolpathSchedule.merge(schedules)
