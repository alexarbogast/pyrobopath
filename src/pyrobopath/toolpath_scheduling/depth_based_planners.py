from typing import Dict
from concurrent.futures import ProcessPoolExecutor

from pyrobopath.toolpath import Toolpath
from pyrobopath.process import AgentModel, DependencyGraph, stratify_digraph

from .schedule import MultiAgentToolpathSchedule
from .toolpath_scheduler import PlanningOptions, MultiAgentToolpathPlanner


class DepthBasedSequentialPlanner:
    def __init__(self, agent_models: Dict[str, AgentModel]):
        self._base_planner = MultiAgentToolpathPlanner(agent_models)

    def plan(
        self, toolpath: Toolpath, dg: DependencyGraph, options: PlanningOptions
    ) -> MultiAgentToolpathSchedule:
        dgs = stratify_digraph(dg)

        results = []
        for subgraph in dgs:
            results.append(self._base_planner.plan(toolpath, subgraph, options))

        return MultiAgentToolpathSchedule.merge(results)


class DepthBasedParallelPlanner:
    def __init__(self, agent_models: Dict[str, AgentModel]):
        self._base_planner = MultiAgentToolpathPlanner(agent_models)

    def _plan_layer(self, args):
        return self._base_planner.plan(*args)

    def plan(
        self, toolpath: Toolpath, dg: DependencyGraph, options: PlanningOptions
    ) -> MultiAgentToolpathSchedule:
        dgs = stratify_digraph(dg)
        args = [(toolpath, subgraph, options) for subgraph in dgs]

        results = []
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self._plan_layer, args))

        return MultiAgentToolpathSchedule.merge(results)
