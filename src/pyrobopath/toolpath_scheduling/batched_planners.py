from typing import Dict
from concurrent.futures import ProcessPoolExecutor

from pyrobopath.toolpath import Toolpath
from pyrobopath.process import AgentModel, DependencyGraph, batch_digraph

from .schedule import MultiAgentToolpathSchedule
from .toolpath_scheduler import PlanningOptions, MultiAgentToolpathPlanner


class BatchedSequentialPlanner:
    def __init__(self, agent_models: Dict[str, AgentModel], batch_size: int):
        self._base_planner = MultiAgentToolpathPlanner(agent_models)
        self._batch_size = batch_size

    def plan(
        self, toolpath: Toolpath, dg: DependencyGraph, options: PlanningOptions
    ) -> MultiAgentToolpathSchedule:
        dgs = batch_digraph(dg, self._batch_size)

        results = []
        for subgraph in dgs:
            results.append(self._base_planner.plan(toolpath, subgraph, options))

        return MultiAgentToolpathSchedule.merge(results)


class BatchedParallelPlanner:
    def __init__(self, agent_models: Dict[str, AgentModel], batch_size: int):
        self._base_planner = MultiAgentToolpathPlanner(agent_models)
        self._batch_size = batch_size

    def _plan_batch(self, args):
        return self._base_planner.plan(*args)

    def plan(
        self, toolpath: Toolpath, dg: DependencyGraph, options: PlanningOptions
    ) -> MultiAgentToolpathSchedule:
        dgs = batch_digraph(dg, self._batch_size)
        args = [(toolpath, subgraph, options) for subgraph in dgs]

        results = []
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self._plan_batch, args))

        return MultiAgentToolpathSchedule.merge(results)
