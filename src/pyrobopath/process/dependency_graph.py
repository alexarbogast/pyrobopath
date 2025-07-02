import networkx as nx
from itertools import product


class DependencyGraph(object):
    """Directed Acyclic Graph"""

    def __init__(self):
        self._graph = nx.DiGraph()
        self._completed_tasks = set()

    def add_node(self, node, dependencies=None):
        self._graph.add_node(node)
        if dependencies is None:
            return

        for d in dependencies:
            if not d in self._graph.nodes:
                self._graph.add_node(d)
        edges = product(dependencies, [node])
        self._graph.add_edges_from(edges)

    def mark_complete(self, node):
        self._completed_tasks.add(node)

    def can_start(self, node):
        parents = self._graph.predecessors(node)
        return all(p in self._completed_tasks for p in parents)

    def pending_tasks(self):
        return [n for n in self._graph.nodes if n not in self._completed_tasks]

    def roots(self):
        return [n for n in self._graph.nodes if self._graph.in_degree(n) == 0]

    def reset(self):
        self._completed_tasks.clear()

    def draw(self, show=True):
        import matplotlib.pyplot as plt
        from networkx.drawing.nx_pydot import graphviz_layout

        _, ax = plt.subplots()
        pos = graphviz_layout(self._graph, prog="dot")
        nx.draw(self._graph, pos, ax, with_labels=True)

        if show:
            plt.show()
