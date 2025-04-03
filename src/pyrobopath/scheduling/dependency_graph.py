import networkx as nx
from itertools import product

class DependencyGraph(object):
    """Directed Acyclic Graph"""

    def __init__(self):
        self._graph = nx.DiGraph()

    def add_node(self, node, dependencies=None):
        self._graph.add_node(node, complete=False)
        if dependencies is None:
            return

        for d in dependencies:
            if not d in self._graph.nodes:
                self._graph.add_node(d, complete=False)
        edges = product(dependencies, [node])
        self._graph.add_edges_from(edges)

    def set_complete(self, node):
        nx.set_node_attributes(self._graph, {node: {"complete": True}})

    def can_start(self, node):
        parents = self._graph.predecessors(node)
        complete = all([self._graph.nodes[p]["complete"] for p in parents])
        return complete

    def reset(self):
        nx.set_node_attributes(self._graph, False, "complete")

    def draw(self, show=True):
        import matplotlib.pyplot as plt
        from networkx.drawing.nx_pydot import graphviz_layout

        pos = graphviz_layout(self._graph, prog="dot")
        nx.draw(self._graph, pos, with_labels=True, )

        if show:
            plt.show()
