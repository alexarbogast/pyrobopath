import numpy as np
import networkx as nx
from typing import List

from pyrobopath.toolpath import Toolpath
from .dependency_graph import DependencyGraph


def create_dependency_graph_by_z(toolpath: Toolpath) -> DependencyGraph:
    """
    Create a layered dependency graph from a toolpath based on contour
    Z-heights.

    Constructs a `DependencyGraph` where each contour is a node, and
    dependencies are defined by the relative Z-height of the contours.
    Specifically, a contour is dependent on all contours in the layer directly
    below it.

    Parameters
    ----------
    toolpath : Toolpath
        A toolpath composed of multiple contours to be layered and ordered.

    Returns
    -------
    DependencyGraph
        A dependency graph with inter-layer dependencies based on Z-height.
    """
    contour_z = []
    for contour in toolpath.contours:
        z_values = np.array(contour.path)[:, 2]
        z_values = set(z_values)
        contour_z.append(z_values.pop())

    dg = DependencyGraph()
    unique_z = sorted(set(contour_z))
    # connect start node
    ind = [i for i, x in enumerate(contour_z) if x == unique_z[0]]
    for first_layer_node in ind:
        dg.add_node(first_layer_node)

    for a, b in zip(unique_z[:-1], unique_z[1:]):
        ind_a = [i for i, x in enumerate(contour_z) if x == a]
        ind_b = [i for i, x in enumerate(contour_z) if x == b]
        for upper in ind_b:
            dg.add_node(upper, ind_a)

    return dg


def batch_digraph(dg: DependencyGraph, max_nodes: int) -> List[DependencyGraph]:
    """
    Split a directed graph into multiple subgraphs with at most `max_nodes`
    nodes each.

    Parameters
    ----------
    dg : DependencyGraph
        The input graph to be split into subgraphs.
    max_nodes : int
        The maximum number of nodes allowed in each subgraph.

    Returns
    -------
    subgraphs : List[DependencyGraph]
        A list of `DependencyGraph` instances, each containing at most
        `max_nodes` nodes. The subgraphs preserve the internal structure of the
        original graph for the included node subsets.

    Notes
    -----
    - This function does not attempt to preserve connectivity between
      subgraphs.
    - Nodes are partitioned in the order returned by `dg._graph.nodes`.
    - The `_graph` attribute is deep-copied into each subgraph to ensure
      isolation.

    Examples
    --------
    >>> dg = DependencyGraph()
    >>> # Assume dg._graph has 10 nodes
    >>> subgraphs = split_digraph(dg, max_nodes=4)
    >>> len(subgraphs)
    3
    >>> [len(sg._graph.nodes) for sg in subgraphs]
    [4, 4, 2]
    """
    subgraphs = []
    nodes = list(dg._graph.nodes)

    # Simple greedy partition
    for i in range(0, len(nodes), max_nodes):
        node_subset = nodes[i : i + max_nodes]
        subgraph = DependencyGraph()
        subgraph._graph = dg._graph.subgraph(node_subset).copy()  # type:ignore
        subgraphs.append(subgraph)
    return subgraphs


def stratify_digraph(dg: DependencyGraph) -> List[DependencyGraph]:
    subgraphs = []
    generations = nx.topological_generations(dg._graph)
    for gen in generations:
        subgraph = DependencyGraph()
        subgraph._graph.add_nodes_from(gen)
        subgraphs.append(subgraph)
    return subgraphs
