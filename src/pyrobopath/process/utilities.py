import numpy as np

from pyrobopath.toolpath import Toolpath
from .dependency_graph import DependencyGraph


def create_dependency_graph_by_layers(toolpath: Toolpath) -> DependencyGraph:
    """
    Create a layered dependency graph from a toolpath based on contour Z-heights.

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
    dg.add_node("start")
    dg.set_complete("start")

    unique_z = sorted(set(contour_z))
    # connect start node
    ind = [i for i, x in enumerate(contour_z) if x == unique_z[0]]
    for first_layer_node in ind:
        dg.add_node(first_layer_node, ["start"])

    for a, b in zip(unique_z[:-1], unique_z[1:]):
        ind_a = [i for i, x in enumerate(contour_z) if x == a]
        ind_b = [i for i, x in enumerate(contour_z) if x == b]
        for upper in ind_b:
            dg.add_node(upper, ind_a)

    return dg
