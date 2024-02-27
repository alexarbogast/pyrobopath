import numpy as np

from pyrobopath.toolpath import Toolpath
from pyrobopath.scheduling import DependencyGraph


def create_dependency_graph_by_layers(toolpath: Toolpath) -> DependencyGraph:
    """Create a dependency graph between contours based on the z height of the
    contour's first point. Contours have dependencies with neighboring
    contours at smaller z-heights.
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
