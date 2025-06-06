from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.widgets import Slider

from pyrobopath.toolpath.toolpath_core import Toolpath


def visualize_toolpath(toolpath: Toolpath, show=True):
    """
    Visualize a 3D toolpath using matplotlib.

    Each contour in the toolpath is plotted in 3D space with a distinct color.
    This is useful for inspecting the overall geometry and sequencing of paths
    in a toolpath object.

    Args:
        toolpath (Toolpath): The toolpath object containing contours to visualize.
        show (bool, optional): Whether to display the figure immediately. Defaults to True.

    Returns:
        tuple: A tuple containing the matplotlib figure and 3D axes objects.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    category_colors = plt.get_cmap("plasma")(
        np.linspace(0.0, 1.0, len(toolpath.contours))
    )

    for contour, color in zip(toolpath.contours, category_colors):
        path = np.array(contour.path)
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=color,
            path_effects=[pe.Stroke(linewidth=3, foreground="black"), pe.Normal()],
        )
    ax.set_aspect("equal")
    if show:
        plt.show()
    return fig, ax


def visualize_toolpath_projection(toolpath: Toolpath, show=True):
    """
    Visualize a 2D projection of the toolpath with an interactive layer slider.

    Projects each contour in the toolpath onto the XY plane and allows the user
    to browse different Z-height layers using a vertical slider. Each tool is
    assigned a distinct color for visual differentiation.

    Args:
        toolpath (Toolpath): The toolpath object containing layered contours.
        show (bool, optional): Whether to display the figure immediately. Defaults to True.

    Returns:
        tuple: A tuple containing the matplotlib figure and 2D axes objects.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    layer_slider = _plot_toolpath_projection(toolpath, fig, ax)
    ax.set_aspect("equal")

    if show:
        plt.show()
    return fig, ax


def _plot_toolpath_projection(toolpath, fig, ax):
    unique_tools = toolpath.tools()
    color_map = plt.get_cmap("Paired")(np.linspace(0.1, 0.9, len(unique_tools)))
    tool_colors = {tool: color_map[i] for i, tool in enumerate(unique_tools)}

    # find a unique set of z values
    contour_z = []
    tools = []
    for contour in toolpath.contours:
        z_values = np.sort(np.array(contour.path)[:, 2])
        contour_z.append(z_values[0])
        tools.append(contour.tool)

    unique_z = sorted(set(contour_z))

    def update_layer(val):
        ax.cla()
        z_height = unique_z[val - 1]
        indices = [i for i, x in enumerate(contour_z) if x == z_height]
        for idx in indices:
            path = np.array(toolpath.contours[idx].path)
            ax.plot(
                path[:, 0],
                path[:, 1],
                path_effects=[pe.Stroke(linewidth=3, foreground="black"), pe.Normal()],
                color=tool_colors[tools[idx]],
            )

    # add slider control
    axlayers = fig.add_axes([0.05, 0.25, 0.0225, 0.63])
    layer_slider = Slider(
        ax=axlayers,
        label="Layer",
        valmin=1,
        valmax=len(unique_z),
        valstep=1,
        orientation="vertical",
    )
    layer_slider.on_changed(update_layer)
    update_layer(1)
    return layer_slider
