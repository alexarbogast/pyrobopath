from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.widgets import Slider

from pyrobopath.toolpath import Toolpath
from .colors import get_contour_colors


def visualize_toolpath(
    toolpath: Toolpath, color_method="tool", color_seq="tab10", show=True
):
    """
    Visualize a 3D toolpath using matplotlib.

    This function displays a 3D plot of the provided toolpath. Each contour is
    rendered in space with a color assigned based on a specified color method.
    Useful for examining path layout, tool usage, or sequencing in a 3D context.

    Parameters
    ----------
    toolpath : Toolpath
        The toolpath object containing contours to be visualized.
    color_method : str, optional
        The strategy used to assign colors to contours. Valid options include
        'tool', 'z', or 'cycle'. Defaults to 'tool'.
    color_seq : str or list, optional
        The name of the matplotlib colormap or a list of color values to cycle
        through. Defaults to 'tab10'.
    show : bool, optional
        Whether to immediately display the plot with `plt.show()`.
        Defaults to True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plot.
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axes on which the toolpath is drawn.

    See Also
    --------
    pyrobopath.toolpath.visualization.colors.get_contour_colors
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    colors = get_contour_colors(toolpath.contours, color_method, color_seq)

    for contour, color in zip(toolpath.contours, colors):
        path = np.array(contour.path)
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=color,
            path_effects=[pe.Stroke(linewidth=3, foreground="black"), pe.Normal()],
        )
    ax.set_aspect("equal")
    fig.tight_layout()
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
