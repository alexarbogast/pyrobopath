from __future__ import annotations
from .toolpath import Contour, Toolpath

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patheffects as pe
import numpy as np



def visualize_toolpath(toolpath: Toolpath, show=True):
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
    if show:
        plt.show()
    return fig, ax


def visualize_toolpath_projection(toolpath: Toolpath, show=True):
    COLORS = plt.get_cmap("Paired")(np.linspace(0.1, 0.9, len(toolpath.tools())))

    # find a unique set of z values
    contour_z = []
    tools = []
    for contour in toolpath.contours:
        z_values = np.sort(np.array(contour.path)[:, 2])
        contour_z.append(z_values[0])
        tools.append(contour.tool)

    fig, ax = plt.subplots(figsize=(10, 8))
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
                color=COLORS[tools[idx]],
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
    ax.set_aspect("equal")

    if show:
        plt.show()
    return fig, ax
