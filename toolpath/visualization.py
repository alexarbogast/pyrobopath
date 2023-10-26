from __future__ import annotations
from .toolpath import Contour, Toolpath

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


def visualize_toolpath(toolpath: Toolpath):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for contour in toolpath.contours:
        path = np.array(contour.path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2])
    plt.show()


def visualize_toolpath_projection(toolpath: Toolpath):
    COLORS = ["#DA3E52", "#001524", "#6969B3"]
    # find a unique set of z values
    contour_z = []
    tools = []
    for contour in toolpath.contours:
        z_values = np.array(contour.path)[:, 2]
        z_values = set(z_values)
        contour_z.append(z_values.pop())
        tools.append(contour.tool)

    fig, ax = plt.subplots()
    unique_z = list(set(contour_z))

    def update_layer(val):
        ax.cla()
        z_height = unique_z[val - 1]
        indices = [i for i, x in enumerate(contour_z) if x == z_height]
        for idx in indices:
            path = np.array(toolpath.contours[idx].path)
            ax.plot(path[:, 0], path[:, 1], COLORS[tools[idx]])

    # add slider control
    axlayers = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
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
    plt.show()
