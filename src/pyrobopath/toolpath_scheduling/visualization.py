from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import matplotlib.patheffects as pe
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec

from pyrobopath.toolpath import Toolpath
from pyrobopath.collision_detection import (
    FCLRobotBBCollisionModel,
)

from pyrobopath.toolpath_scheduling.system_model import AgentModel
from pyrobopath.toolpath_scheduling.schedule import (
    ContourEvent,
    MultiAgentToolpathSchedule,
    ToolpathSchedule,
)

# temporary fix to override user-specific rcParams that distort scheduling
# animations
import matplotlib as mpl

mpl.rcParams = mpl.rcParamsDefault


def draw_multi_agent_schedule(s: MultiAgentToolpathSchedule, show=True):
    fig, ax = plt.subplots(figsize=(9, 4))

    _plot_multi_agent_schedule(s, ax)

    ax.set_xlabel("Time")
    ax.set_title("Multi-agent Schedule")
    if show:
        plt.show()
    return fig, ax


def _plot_multi_agent_schedule(s: MultiAgentToolpathSchedule, ax):
    # get unique materials
    unique_tools = set()
    for sched in s.schedules.values():
        unique_tools.update(
            [e.contour.tool for e in sched._events if isinstance(e, ContourEvent)]
        )
    unique_tools = list(unique_tools)
    color_map = plt.get_cmap("Paired")(np.linspace(0.1, 0.9, len(unique_tools)))

    # build material dictionary
    tool_colors = {tool: color_map[i] for i, tool in enumerate(unique_tools)}

    for agent, schedule in s.schedules.items():
        for event in schedule._events:
            color = "grey"
            if isinstance(event, ContourEvent):
                color = tool_colors[event.contour.tool]

            p = ax.barh(
                agent,
                left=event.start,
                width=event.duration,
                height=0.5,
                edgecolor="black",
                color=color,
            )
            # label = str(event.data)
            # ax.bar_label(p, labels=[label], label_type="center")


def animate_multi_agent_toolpath_schedule(
    schedule: MultiAgentToolpathSchedule,
    agent_models: Dict[str, AgentModel],
    step,
    plot_toolpath=True,
    show=True,
):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    ax.set_xlim((-6, 6))
    ax.set_ylim((-3, 3))
    ax.autoscale_view(False)

    # create models to animate
    anim_models = []
    for a in schedule.agents():
        model = None
        if isinstance(agent_models[a].collision_model, FCLRobotBBCollisionModel):
            model = RobotBBAnimationModel(agent_models[a], schedule[a], ax)
        else:
            model = AnimationModel(agent_models[a], schedule[a], ax)
        anim_models.append(model)

    # update all models on slider change
    def update(val):
        for model in anim_models:
            model.update(val)
        fig.canvas.draw_idle()

    # add slider control
    axtime = plt.axes((0.25, 0.1, 0.65, 0.03))
    anim_slider = Slider(
        ax=axtime,
        label="time",
        valmin=schedule.start_time(),
        valmax=schedule.end_time(),
        valstep=step,
        valinit=schedule.start_time(),
    )
    anim_slider.on_changed(update)
    update(schedule.start_time())
    ax.set_aspect("equal")

    if show:
        plt.show()
    return fig, ax


def animate_multi_agent_toolpath_full(
    toolpath: Toolpath,
    schedule: MultiAgentToolpathSchedule,
    agent_models: Dict[str, AgentModel],
    step=0.01,
    limits=((-500, 500), (-500, 500)),
    show=True,
):
    fig = plt.figure(figsize=(13, 9))
    gs = GridSpec(3, 2, height_ratios=[1, 3, 0.15], width_ratios=[1, 50])
    sched_ax = plt.subplot(gs[0, :])
    anim_ax = plt.subplot(gs[1, 1])

    # ================= schedule =================
    _plot_multi_agent_schedule(schedule, sched_ax)
    (sched_line,) = sched_ax.plot(
        [],
        [],
        lw=2,
        color=(0, 1, 0.31),
        path_effects=[
            pe.Stroke(linewidth=4, foreground="black"),
            pe.Normal(),
        ],
    )

    sched_ax.set_xlabel("Time")
    sched_ax.set_title("Multi-agent Schedule")

    # ================= toolpath =================
    unique_tools = toolpath.tools()
    color_map = plt.get_cmap("Paired")(np.linspace(0.1, 0.9, len(unique_tools)))
    tool_colors = {tool: color_map[i] for i, tool in enumerate(unique_tools)}
    contour_z = []
    tools = []
    for contour in toolpath.contours:
        z_values = np.sort(np.array(contour.path)[:, 2])
        contour_z.append(z_values[0])
        tools.append(contour.tool)
    unique_z = sorted(set(contour_z))

    contour_lines = []

    def update_layer(val):
        for line in contour_lines:
            mpl_line = line.pop(0)
            mpl_line.remove()
        contour_lines.clear()
        z_height = unique_z[val - 1]
        indices = [i for i, x in enumerate(contour_z) if x == z_height]
        for idx in indices:
            path = np.array(toolpath.contours[idx].path)
            contour_lines.append(
                anim_ax.plot(
                    path[:, 0],
                    path[:, 1],
                    path_effects=[
                        pe.Stroke(linewidth=3, foreground="black"),
                        pe.Normal(),
                    ],
                    color=tool_colors[tools[idx]],
                    zorder=0,
                )
            )

    # add slider control
    axlayers = plt.subplot(gs[1, 0])
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

    # ================= agent simulation =================
    anim_ax.set_xlim(limits[0])
    anim_ax.set_ylim(limits[1])
    anim_ax.autoscale_view(False)

    # create models to animate
    anim_models = []
    for a in schedule.agents():
        model = None
        if isinstance(agent_models[a].collision_model, FCLRobotBBCollisionModel):
            model = RobotBBAnimationModel(agent_models[a], schedule[a], anim_ax)  # type: ignore
        else:
            model = AnimationModel(agent_models[a], schedule[a], anim_ax)  # type: ignore
        anim_models.append(model)

    # update all models and schedule line on slider change
    def update_anim(val):
        for model in anim_models:
            model.update(val)

        sched_line.set_data([val, val], [-3, 3])
        fig.canvas.draw_idle()

    # add slider control
    axtime = plt.subplot(gs[2, 1:])
    anim_slider = Slider(
        ax=axtime,
        label="time",
        valmin=schedule.start_time(),
        valmax=schedule.end_time(),
        valstep=step,
        valinit=schedule.start_time(),
    )
    anim_slider.on_changed(update_anim)
    update_anim(schedule.start_time())
    anim_ax.set_aspect("equal")
    plt.tight_layout()

    if show:
        plt.show()
    return fig

    # Animation
    # import matplotlib.animation as animation
    # fps=30
    # writer = animation.FFMpegWriter(fps=fps)
    # anim = animation.FuncAnimation(fig, update_anim, frames=np.arange(schedule.start_time(), schedule.end_time(), 0.6))
    # anim.save('test2.mp4', writer=writer)


class AnimationModel(object):
    def __init__(self, agent_model: AgentModel, schedule: ToolpathSchedule, ax):
        self.model = agent_model
        self.sched = schedule
        self.ax = ax

        (self.line,) = ax.plot([], [], lw=2)

    def update(self, t):
        pos = self.sched.get_state(t, default=self.model.home_position)
        self.line.set_data(
            [self.model.base_frame_position[0], pos[0]],
            [self.model.base_frame_position[1], pos[1]],
        )


class RobotBBAnimationModel(AnimationModel):
    def __init__(self, agent_model: AgentModel, schedule: ToolpathSchedule, ax):
        super(RobotBBAnimationModel, self).__init__(agent_model, schedule, ax)

        # create bounding box
        self.dim = agent_model.collision_model.box.side
        self.rect = patches.Rectangle(
            (0, 0), self.dim[0], self.dim[1], fill=None, linewidth=2
        )
        ax.add_patch(self.rect)

        # modify attach line
        (self.line,) = ax.plot([], [], lw=2, linestyle="--", color="black")

        # create base
        r = 0.3 * self.dim[1]
        bf = self.model.base_frame_position[:2]
        hatch = r * np.cos(np.pi / 4)
        base = patches.Circle(
            self.model.base_frame_position[:2], r, fill=None, linewidth=2
        )
        ax.add_patch(base)
        ax.plot(
            (bf[0] - hatch, bf[0] + hatch),
            (bf[1] - hatch, bf[1] + hatch),
            linewidth=1,
            color="k",
        )
        ax.plot(
            (bf[0] - hatch, bf[0] + hatch),
            (bf[1] + hatch, bf[1] - hatch),
            linewidth=1,
            color="k",
        )

    def update(self, t):
        pos = self.sched.get_state(t, default=self.model.home_position)
        self.model.collision_model.translation = pos

        # end-effector in world frame
        T_w_e = np.identity(3)
        T_w_e[:2, :2] = self.model.collision_model.rotation[:2, :2]
        T_w_e[:2, 2] = self.model.collision_model.translation[:2]

        # bottom-left corner in end-effector frame
        T_e_bl = np.identity(3)
        T_e_bl[:2, 2] = self.model.collision_model.offset[:2] + np.array(
            [-self.dim[0], -self.dim[1] / 2]
        )

        T_w_bl = T_w_e @ T_e_bl
        tf = transforms.Affine2D(T_w_bl)
        self.rect.set_transform(tf + self.ax.transData)

        self.line.set_data(
            [self.model.base_frame_position[0], pos[0]],
            [self.model.base_frame_position[1], pos[1]],
        )
