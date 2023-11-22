from typing import Hashable, Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.widgets import Slider

from ..collision_detection import FCLRobotBBCollisionModel
from .system_model import AgentModel
from .schedule import ContourEvent, MultiAgentToolpathSchedule, ToolpathSchedule


def draw_multi_agent_schedule(s: MultiAgentToolpathSchedule, show=True):
    fig, ax = plt.subplots(figsize=(9, 4))

    # get unique materials
    unique_tools = set()
    for sched in s.schedules.values():
        unique_tools.update(
            [e.contour.tool for e in sched._events if isinstance(e, ContourEvent)]
        )
    unique_tools.discard(-1)
    category_colors = plt.get_cmap("Paired")(np.linspace(0.1, 0.9, len(unique_tools)))

    for agent, schedule in s.schedules.items():
        for event in schedule._events:
            color = "grey"
            if isinstance(event, ContourEvent):
                color = category_colors[event.contour.tool]

            p = ax.barh(
                agent,
                left=event.start,
                width=event.duration,
                height=0.5,
                edgecolor="black",
                color=color,
            )
            label = str(event.data)
            ax.bar_label(p, labels=[label], label_type="center")

    ax.set_xlabel("Time")
    ax.set_title("Multi-agent Schedule")
    if show:
        plt.show()
    return fig, ax


def animate_multi_agent_toolpath_schedule(
    schedule: MultiAgentToolpathSchedule,
    agent_models: Dict[Hashable, AgentModel],
    step,
    show=True,
):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    ax.set_xlim((-6, 6))
    ax.set_ylim((-3, 3))
    ax.autoscale_view(False)

    anim_models = []
    for a in schedule.agents():
        model = None
        if isinstance(agent_models[a].collision_model, FCLRobotBBCollisionModel):
            model = RobotBBAnimationModel(agent_models[a], schedule[a], ax)
        else:
            model = AnimationModel(agent_models[a], schedule[a], ax)
        anim_models.append(model)

    def update(val):
        for model in anim_models:
            model.update(val)
        fig.canvas.draw_idle()

    # add slider control
    axtime = plt.axes([0.25, 0.1, 0.65, 0.03])
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
        r = 0.1
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

        T_w_e = np.identity(3)
        T_w_e[:2, :2] = self.model.collision_model.rotation[:2, :2]
        T_w_e[:2, 2] = self.model.collision_model.translation[:2]

        T_e_bl = np.identity(3)
        T_e_bl[:2, 2] = np.array([-self.dim[0], -self.dim[1] / 2])

        T_w_bl = T_w_e @ T_e_bl
        tf = transforms.Affine2D(T_w_bl)
        self.rect.set_transform(tf + self.ax.transData)

        v_w_line = T_w_e @ np.array([-self.dim[0], 0, 1])
        self.line.set_data(
            [self.model.base_frame_position[0], v_w_line[0]],
            [self.model.base_frame_position[1], v_w_line[1]],
        )
