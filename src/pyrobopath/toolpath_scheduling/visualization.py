import numpy as np
import matplotlib.pyplot as plt
from ..scheduling import MultiAgentSchedule


def draw_multi_agent_schedule(s: MultiAgentSchedule, show=True):
    fig, ax = plt.subplots(figsize=(9, 4))
    category_colors = plt.get_cmap('Pastel1')(
        np.linspace(0.15, 0.85, s.n_agents()))

    for (agent, schedule), color in zip(s.schedules.items(), category_colors):
        labels = []
        for event in schedule._events:
            p = ax.barh(
                agent,
                left=event.start,
                width=event.duration,
                height=0.5,
                edgecolor="black",
                color=color
            )
            label = str(event.data)
            ax.bar_label(p, labels=[label], label_type="center")

    if show:
        plt.show()
    return fig, ax