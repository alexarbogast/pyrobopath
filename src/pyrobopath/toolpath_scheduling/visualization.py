import numpy as np
from ..scheduling import MultiAgentSchedule


def draw_multi_agent_schedule(s: MultiAgentSchedule, show=True):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))

    # get unique materials
    unique_tools = set()
    for sched in s.schedules.values():
        unique_tools.update([e.data.tool for e in sched._events])
    unique_tools.discard(-1)
    category_colors = plt.get_cmap("Paired")(np.linspace(0.1, 0.9, len(unique_tools)))

    for agent, schedule in s.schedules.items():
        for event in schedule._events:
            color = category_colors[event.data.tool]
            if event.data.tool == -1:
                color = "grey"

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
