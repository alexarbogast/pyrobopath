import numpy as np

from .schedule import Event, Schedule, MultiAgentSchedule


def draw_schedule(s: Schedule, show=True):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(s.start_time(), s.end_time())
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("time")

    colors = plt.get_cmap("Pastel2")(np.linspace(0.15, 0.85, s.n_events()))

    for event, color in zip(s._events, colors):
        p = ax.barh(
            "agent",
            left=event.start,
            width=event.duration,
            height=0.5,
            edgecolor="black",
            color=color,
        )
        ax.bar_label(p, label_type="center")

    if show:
        plt.show()
    return fig, ax


def draw_multi_agent_schedule(s: MultiAgentSchedule, show=True):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    category_colors = plt.get_cmap("Pastel1")(np.linspace(0.15, 0.85, s.n_agents()))

    for (agent, schedule), color in zip(s.schedules.items(), category_colors):
        for event in schedule._events:
            p = ax.barh(
                agent,
                left=event.start,
                width=event.duration,
                height=0.5,
                edgecolor="black",
                color=color,
            )
            ax.bar_label(p, label_type="center")

    if show:
        plt.show()
    return fig, ax


if __name__ == "__main__":
    schedule = Schedule()
    schedule.add_event(Event("eventA", 0.0, 5.0))
    schedule.add_event(Event("eventB", 5.0, 2.0))
    schedule.add_event(Event("eventC", 7.0, 5.0))
    schedule.add_event(Event("eventD", 12.0, 10.0))
    schedule.add_event(Event("eventE", 22.0, 45.0))
    schedule.add_event(Event("eventF", 67.0, 15.0))

    draw_schedule(schedule)

    schedule = MultiAgentSchedule()
    schedule.add_event(Event("eventA1", -1.0, 5.0), "agent1")
    schedule.add_event(Event("eventB1", 5.0, 2.0), "agent1")
    schedule.add_event(Event("eventC1", 7.0, 5.0), "agent1")
    schedule.add_event(Event("eventD1", 12.0, 10.0), "agent1")
    schedule.add_event(Event("eventE1", 22.0, 45.0), "agent1")
    schedule.add_event(Event("eventF1", 67.0, 15.0), "agent1")

    schedule.add_event(Event("eventA2", 0.0, 5.0), "agent2")
    schedule.add_event(Event("eventB2", 5.0, 4.0), "agent2")
    schedule.add_event(Event("eventC2", 9.0, 10.0), "agent2")
    schedule.add_event(Event("eventD2", 19.0, 10.0), "agent2")
    schedule.add_event(Event("eventE2", 67.0, 16.0), "agent2")

    other = Schedule()
    other.add_event(Event("eventA3", -2.0, 5.0))
    other.add_event(Event("eventB3", 70.0, 20.0))
    schedule.add_schedule(other, "agent3")

    draw_multi_agent_schedule(schedule)
