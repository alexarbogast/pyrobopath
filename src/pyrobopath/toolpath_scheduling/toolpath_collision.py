from ..scheduling import Schedule, MultiAgentSchedule
from ..collision_detection import Trajectory, trajectory_collision_query

from .schedule_toolpath import ContourEvent


def toolpath_schedule_to_trajectory(schedule: Schedule) -> Trajectory:
    traj = Trajectory()
    for event in schedule._events:
        traj += Trajectory.from_const_vel_path(
            event.data.path, event.velocity, start_time=event.start
        )
    return traj


def event_causes_collision(
    event: ContourEvent, agent, schedule: MultiAgentSchedule, collision_models
):
    """
    Determines if adding 'event' to 'schedule' will cause a collision in the
    resulting trajectory

    Args:
        event (ContourEvent): The event to be added (with Event.data = Contour)
        agent (Hashable): The agent for the event
        schedule (MultiAgentSchedule): A schedule that is assumed collision-free
    """
    new_schedule = Schedule()
    new_schedule.add_event(event)
    new_traj = toolpath_schedule_to_trajectory(new_schedule)

    sliced_schedule = schedule.slice(event.start, event.start + event.duration)
    for a, s in sliced_schedule.schedules.items():
        if a == agent or s.n_events() == 0:
            continue

        traj = toolpath_schedule_to_trajectory(s)
        traj = traj.slice(event.start, event.start + event.duration)
        collide = trajectory_collision_query(
            collision_models[agent], new_traj, collision_models[a], traj
        )
        if collide:
            return True
    return False
