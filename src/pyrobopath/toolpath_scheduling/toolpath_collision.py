from ..collision_detection import (
    Trajectory,
    TrajectoryPoint,
    trajectory_collision_query,
)
from itertools import tee
from .toolpath_scheduler import ContourEvent
from .schedule import ToolpathSchedule, MultiAgentToolpathSchedule


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def schedule_to_trajectory(
    schedule: ToolpathSchedule, t_start: float, t_end: float, default_state
) -> Trajectory:
    """Slice a toolpath schedule to a continuous trajectory.

    The trajectory is guaranteed to have points at times t_start and t_end.
    The trajectories in any ContourEvent that fall in the time window are
    concatenated in the resulting trajectory

    Args:
        schedule: The schedule to be sliced
        t_start: start time of slice
        t_end: end time of slice
        default state: The default trajectory state for times with no known
                       state in the schedule

    Returns:
        Trajectory: The trajectory inferred from the schedule

    """
    traj = Trajectory()
    for e in schedule._events:
        if e.end < t_start:
            continue
        if e.start >= t_end:
            break
        traj += e.traj.slice(max(e.start, t_start), min(e.end, t_end))

    # add endpoints if traj_start != t_start or traj_end != t_end
    if traj.n_points():
        if traj.start_time() > t_start:
            start_state = schedule.get_state(t_start, default_state)
            traj.insert_traj_point(0, TrajectoryPoint(start_state, t_start))
        if traj.end_time() < t_end:
            end_state = schedule.get_state(t_end, default_state)
            traj.add_traj_point(TrajectoryPoint(end_state, t_end))
    # add last known states as start and end of traj
    else:
        start_state = schedule.get_state(t_start, default_state)
        end_state = schedule.get_state(t_end, default_state)
        traj.insert_traj_point(0, TrajectoryPoint(start_state, t_start))
        traj.add_traj_point(TrajectoryPoint(end_state, t_end))

    return traj


def event_causes_collision(
    event: ContourEvent, agent, schedule: MultiAgentToolpathSchedule, agent_models
):
    """
    Determines if adding 'event' to 'schedule' will cause a collision in the
    resulting trajectory.

    The events in schedule from event.start_time() to schedule.end_time() are
    checked for collision. This is necessary because adding an event at a
    previous time can cause collisions in the future.

    Args:
        event (ContourEvent): The event to be added (with Event.data = Contour)
        agent (Hashable): The agent for the event
        schedule (MultiAgentSchedule): A schedule that is assumed collision-free
    """
    
    et = max(schedule.end_time(), event.end)

    new_sched = ToolpathSchedule()
    new_sched.add_event(event)
    event_traj = schedule_to_trajectory(
        new_sched, event.start, et, agent_models[agent].home_position
    )

    for a, s in schedule.schedules.items():
        if a == agent:
            continue

        traj = schedule_to_trajectory(s, event.start, et, agent_models[a].home_position)
        collide = trajectory_collision_query(
            agent_models[agent].collision_model,
            event_traj,
            agent_models[a].collision_model,
            traj,
        )
        if collide:
            return True
    return False
