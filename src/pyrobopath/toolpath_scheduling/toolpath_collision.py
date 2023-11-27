from typing import List, Dict, Hashable
from intervaltree import IntervalTree

from ..collision_detection import (
    Trajectory,
    TrajectoryPoint,
    trajectory_collision_query,
)
from ..scheduling import Interval
from .schedule import ContourEvent, ToolpathSchedule, MultiAgentToolpathSchedule
from .system_model import AgentModel


def schedule_to_trajectory(
    schedule: ToolpathSchedule, t_start: float, t_end: float, default_state
) -> Trajectory:
    """
    Slice a toolpath schedule to a continuous trajectory.

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
    traj.add_traj_point(TrajectoryPoint(None, float("nan")))
    for e in schedule._events:
        if e.end < t_start:
            continue
        if e.start >= t_end:
            break
        event_traj = e.traj.slice(max(e.start, t_start), min(e.end, t_end))

        # remove duplicates between contiguous events
        if event_traj[0].time == traj.points[-1].time:
            event_traj.points.pop(0)

        traj += event_traj
    traj.points.pop(0)

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


def schedule_to_trajectories(
    schedule: ToolpathSchedule, t_start: float, t_end: float
) -> Trajectory:
    """
    Slice a toolpath schedule to a list of continuous trajectories.
    The trajectories of MoveEvents are sliced to fit in the window
    [t_start, t_end].

    Args:
        schedule: The schedule to be sliced
        t_start: start time of slice
        t_end: end time of slice

    Returns:
        List[Trajectory]: The list trajectories inferred from the schedule

    """

    trajs = []
    interval = Interval(t_start, t_end)
    for e in schedule._events:
        # event is not in interval
        if e.precedes(interval):
            continue
        if e.preceded_by(interval):
            break

        # use Allen's interval algebra to determine which parts of the
        # event's trajectory should be added to the list
        traj = Trajectory()
        if e.meets(interval):
            traj.add_traj_point(e.traj.points[-1])
        elif e.overlaps(interval):
            traj = e.traj.slice(interval.start, e.end)
        elif (
            e.starts(interval)
            or e.during(interval)
            or e.finishes(interval)
            or e.equals(interval)
        ):
            traj = e.traj
        elif e.finished_by(interval) or e.contains(interval) or e.started_by(interval):
            traj = e.traj.slice(interval.start, interval.end)
        elif e.overlapped_by(interval):
            traj = e.traj.slice(e.start, interval.end)
        elif e.met_by(interval):
            traj.add_traj_point(e.traj.points[0])
        trajs.append(traj)

    return trajs


def event_causes_collision(
    event: ContourEvent,
    agent: Hashable,
    schedule: MultiAgentToolpathSchedule,
    agent_models: Dict[str, AgentModel],
    threshold: float,
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
        agent_models (Dict[str, AgentModel]): Context info about the system
        threshold: The maximum collision checking step distance
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
            threshold,
        )
        if collide:
            return True
    return False


def chop_concurrent_trajectories(tr1: List[Trajectory], tr2: List[Trajectory]):
    tree2 = IntervalTree()
    for t in tr2:
        tree2.addi(t.start_time(), t.end_time(), t)

    traj_pairs = []
    for traj in tr1:
        overlap_set = tree2.overlap(traj.start_time(), traj.end_time())
        for inter in overlap_set:
            st = max(traj.start_time(), inter.begin)
            et = min(traj.end_time(), inter.end)
            traj_pairs.append((traj.slice(st, et), inter.data.slice(st, et)))
    return traj_pairs


def events_cause_collision(
    events: List[ContourEvent],
    agent: Hashable,
    schedule: MultiAgentToolpathSchedule,
    agent_models: Dict[str, AgentModel],
    threshold: float,
):
    """
    Determines if adding the 'events' to 'schedule' will cause a collision in the
    resulting trajectory.

    For timesteps with no event, agents are assumed to be in collision-free positions

    Args:
        events (List[ContourEvent]): The events to be added
        agent (Hashable): The agent for the event
        schedule (MultiAgentSchedule): A schedule that is assumed collision-free
        agent_models (Dict[str, AgentModel]): Context info about the system
        threshold: The maximum collision checking step distance
    """
    st = min([e.start for e in events])
    et = max([e.end for e in events])

    new_sched = ToolpathSchedule()
    for event in events:
        new_sched.add_event(event)

    event_trajs = schedule_to_trajectories(new_sched, st, et)
    for a, s in schedule.schedules.items():
        if a == agent:
            continue

        trajs = schedule_to_trajectories(s, st, et)
        concurrent_trajs = chop_concurrent_trajectories(event_trajs, trajs)
        for pair in concurrent_trajs:
            collide = trajectory_collision_query(
                agent_models[agent].collision_model,
                pair[0],
                agent_models[a].collision_model,
                pair[1],
                threshold,
            )

            if collide:
                return True
    return False
