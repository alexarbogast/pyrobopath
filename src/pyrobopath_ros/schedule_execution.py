"""Pyrobopath interfaces for schedule execution in ROS 

"""

import numpy as np
from collections import defaultdict
from gcodeparser import GcodeParser

# ros
import rospy
import tf2_ros
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from cartesian_planning_server.srv import *

# pyrobopath
from pyrobopath.toolpath import Toolpath
from pyrobopath.scheduling import DependencyGraph
from pyrobopath.toolpath_scheduling import *

from .agent_execution_context import AgentExecutionContext


def toolpath_from_gcode(filepath) -> Toolpath:
    """Parse gcode file to internal toolpath representation.

    :param filepath: The absolute path to a Gcode file
    :type filepath: str
    :return: A toolpath created from the input filepath
    :rtype: Toolpath
    """

    with open(filepath, "r") as f:
        gcode = f.read()
    parsed_gcode = GcodeParser(gcode)

    toolpath = Toolpath.from_gcode(parsed_gcode.lines)
    return toolpath


def print_schedule_info(schedule: MultiAgentToolpathSchedule):
    """Print the schedule duration, total number of events,
    and events for each agent

    :param schedule: The schedule to print info
    :type schedule: MultiAgentToolpathSchedule
    """
    print(f"Schedule duration: {schedule.duration()}")
    print(f"Total Events: {schedule.n_events()}")
    agents_info = "Agent Events: "
    for agent, sched in schedule.schedules.items():
        agents_info += f"{agent}: {len(sched._events)}, "
    print(agents_info)


class ScheduleExecution(object):
    """The ScheduleExecution class connects the necessary ROS interfaces to
    execute probopath `ToolpathSchedules` in ROS

    # TODO: List ROS parameters
    """

    def __init__(self) -> None:
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(0.1)

        self._namespaces = rospy.get_param("namespaces")
        self._contexts = dict()
        for ns in self._namespaces:
            self._build_agent_contexts(ns)

        rospy.loginfo("Waiting for plan_cartesian_trajectory servers...")
        for context in self._contexts.values():
            context.planning_client.wait_for_service()

        rospy.loginfo("Waiting for follow_trajectory_action servers...")
        for context in self._contexts.values():
            context.action_client.wait_for_server()

        self._schedule = None
        self._schedule_plan_buffer = defaultdict(list)

        self._initialize_pyrobopath()
        rospy.loginfo("Pyrobopath: ready to plan!")

    def _initialize_pyrobopath(self):
        """Initialize the pyrobopath toolpath planner and planning options
        from ROS parameters
        """
        agent_models = {id: context.agent for id, context in self._contexts.items()}

        retract_height = rospy.get_param("retract_height", 0.0)
        collision_offset = rospy.get_param("collision_offset", 1.0)
        collision_gap_threshold = rospy.get_param("collision_gap_treshold", 0.003)

        self._planner = MultiAgentToolpathPlanner(agent_models)
        self._options = PlanningOptions(
            retract_height=retract_height,
            collision_offset=collision_offset,
            collision_gap_threshold=collision_gap_threshold,
        )

    def move_home(self):
        """Moves all agents to the joint positions in the `/{ns}/home_position`
        parameter.
        """
        # send trajectories to home
        for id in self._contexts.keys():
            start_state = rospy.wait_for_message(f"/{id}/joint_states", JointState)

            point_start = JointTrajectoryPoint()
            point_start.positions = start_state.position
            point_start.time_from_start = rospy.Duration(0.0)

            point_goal = JointTrajectoryPoint()
            point_goal.positions = self._contexts[id].joint_home
            point_goal.time_from_start = rospy.Duration(2.0)

            goal = FollowJointTrajectoryGoal()
            goal.trajectory.joint_names = start_state.name
            goal.trajectory.points = [point_start, point_goal]
            self._contexts[id].action_client.send_goal(goal)

        # wait for completion
        for id in self._contexts.keys():
            self._contexts[id].action_client.wait_for_result()

    def _build_agent_contexts(self, id: str):
        """Build an AgentExecutionContext with a unique id

        :param id: Unique id for agent.
        :type id: str
        """
        self._contexts[id] = AgentExecutionContext(id, self.tf_buffer)

    def plan_toolpath(
        self, toolpath: Toolpath, dependency_graph: DependencyGraph = None
    ):
        """Finds the schedule for the provided toolpath and performs
        Cartesian motion planning on the resulting schedule.

        If no dependency graph is provided, a default all-to-all dependency
        graph is created between the layers in the toolpath. The resulting plan
        is stored internally.

        :param toolpath: The pyrobopath toolpath
        :type toolpath: Toolpath
        :param dependency_graph: an optional dependency graph, defaults to None
        :type dependency_graph: DependencyGraph, optional
        """
        if dependency_graph is None:
            dependency_graph = create_dependency_graph_by_layers(toolpath)

        for context in self._contexts.values():
            context.update_tf(self.tf_buffer)

        """ Schedule multi agent toolpath """
        rospy.loginfo(f"\n{(50 * '#')}\nScheduling Toolpath:\n{(50 * '#')}\n")
        self._schedule = self._planner.plan(toolpath, dependency_graph, self._options)
        rospy.loginfo(f"\n{(50 * '#')}\nFound Toolpath Plan!\n{(50 * '#')}\n")
        print_schedule_info(self._schedule)
        """ Cartesian motion planning for schedule events """
        self._plan_multi_agent_schedule(self._schedule)

    def execute_plan(self):
        """Executes the plan created by plan_toolpath"""
        rospy.loginfo(f"\n\n{(50 * '#')}\nExecuting Schedule\n{(50 * '#')}\n")
        start_time = rospy.get_time()
        rate = rospy.Rate(10)
        while any(self._schedule_plan_buffer.values()) and not rospy.is_shutdown():
            now = rospy.get_time()
            for agent, plans in self._schedule_plan_buffer.items():
                if not plans or now - start_time < plans[0][0]:
                    continue

                _, jt_goal = self._schedule_plan_buffer[agent].pop(0)
                rospy.loginfo(f"Starting event for {agent}")
                self._contexts[agent].action_client.send_goal(jt_goal)
            rate.sleep()

    def _plan_multi_agent_schedule(self, schedule: MultiAgentToolpathSchedule):
        """Populates the schedule plan buffer with motion plans from each
        event in `schedule`.

        :param schedule: The schedule
        :type schedule: MultiAgentToolpathSchedule
        """

        rospy.loginfo(f"\n{(50 * '#')}\nPlanning events\n{(50 * '#')}\n")
        rospy.loginfo("Planning and buffering events in schedule")
        for agent, sched in schedule.schedules.items():
            start_state = rospy.wait_for_message(f"/{agent}/joint_states", JointState)
            for event in sched._events:
                resp = self._plan_event(event, agent, start_state)
                if resp.success:
                    # create trajectory action server goal
                    goal = FollowJointTrajectoryGoal()
                    goal.trajectory = resp.trajectory
                    self._schedule_plan_buffer[agent].append((event.start, goal))

                    start_state.position = resp.trajectory.points[-1].positions
                    start_state.velocity = resp.trajectory.points[-1].velocities

    def _plan_event(
        self, event: MoveEvent, agent, start_state: JointState
    ):
        """Peforms Cartesian motion planning for a pyrobopath MoveEvent

        :param event: event with cartesian path
        :type event: MoveEvent
        :param agent: the agent to plan for
        :type agent: Hashable
        :param start_state: the starting joint configuration
        :type start_state: JointState
        
        :return: the response from the cartesian planning server
        :rtype: PlanCartesianTrajectoryResponse
        """
        context = self._contexts[agent]
        path_base = [(context.task_to_base @ np.array([*p, 1]))[:3] for p in event.data]
        req = PlanCartesianTrajectoryRequest()
        req.start_state = start_state

        for point in path_base:
            pose = context.create_pose(point)
            req.path.append(pose)

        if isinstance(event, ContourEvent):
            req.velocity = context.agent.velocity
        else:
            req.velocity = context.agent.travel_velocity

        resp = None
        try:
            resp = context.planning_client(req)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to plan cartesian trajectory: " + str(e))
        return resp
