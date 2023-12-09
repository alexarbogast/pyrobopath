import numpy as np
import quaternion

import rospy
import actionlib
import tf2_ros

from pyrobopath.collision_detection import FCLRobotBBCollisionModel
from pyrobopath.toolpath_scheduling import AgentModel

from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from cartesian_planning_server.srv import *


def transform_tf_to_np(transform):
    p = transform.translation
    q = transform.rotation

    p = np.array([p.x, p.y, p.z])
    q = np.quaternion(q.w, q.x, q.y, q.z)

    np_tf = np.identity(4, dtype=np.float64)
    np_tf[:3, :3] = quaternion.as_rotation_matrix(q)
    np_tf[:3, 3] = p
    return np_tf


class AgentExecutionContext(object):
    def __init__(self, id, tf_buffer: tf2_ros.Buffer):
        self.id = id
        self.agent = AgentModel()
        
        self.read_parameters()
        self.update_tf(tf_buffer)
        self.agent.collision_model = FCLRobotBBCollisionModel(
            self.col_dim[0],
            self.col_dim[1],
            self.col_dim[2],
            self.agent.base_frame_position,
        )

        self.planning_client = rospy.ServiceProxy(
            f"{self.id}/cartesian_planning_server/plan_cartesian_trajectory",
            PlanCartesianTrajectory,
        )
        self.action_client = actionlib.SimpleActionClient(
            f"{self.id}/position_trajectory_controller/follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )

    def read_parameters(self):
        self.base_frame = rospy.get_param(f"{self.id}/base_frame", None)
        self.eef_frame = rospy.get_param(f"{self.id}/eef_frame", None)
        self.task_frame = rospy.get_param(f"{self.id}/task_frame", None)
        if self.base_frame is None or self.eef_frame is None or self.task_frame is None:
            rospy.logerr(
                "Could not find one of 'base_frame', 'eef_frame', or 'task_frame'"
                + f" for pyrobopath in namespace {self.id}"
            )

        self.agent.capabilities = rospy.get_param(f"{self.id}/capabilities", [0])
        self.col_dim = (
            rospy.get_param(f"{self.id}/collision/width", None),
            rospy.get_param(f"{self.id}/collision/length", None),
            rospy.get_param(f"{self.id}/collision/height", None),
        )
        if any([dim is None for dim in self.col_dim]):
            rospy.logerr(
                "Could not find collision model parameters "
                + f"(width or length or height) in namespace {self.id}/collision"
            )

    def update_tf(self, tf_buffer):
        try:
            base_to_task = tf_buffer.lookup_transform(
                self.task_frame, self.base_frame, rospy.Time()
            )
            task_to_base = tf_buffer.lookup_transform(
                self.base_frame, self.task_frame, rospy.Time()
            )
            eef_to_task = tf_buffer.lookup_transform(
                self.task_frame, self.eef_frame, rospy.Time()
            )
        except:
            rospy.logfatal(f"Failed to find transforms for agent {self.id}")
        self.eef_to_task = transform_tf_to_np(eef_to_task.transform)
        self.task_to_base = transform_tf_to_np(task_to_base.transform)
        self.base_to_task = transform_tf_to_np(base_to_task.transform)
        self.agent.home_position = self.eef_to_task[:3, 3]
        self.agent.base_frame_position = self.base_to_task[:3, 3]
