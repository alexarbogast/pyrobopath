import numpy as np
import quaternion
from typing import Hashable

# ros
import rospy
import actionlib
import tf2_ros
from geometry_msgs.msg import Pose
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from cartesian_planning_msgs.srv import *

# pyrobopath
from pyrobopath.collision_detection import FCLRobotBBCollisionModel
from pyrobopath.toolpath_scheduling import AgentModel


def transform_tf_to_np(transform):
    """Returns a 4x4 numpy homogeneous transformation from a provided tf frame

    :param transform: the tf transform to convert
    :type transform: :class:`geometry_msgs.msg.TransformStamped`
    :return: a 4x4 homogeneous transformation matrix
    :rtype: np.ndarray
    """
    p = transform.translation
    q = transform.rotation

    p = np.array([p.x, p.y, p.z])
    q = np.quaternion(q.w, q.x, q.y, q.z)

    np_tf = np.identity(4, dtype=np.float64)
    np_tf[:3, :3] = quaternion.as_rotation_matrix(q)
    np_tf[:3, 3] = p
    return np_tf


class AgentExecutionContext(object):
    """All items related to a single agents execution. The context is composed
    of the following components:

    PlanCartesianTrajectory client:
        This service client finds joint trajectories from Cartesian schedules
        found with pyrobopath.

    FollowJointTrajectory client:
        This action client executes joint trajectories found from the Cartesian
        planning server.

    This class also stores a copy of the AgentModel that is created from
    parameters on the ROS parameter server.

    :param id: A unique ID for the agent
    :type id: Hashable
    :param tf_buffer: A reference to the tf2 buffer for locating agent frames
    :type tf_buffer: tf2_ros.Buffer
    """

    def __init__(self, id: Hashable, tf_buffer: tf2_ros.Buffer):
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
        """Initialize the context with values from the ROS parameter server"""
        self.base_frame = rospy.get_param(f"{self.id}/base_frame")
        self.eef_frame = rospy.get_param(f"{self.id}/eef_frame")
        self.task_frame = rospy.get_param(f"{self.id}/task_frame")

        self.agent.capabilities = rospy.get_param(f"{self.id}/capabilities", [0])
        self.col_dim = (
            rospy.get_param(f"{self.id}/collision/length", None),
            rospy.get_param(f"{self.id}/collision/width", None),
            rospy.get_param(f"{self.id}/collision/height", None),
        )
        if any([dim is None for dim in self.col_dim]):
            rospy.logerr(
                "Could not find collision model parameters "
                + f"(width or length or height) in namespace {self.id}/collision"
            )

        self.joint_home = rospy.get_param(f"{self.id}/home_position")
        eef_rotation = rospy.get_param(f"{self.id}/eef_rotation", [1.0, 0.0, 0.0, 0.0])
        eef_rotation = [float(r) for r in eef_rotation]
        self.eef_rotation = np.quaternion(
            eef_rotation[0], eef_rotation[1], eef_rotation[2], eef_rotation[3]
        )

        self.agent.velocity = rospy.get_param(f"{self.id}/velocity")
        self.agent.travel_velocity = rospy.get_param(
            f"{self.id}/travel_velocity", self.agent.velocity
        )

    def update_tf(self, tf_buffer: tf2_ros.Buffer):
        """Updates the values of the task frame, base frame, and end effector
        frame from the tf2 buffer

        :param tf_buffer: the buffer from which to update the frame data
        :type tf_buffer: :class:`tf2_ros.Buffer`
        """
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

    def create_pose(self, point: np.ndarray):
        """Create a pose from a given point that aligns the robot configuration
        with that required from an FCLRobotBBCollisionModel

        The default pose is a frame that is initially coincident with the robot's
        base frame. This frame is rotated about the vertical z-axis until the
        x-axis aims towards the end effector. Then the frame is translated to
        `point`.

        :param point: A 3D point from which to make the pose
        :type point: np.ndarray
        :return: A pose from the provided point
        :rtype: geometry_msgs.msg.Pose
        """
        pose = Pose()
        pose.position.x = point[0]
        pose.position.y = point[1]
        pose.position.z = point[2]

        theta = np.arctan2(point[1], point[0])
        q = np.quaternion(np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2))
        rot = q * self.eef_rotation

        pose.orientation.w = rot.w
        pose.orientation.x = rot.x
        pose.orientation.y = rot.y
        pose.orientation.z = rot.z
        return pose
