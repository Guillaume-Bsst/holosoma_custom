"""ROS2 robot interface for holosoma inference.

Publishes low-level joint commands and subscribes to robot state via ROS2 topics.
This allows integrating the holosoma policy into any ROS2-based robot stack.

Topics:
    Published:
        ~/low_cmd (sensor_msgs/JointState): Target joint positions, velocities, and efforts.
            - position: target joint positions (N,)
            - velocity: target joint velocities (N,)
            - effort: feedforward torques (N,)
            Joint names are taken from robot_config.dof_names.

    Subscribed:
        ~/low_state (sensor_msgs/JointState): Current robot joint state.
            - position: current joint positions (N,)
            - velocity: current joint velocities (N,)
        ~/imu (sensor_msgs/Imu): IMU orientation and angular velocity.
            - orientation: quaternion (x, y, z, w)
            - angular_velocity: (x, y, z)

Usage:
    python run_policy.py inference:g1-29dof-loco --robot.sdk-type=ros2 --task.model-path model.onnx
"""

import threading

import numpy as np
import rclpy
from loguru import logger
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState

from holosoma_inference.config.config_types import RobotConfig
from holosoma_inference.sdk.base.base_interface import BaseInterface


class ROS2Interface(BaseInterface):
    """Interface for robot communication via ROS2 topics."""

    def __init__(self, robot_config: RobotConfig, domain_id=0, interface_str=None, use_joystick=True):
        super().__init__(robot_config, domain_id, interface_str, use_joystick)

        self._kp_level = 1.0
        self._kd_level = 1.0
        self.num_dofs = robot_config.num_joints

        # State buffers (protected by lock)
        self._lock = threading.Lock()
        self._joint_pos = np.zeros(self.num_dofs)
        self._joint_vel = np.zeros(self.num_dofs)
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self._ang_vel = np.zeros(3)
        self._state_received = False

        self._init_ros2()
        logger.info("ROS2Interface initialized")

    def _init_ros2(self):
        """Initialize ROS2 node, publishers, and subscribers."""
        if not rclpy.ok():
            rclpy.init()

        self._node = Node("holosoma_policy")

        # Publisher: joint commands
        self._cmd_pub = self._node.create_publisher(JointState, "~/low_cmd", 10)

        # Subscribers: robot state
        self._state_sub = self._node.create_subscription(
            JointState, "~/low_state", self._low_state_callback, 10
        )
        self._imu_sub = self._node.create_subscription(
            Imu, "~/imu", self._imu_callback, 10
        )

        # Spin in background thread
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True)
        self._spin_thread.start()

        logger.info(
            f"ROS2 topics: pub={self._cmd_pub.topic_name}, "
            f"sub=[{self._state_sub.topic_name}, {self._imu_sub.topic_name}]"
        )

    def _low_state_callback(self, msg: JointState):
        """Handle incoming joint state."""
        with self._lock:
            n = min(len(msg.position), self.num_dofs)
            self._joint_pos[:n] = msg.position[:n]
            if msg.velocity:
                nv = min(len(msg.velocity), self.num_dofs)
                self._joint_vel[:nv] = msg.velocity[:nv]
            self._state_received = True

    def _imu_callback(self, msg: Imu):
        """Handle incoming IMU data."""
        with self._lock:
            # ROS2 Imu uses (x, y, z, w), holosoma uses (w, x, y, z)
            self._quat = np.array([
                msg.orientation.w,
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
            ])
            self._ang_vel = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ])

    def get_low_state(self) -> np.ndarray:
        """Get robot state as numpy array.

        Returns:
            np.ndarray shape (1, 13+2N):
            [base_pos(3), quat(4), joint_pos(N), lin_vel(3), ang_vel(3), joint_vel(N)]
        """
        with self._lock:
            state = np.concatenate([
                np.zeros(3),           # base_pos (not available from topics)
                self._quat,            # quaternion (w, x, y, z)
                self._joint_pos,       # joint positions
                np.zeros(3),           # lin_vel (not available from topics)
                self._ang_vel,         # angular velocity from IMU
                self._joint_vel,       # joint velocities
            ])
        return state.reshape(1, -1)

    def send_low_command(
        self,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
        dof_pos_latest: np.ndarray = None,
        kp_override: np.ndarray = None,
        kd_override: np.ndarray = None,
    ):
        """Publish joint command to ROS2 topic."""
        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = list(self.robot_config.dof_names)
        msg.position = cmd_q.tolist()
        msg.velocity = cmd_dq.tolist()
        msg.effort = cmd_tau.tolist()
        self._cmd_pub.publish(msg)

    def get_joystick_msg(self):
        """No joystick via ROS2 — use keyboard or a separate Joy node."""
        return None

    def get_joystick_key(self, wc_msg=None):
        """No joystick via ROS2."""
        return None

    @property
    def kp_level(self):
        return self._kp_level

    @kp_level.setter
    def kp_level(self, value):
        self._kp_level = value

    @property
    def kd_level(self):
        return self._kd_level

    @kd_level.setter
    def kd_level(self, value):
        self._kd_level = value

    def __del__(self):
        """Cleanup ROS2 resources."""
        if hasattr(self, "_node") and self._node is not None:
            self._node.destroy_node()
