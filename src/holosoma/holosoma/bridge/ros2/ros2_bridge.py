"""ROS2 bridge between the simulator and holosoma_inference.

Mirrors the topics defined by the inference-side ROS2Interface so that
run_sim.py and run_policy.py can communicate over standard ROS2 messages:

    Sim  ──publish──►  /holosoma/low_state  (sensor_msgs/JointState)
    Sim  ──publish──►  /holosoma/imu        (sensor_msgs/Imu)
    Sim  ◄──subscribe── /holosoma/low_cmd   (sensor_msgs/JointState)
    Sim  ◄──subscribe── /holosoma/pd_gains  (sensor_msgs/JointState)
"""

import threading

import numpy as np
import rclpy
from loguru import logger
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState

from holosoma.bridge.base import BasicSdk2Bridge


class ROS2Bridge(BasicSdk2Bridge):
    """Simulator-side ROS2 bridge, compatible with the inference ROS2Interface."""

    def _init_sdk_components(self):
        """Initialize ROS2 node, publishers and subscribers."""
        if not rclpy.ok():
            rclpy.init()

        self._node = Node("holosoma_sim_bridge")

        # Publishers — match what ROS2Interface subscribes to
        self._state_pub = self._node.create_publisher(JointState, "/holosoma/low_state", 10)
        self._imu_pub = self._node.create_publisher(Imu, "/holosoma/imu", 10)

        # Subscribers — match what ROS2Interface publishes
        self._cmd_sub = self._node.create_subscription(
            JointState, "/holosoma/low_cmd", self._low_cmd_callback, 10
        )
        self._gains_sub = self._node.create_subscription(
            JointState, "/holosoma/pd_gains", self._pd_gains_callback, 10
        )

        # Command buffer
        self._lock = threading.Lock()
        self._cmd_q = np.zeros(self.num_motor)
        self._cmd_dq = np.zeros(self.num_motor)
        self._cmd_tau = np.zeros(self.num_motor)
        self._cmd_received = False

        # Default control gains from robot config (partial key matching)
        stiffness = self.robot.control.stiffness
        damping = self.robot.control.damping
        self._kp = np.zeros(self.num_motor)
        self._kd = np.zeros(self.num_motor)
        for i, dof_name in enumerate(self.robot.dof_names):
            name_clean = dof_name.replace("_joint", "")
            for key in stiffness:
                if key in name_clean:
                    self._kp[i] = stiffness[key]
                    self._kd[i] = damping[key]
                    break

        # Spin in background thread
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True)
        self._spin_thread.start()

        logger.info(
            f"ROS2Bridge topics: pub=[{self._state_pub.topic_name}, {self._imu_pub.topic_name}], "
            f"sub=[{self._cmd_sub.topic_name}, {self._gains_sub.topic_name}]"
        )

    def _low_cmd_callback(self, msg: JointState):
        """Handle incoming joint commands from the policy."""
        with self._lock:
            n = min(len(msg.position), self.num_motor)
            self._cmd_q[:n] = msg.position[:n]
            if msg.velocity:
                nv = min(len(msg.velocity), self.num_motor)
                self._cmd_dq[:nv] = msg.velocity[:nv]
            if msg.effort:
                nt = min(len(msg.effort), self.num_motor)
                self._cmd_tau[:nt] = msg.effort[:nt]
            self._cmd_received = True

    def _pd_gains_callback(self, msg: JointState):
        """Handle incoming PD gains from the policy (KP in position, KD in velocity)."""
        with self._lock:
            n = min(len(msg.position), self.num_motor)
            self._kp[:n] = msg.position[:n]
            if msg.velocity:
                nv = min(len(msg.velocity), self.num_motor)
                self._kd[:nv] = msg.velocity[:nv]

    def low_cmd_handler(self, msg=None):
        """Poll is not needed — commands arrive via ROS2 subscription callback."""

    def publish_low_state(self):
        """Publish joint state and IMU data via ROS2."""
        now = self._node.get_clock().now().to_msg()

        # --- Joint state ---
        positions, velocities, _accelerations = self._get_dof_states()
        actuator_forces = self._get_actuator_forces()

        state_msg = JointState()
        state_msg.header.stamp = now
        state_msg.name = list(self.robot.dof_names)
        state_msg.position = positions.tolist()
        state_msg.velocity = velocities.tolist()
        state_msg.effort = actuator_forces.tolist()
        self._state_pub.publish(state_msg)

        # --- IMU ---
        quaternion, gyro, _accel = self._get_base_imu_data()
        quat_np = quaternion.detach().cpu().numpy()  # [w, x, y, z]
        gyro_np = gyro.detach().cpu().numpy()

        imu_msg = Imu()
        imu_msg.header.stamp = now
        imu_msg.header.frame_id = "base_link"
        # ROS2 Imu uses (x, y, z, w)
        imu_msg.orientation.x = float(quat_np[1])
        imu_msg.orientation.y = float(quat_np[2])
        imu_msg.orientation.z = float(quat_np[3])
        imu_msg.orientation.w = float(quat_np[0])
        imu_msg.angular_velocity.x = float(gyro_np[0])
        imu_msg.angular_velocity.y = float(gyro_np[1])
        imu_msg.angular_velocity.z = float(gyro_np[2])
        self._imu_pub.publish(imu_msg)

    def compute_torques(self):
        """Compute PD torques from the latest received command."""
        with self._lock:
            if not self._cmd_received:
                return self.torques
            q_target = self._cmd_q.copy()
            dq_target = self._cmd_dq.copy()
            tau_ff = self._cmd_tau.copy()
            kp = self._kp.copy()
            kd = self._kd.copy()

        return self._compute_pd_torques(
            tau_ff=tau_ff,
            kp=kp,
            kd=kd,
            q_target=q_target,
            dq_target=dq_target,
        )

    def __del__(self):
        """Cleanup ROS2 resources."""
        if hasattr(self, "_node") and self._node is not None:
            self._node.destroy_node()
