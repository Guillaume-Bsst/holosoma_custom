#!/usr/bin/env python3
"""Bridge node: unitree_simulation (PyBullet) ↔ holosoma inference policy via ROS2.

Translates between the unitree_control_interface (27 actuated DOF) and
the holosoma policy format (27 or 29 DOF), bridging these ROS2 topics:

  Simulation → Bridge → holosoma policy:
    /lowstate  ──►  /holosoma/low_state  (sensor_msgs/JointState)  joint pos & vel
    /lowstate  ──►  /holosoma/imu        (sensor_msgs/Imu)          quaternion & gyro

  holosoma policy → Bridge → Simulation:
    /holosoma/low_cmd   ──►  /lowcmd  (via G1ControlInterface)
    /holosoma/pd_gains  ──►  /lowcmd  (via G1ControlInterface)

The bridge auto-detects the policy DOF mode from the first /holosoma/low_cmd
message:
  - 27 DOF (G1 base): direct pass-through, no conversion needed.
  - 29 DOF (G1 pro):  waist_roll (idx 13) and waist_pitch (idx 14) are dropped.

Joint mapping (G1 27-DOF hardware ↔ 29-DOF policy):
  unitree[0:13]  ↔  holosoma[0:13]   left leg + right leg + waist_yaw
  0.0 (fixed)    ↔  holosoma[13]     waist_roll  (locked in mode_machine=6)
  0.0 (fixed)    ↔  holosoma[14]     waist_pitch (locked in mode_machine=6)
  unitree[13:27] ↔  holosoma[15:29]  left arm + right arm
"""

import threading

import numpy as np
import rclpy
from loguru import logger
from rclpy.node import Node
from std_msgs.msg import Empty
from sensor_msgs.msg import Imu, JointState
from sdk import ros2
from unitree_hg.msg import LowState

try:
    import zmq
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyzmq"])
    import zmq

from unitree_control_interface_py import G1ControlInterface

# ── Joint dimension constants ──────────────────────────────────────────────────
N_UNITREE  = 27   # G1ControlInterface actuated DOF (waist_roll + waist_pitch are locked)
N_HOLOSOMA = 29   # holosoma g1-29dof config DOF

# Joint limits in 27-DOF unitree order (from g1_default_limits.yaml) — for logging only.
_Q_MIN = np.array([-2.5307,-0.5236,-2.7576,-0.087267,-0.88,-0.2618,
                   -2.5307,-2.9671,-2.7576,-0.087267,-0.88,-0.2618,
                   -2.618,-3.0892,-1.5882,-2.618,-1.0472,-1.972222054,-1.614429558,-1.614429558,
                   -3.0892,-2.2515,-2.618,-1.0472,-1.972222054,-1.614429558,-1.614429558])
_Q_MAX = np.array([2.8798,2.9671,2.7576,2.8798,0.53,0.2618,
                   2.8798,0.5236,2.7576,2.8798,0.53,0.2618,
                   2.618,2.6704,2.2515,2.618,2.0944,1.972222054,1.614429558,1.614429558,
                   2.6704,1.5882,2.618,2.0944,1.972222054,1.614429558,1.614429558])
_JOINT_NAMES_27 = [
    "left_hip_pitch","left_hip_roll","left_hip_yaw","left_knee","left_ankle_pitch","left_ankle_roll",
    "right_hip_pitch","right_hip_roll","right_hip_yaw","right_knee","right_ankle_pitch","right_ankle_roll",
    "waist_yaw",
    "left_shoulder_pitch","left_shoulder_roll","left_shoulder_yaw","left_elbow","left_wrist_roll","left_wrist_pitch","left_wrist_yaw",
    "right_shoulder_pitch","right_shoulder_roll","right_shoulder_yaw","right_elbow","right_wrist_roll","right_wrist_pitch","right_wrist_yaw",
]

# G1 standing configuration in G1ControlInterface URDF order (27 values).
# From unitree_simulation G1Configuration: hip_pitch=-0.5, knee=1.0, ankle_pitch=-0.5
_G1_STAND_Q = [
    -0.5, 0.0, 0.0, 1.0, -0.5, 0.0,          # left  leg: hip_pitch/roll/yaw, knee, ankle_pitch/roll
    -0.5, 0.0, 0.0, 1.0, -0.5, 0.0,          # right leg: same
     0.0,                                      # waist_yaw
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,      # left  arm (7 joints)
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,      # right arm (7 joints)
]


def _unitree_to_holosoma(arr: np.ndarray, n_policy: int) -> np.ndarray:
    """Expand 27-DOF unitree array → policy DOF format.

    If *n_policy* is 27 the array is returned as-is.
    If *n_policy* is 29 zeros are inserted for locked waist joints.
    """
    if n_policy == N_UNITREE:
        return arr.copy()
    out = np.zeros(N_HOLOSOMA)
    out[:13] = arr[:13]   # left leg + right leg + waist_yaw
    # out[13] = 0.0        # waist_roll  (locked, stays 0)
    # out[14] = 0.0        # waist_pitch (locked, stays 0)
    out[15:] = arr[13:]   # left arm + right arm
    return out


def _holosoma_to_unitree(arr: np.ndarray, n_policy: int) -> np.ndarray:
    """Contract policy DOF array → 27-DOF unitree.

    If *n_policy* is 27 the array is returned as-is.
    If *n_policy* is 29, waist_roll (idx 13) and waist_pitch (idx 14) are dropped.
    """
    if n_policy == N_UNITREE:
        return arr.copy()
    out = np.zeros(N_UNITREE)
    out[:13] = arr[:13]   # left leg + right leg + waist_yaw
    out[13:] = arr[15:]   # left arm + right arm (skip indices 13 and 14)
    return out


class UnitreePybulletBridgeNode(Node):
    """ROS2 bridge connecting unitree_simulation (PyBullet) to a holosoma inference policy.

    Startup sequence:
      1. Robot moves to standing configuration (via G1ControlInterface start routine, ~5 s).
      2. Bridge auto-unlocks with a 1 s transition.
      3. Policy commands arriving on /holosoma/low_cmd are forwarded to the simulation.
    """

    # holosoma G1 joint names in 29-DOF order (with _joint suffix used by holosoma configs)
    _JOINT_NAMES_29 = [
        "left_hip_pitch_joint",       "left_hip_roll_joint",       "left_hip_yaw_joint",
        "left_knee_joint",            "left_ankle_pitch_joint",    "left_ankle_roll_joint",
        "right_hip_pitch_joint",      "right_hip_roll_joint",      "right_hip_yaw_joint",
        "right_knee_joint",           "right_ankle_pitch_joint",   "right_ankle_roll_joint",
        "waist_yaw_joint",            "waist_roll_joint",          "waist_pitch_joint",
        "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",  "left_shoulder_yaw_joint",
        "left_elbow_joint",           "left_wrist_roll_joint",     "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint",          "right_wrist_roll_joint",    "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    # holosoma G1 joint names in 27-DOF order (waist_roll + waist_pitch removed)
    _JOINT_NAMES_27 = [
        n for n in _JOINT_NAMES_29 if n not in ("waist_roll_joint", "waist_pitch_joint")
    ]

    def __init__(self):
        super().__init__("unitree_pybullet_bridge")

        # ── holosoma publishers ──────────────────────────────────────────────
        self._state_pub = self.create_publisher(JointState, "/holosoma/low_state", 10)
        self._imu_pub   = self.create_publisher(Imu,        "/holosoma/imu",       10)
        self._unlock_pub = self.create_publisher(Empty,     "/unlock_base",        10)

        # ── IMU buffer — subscribe to /lowstate to access imu_state directly ─
        # (G1ControlInterface callback only exposes joint positions/velocities)
        self._imu_lock = threading.Lock()
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        self._gyro = np.zeros(3)
        self.create_subscription(LowState, "/lowstate", self._lowstate_imu_cb, 10)

        # ── command buffers (populated by holosoma policy callbacks) ─────────
        self._cmd_lock     = threading.Lock()
        self._cmd_q        = np.array(_G1_STAND_Q)    # default: hold standing pose
        self._cmd_dq       = np.zeros(N_UNITREE)
        self._cmd_tau      = np.zeros(N_UNITREE)
        self._kp           = np.full(N_UNITREE, 75.0) # G1 default stiffness
        self._kd           = np.full(N_UNITREE,  1.0) # G1 default damping
        self._cmd_received = False
        self._n_policy: int = 0  # auto-detected from first /holosoma/low_cmd message

        self.create_subscription(JointState, "/holosoma/low_cmd",  self._low_cmd_cb,  10)
        self.create_subscription(JointState, "/holosoma/pd_gains", self._pd_gains_cb, 10)

        # ── G1ControlInterface (handles watchdog, safety, start routine) ─────
        self._robot_if = G1ControlInterface(self)
        self._robot_if.register_callback(self._joint_state_cb)
        self._unlocked = False

        # ── ZMQ clock publisher — feeds sim time to WBT policy ───────────────
        self._zmq_context = zmq.Context()
        self._zmq_clock_socket = self._zmq_context.socket(zmq.PUB)
        self._zmq_clock_socket.bind("tcp://*:5555")

        self._robot_if.start_async(_G1_STAND_Q)
        logger.info("Bridge started — moving to standing config, waiting for watchdog...")

    # ── G1ControlInterface callback (~1 kHz) ──────────────────────────────────

    def _joint_state_cb(self, t: float, q, dq, ddq):
        """Receive joint state (27 DOF, URDF order); publish to holosoma and forward commands."""
        try:
            self._zmq_clock_socket.send_string(str(int(t * 1000)), zmq.NOBLOCK)
        except zmq.Again:
            pass

        # Log actual joint positions that exceed watchdog limits
        for i, (q_i, qmin, qmax, name) in enumerate(zip(q, _Q_MIN, _Q_MAX, _JOINT_NAMES_27)):
            if q_i < qmin or q_i > qmax:
                logger.warning(f"Actual q out of limits: {name}[{i}] = {q_i:.4f} (limits [{qmin:.4f}, {qmax:.4f}])")

        q_np  = np.asarray(q)
        dq_np = np.asarray(dq)

        # Auto-unlock once the robot has reached the standing configuration
        if not self._unlocked and self._robot_if.can_be_unlocked():
            self._robot_if.unlock(transition_duration=1.0)
            self._unlocked = True
            logger.info(
                "Standing configuration reached — unlocked with 1 s transition. "
                "Waiting for policy commands on /holosoma/low_cmd ..."
            )
            self._unlock_pub.publish(Empty())

        # Publish joint state to holosoma (expand to policy DOF if needed)
        n_pol = self._n_policy if self._n_policy else N_HOLOSOMA  # default 29 before detection
        now = self.get_clock().now().to_msg()

        state_msg = JointState()
        state_msg.header.stamp = now
        state_msg.name         = self._JOINT_NAMES_27 if n_pol == N_UNITREE else self._JOINT_NAMES_29
        state_msg.position     = _unitree_to_holosoma(q_np, n_pol).tolist()
        state_msg.velocity     = _unitree_to_holosoma(dq_np, n_pol).tolist()
        self._state_pub.publish(state_msg)

        # Publish IMU data (latest values from _lowstate_imu_cb)
        with self._imu_lock:
            quat = self._quat.copy()
            gyro = self._gyro.copy()

        imu_msg = Imu()
        imu_msg.header.stamp       = now
        imu_msg.header.frame_id    = "base_link"
        # unitree quaternion is [w, x, y, z]; holosoma ros2_interface expects
        # sensor_msgs/Imu with (x, y, z, w) and internally converts to (w, x, y, z)
        imu_msg.orientation.w      = float(quat[0])
        imu_msg.orientation.x      = float(quat[1])
        imu_msg.orientation.y      = float(quat[2])
        imu_msg.orientation.z      = float(quat[3])
        imu_msg.angular_velocity.x = float(gyro[0])
        imu_msg.angular_velocity.y = float(gyro[1])
        imu_msg.angular_velocity.z = float(gyro[2])
        self._imu_pub.publish(imu_msg)

        # Forward policy commands once ready
        if self._robot_if.can_be_controlled() and self._cmd_received:
            with self._cmd_lock:
                q_cmd   = self._cmd_q.copy()
                dq_cmd  = self._cmd_dq.copy()
                tau_cmd = self._cmd_tau.copy()
                kp      = self._kp.copy()
                kd      = self._kd.copy()
            self._robot_if.send_command(
                q_cmd.tolist(), dq_cmd.tolist(), tau_cmd.tolist(),
                kp.tolist(), kd.tolist(),
            )

    def _lowstate_imu_cb(self, msg: LowState):
        """Buffer IMU data from the raw unitree LowState message.

        unitree imu_state.quaternion: [w, x, y, z]
        unitree imu_state.gyroscope:  angular velocity in base (local) frame [rad/s]
        """
        with self._imu_lock:
            q = msg.imu_state.quaternion
            self._quat[:] = [q[0], q[1], q[2], q[3]]
            g = msg.imu_state.gyroscope
            self._gyro[:] = [g[0], g[1], g[2]]

    # ── holosoma policy callbacks ─────────────────────────────────────────────

    def _detect_policy_dof(self, n_values: int) -> int:
        """Auto-detect policy DOF from the first message and log it once."""
        if self._n_policy:
            return self._n_policy
        if n_values <= N_UNITREE:
            self._n_policy = N_UNITREE
        else:
            self._n_policy = N_HOLOSOMA
        logger.info(
            f"Policy DOF auto-detected: {self._n_policy} "
            f"({'G1 base 27-DOF' if self._n_policy == N_UNITREE else 'G1 pro 29-DOF'})"
        )
        return self._n_policy

    def _low_cmd_cb(self, msg: JointState):
        """Receive joint commands from holosoma policy (27 or 29 DOF) and map to 27 DOF."""
        with self._cmd_lock:
            n_pol = self._detect_policy_dof(len(msg.position))

            h_q = np.zeros(n_pol)
            n = min(len(msg.position), n_pol)
            h_q[:n] = msg.position[:n]
            self._cmd_q = _holosoma_to_unitree(h_q, n_pol)

            # Log any q_target that exceeds watchdog limits
            for i, (q, qmin, qmax, name) in enumerate(zip(self._cmd_q, _Q_MIN, _Q_MAX, _JOINT_NAMES_27)):
                if q < qmin or q > qmax:
                    logger.warning(f"Policy q_target out of limits: {name}[{i}] = {q:.4f} (limits [{qmin:.4f}, {qmax:.4f}])")

            if msg.velocity:
                h_dq = np.zeros(n_pol)
                nv = min(len(msg.velocity), n_pol)
                h_dq[:nv] = msg.velocity[:nv]
                self._cmd_dq = _holosoma_to_unitree(h_dq, n_pol)

            if msg.effort:
                h_tau = np.zeros(n_pol)
                nt = min(len(msg.effort), n_pol)
                h_tau[:nt] = msg.effort[:nt]
                self._cmd_tau = _holosoma_to_unitree(h_tau, n_pol)

            self._cmd_received = True

    def _pd_gains_cb(self, msg: JointState):
        """Receive PD gains from holosoma policy (27 or 29 DOF) and map to 27 DOF."""
        with self._cmd_lock:
            n_pol = self._n_policy if self._n_policy else N_HOLOSOMA
            if msg.position:
                h_kp = np.full(n_pol, 75.0)
                n = min(len(msg.position), n_pol)
                h_kp[:n] = msg.position[:n]
                self._kp = _holosoma_to_unitree(h_kp, n_pol)
            if msg.velocity:
                h_kd = np.full(n_pol, 1.0)
                nv = min(len(msg.velocity), n_pol)
                h_kd[:nv] = msg.velocity[:nv]
                self._kd = _holosoma_to_unitree(h_kd, n_pol)


def main():
    rclpy.init()
    node = UnitreePybulletBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
