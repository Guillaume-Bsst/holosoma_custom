# Holosoma — Inference & Sim-to-Sim

All commands are run from the `holosoma/` directory with conda (no Apptainer needed).
See [INSTALL.md](INSTALL.md) for environment setup.

> To obtain an ONNX model from a training checkpoint, see [EVALUATION.md](EVALUATION.md).

---

## Real robot inference

### Via native SDK (Unitree)

```bash
source scripts/source_inference_setup.sh
python src/holosoma_inference/holosoma_inference/run_policy.py \
    inference:g1-29dof-loco \
    --task.model-path https://wandb.ai/guibsst-inria/WholeBodyTracking/runs/dcpxirww/files/model_16000.onnx \
    --task.use-sim-time \
    --task.rl-rate 50
```

### Via ROS2 (real robot)

Set `--robot.sdk-type=ros2` to route commands and state through ROS2 topics instead of the
native SDK. A bridge node on the robot side is needed to relay topics to/from the native SDK.

```bash
source scripts/source_inference_setup.sh
python src/holosoma_inference/holosoma_inference/run_policy.py \
    inference:g1-29dof-loco \
    --robot.sdk-type=ros2 \
    --task.model-path path/to/model.onnx
```

---

## Sim-to-sim — MuJoCo

Run the policy against a local MuJoCo simulation via ROS2. Requires two terminals.

**Terminal 1 — Simulator:**

```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py \
    simulator:mujoco \
    robot:g1-29dof \
    terrain:terrain-locomotion-plane \
    --robot.bridge.sdk-type=ros2 \
    --simulator.config.bridge.enabled=True \
    --simulator.config.bridge.interface=lo
```

**Terminal 2 — Policy:**

> Use `inference:g1-27dof-wbt` or `inference:g1-29dof-wbt` depending on the DOF
> count the model was trained with (27-DOF models lock waist roll/pitch).

```bash
source scripts/source_inference_setup.sh
python src/holosoma_inference/holosoma_inference/run_policy.py \
    inference:g1-29dof-wbt \
    --robot.sdk-type=ros2 \
    --task.model-path wandb://guibsst-inria/WholeBodyTracking/dcpxirww/model_16000.onnx \
    --task.no-use-joystick \
    --task.use-sim-time \
    --task.rl-rate 50 \
    --task.interface lo
```

ROS2 topics used:

| Direction | Topic | Type | Content |
|-----------|-------|------|---------|
| Sim → Policy | `/holosoma/low_state` | `sensor_msgs/JointState` | Joint positions and velocities |
| Sim → Policy | `/holosoma/imu` | `sensor_msgs/Imu` | Orientation quaternion and angular velocity |
| Policy → Sim | `/holosoma/low_cmd` | `sensor_msgs/JointState` | Target positions, velocities, feedforward torques |
| Policy → Sim | `/holosoma/pd_gains` | `sensor_msgs/JointState` | PD gains (KP in position, KD in velocity) |

---

## Sim-to-sim — PyBullet (unitree_simulation)

> **Prerequisite:** Requires a separate ROS2 workspace built from
> [inria-paris-robotics-lab/unitree_control_interface](https://github.com/inria-paris-robotics-lab/unitree_control_interface).
> Follow the installation instructions in that repository. The workspace ends up at
> `unitree_ros2/cyclonedds_ws/` alongside `holosoma/`.

### Manual launch (4 terminals)

All terminals are run from `holosoma/`.

**Terminal 1 — PyBullet simulation**:

```bash
mamba activate unitree_control_interface
source ../unitree_ros2/cyclonedds_ws/install/setup.bash
source <(ros2 run unitree_control_interface autoset_environment_dds.py SIMULATION)
ros2 launch unitree_simulation launch_sim.launch.py robot:=g1 unlock_base:=False
```

**Terminal 2 — Policy**:

> Use `inference:g1-27dof-wbt` or `inference:g1-29dof-wbt` depending on the DOF
> count the model was trained with (27-DOF models lock waist roll/pitch).

```bash
source scripts/source_inference_setup.sh
source ../unitree_ros2/cyclonedds_ws/install/setup.bash
source <(ros2 run unitree_control_interface autoset_environment_dds.py SIMULATION)
python src/holosoma_inference/holosoma_inference/run_policy.py \
    inference:g1-29dof-wbt \
    --robot.sdk-type=ros2 \
    --task.model-path https://wandb.ai/guibsst-inria/WholeBodyTracking/runs/dcpxirww/files/model_16000.onnx \
    --task.use-sim-time \
    --task.rl-rate 50
```

> **Once the policy is running, immediately press `ENTER` then `]` to activate the policy.**

**Terminal 3 — Watchdog**:

```bash
mamba activate unitree_control_interface
source ../unitree_ros2/cyclonedds_ws/install/setup.bash
source <(ros2 run unitree_control_interface autoset_environment_dds.py SIMULATION)
ros2 launch unitree_control_interface watchdog.launch.py robot_type:=g1
```

**Terminal 4 — Bridge**:

```bash
mamba activate unitree_control_interface
source ../unitree_ros2/cyclonedds_ws/install/setup.bash
source <(ros2 run unitree_control_interface autoset_environment_dds.py SIMULATION)
python src/holosoma_inference/holosoma_inference/unitree_pybullet_bridge.py
```

The bridge will automatically move the robot to the standing configuration (~5 s), then hand control over to the policy.

> **27-DOF / 29-DOF auto-detection:** The bridge automatically detects whether the
> connected policy outputs 27 DOF (G1 base) or 29 DOF (G1 pro) from the first
> `/holosoma/low_cmd` message. When a 29-DOF policy is detected, `waist_roll` and
> `waist_pitch` commands are dropped before forwarding to the hardware (which only
> has 27 actuated joints). When a 27-DOF policy is detected, commands are passed
> through directly. The detected mode is logged at startup:
> `Policy DOF auto-detected: 27 (G1 base 27-DOF)` or `29 (G1 pro 29-DOF)`.

Then press `s` in the policy terminal to start the motion clip.