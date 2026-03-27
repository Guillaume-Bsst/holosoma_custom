# Holosoma — Inference & Sim-to-Sim

All commands are run from the `holosoma/` directory with conda (no Apptainer needed).
See [INSTALL.md](INSTALL.md) for environment setup.

---

## Real robot inference

### Via native SDK (Unitree)

```bash
source scripts/source_inference_setup.sh
python src/holosoma_inference/holosoma_inference/run_policy.py \
    inference:g1-29dof-loco \
    --task.model-path path/to/model.onnx
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
> Terminals 1–3 are run from `unitree_ros2/cyclonedds_ws/` with its conda environment active.

**Terminal 1 — PyBullet simulation** (from `unitree_ros2/cyclonedds_ws/`):

```bash
mamba activate unitree_control_interface
source install/setup.bash
source <(ros2 run unitree_control_interface autoset_environment_dds.py SIMULATION)
ros2 launch unitree_simulation launch_sim.launch.py robot:=g1
```

**Terminal 2 — Watchdog** (from `unitree_ros2/cyclonedds_ws/`):

```bash
mamba activate unitree_control_interface
source install/setup.bash
source <(ros2 run unitree_control_interface autoset_environment_dds.py SIMULATION)
ros2 launch unitree_control_interface watchdog.launch.py robot_type:=g1
```

**Terminal 3 — Bridge** (from `holosoma/`):

```bash
mamba activate unitree_control_interface
source ../unitree_ros2/cyclonedds_ws/install/setup.bash
source <(ros2 run unitree_control_interface autoset_environment_dds.py SIMULATION)
python src/holosoma_inference/holosoma_inference/unitree_pybullet_bridge.py
```

**Terminal 4 — Policy** (from `holosoma/`):

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