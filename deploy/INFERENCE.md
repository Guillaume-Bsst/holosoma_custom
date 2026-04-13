# Holosoma — Inference & Sim-to-Sim (quick reference)

This file is a deploy-side quick reference for running inference.
For the full inference documentation, workflow details, and policy controls, see [src/holosoma_inference/README.md](../src/holosoma_inference/README.md).

All commands are run from the `holosoma/` directory.

## Setup

```bash
source scripts/source_inference_setup.sh
```

## Real robot inference

### Native SDK (Unitree)

```bash
python src/holosoma_inference/holosoma_inference/run_policy.py \
    inference:g1-29dof-loco \
    --task.model-path path/to/model.onnx \
    --task.use-sim-time \
    --task.rl-rate 50
```

### ROS2 robot inference

```bash
python src/holosoma_inference/holosoma_inference/run_policy.py \
    inference:g1-29dof-loco \
    --robot.sdk-type=ros2 \
    --task.model-path path/to/model.onnx
```

## Sim-to-sim — MuJoCo

Run the policy against a local MuJoCo simulation via ROS2. This requires two terminals.

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
    --task.model-path path/to/model.onnx \
    --task.no-use-joystick \
    --task.use-sim-time \
    --task.rl-rate 50 \
    --task.interface lo
```

## Notes

- `--task.use-sim-time` keeps the policy synchronized with the simulator.
- `--task.no-use-joystick` disables joystick control for sim-to-sim runs.
- For full deployment workflows, read [src/holosoma_inference/README.md](../src/holosoma_inference/README.md).

## See also

- [EVALUATION.md](EVALUATION.md) — ONNX export quick reference
- [src/holosoma_inference/docs/workflows/real-robot-locomotion.md](../src/holosoma_inference/docs/workflows/real-robot-locomotion.md)
- [src/holosoma_inference/docs/workflows/real-robot-wbt.md](../src/holosoma_inference/docs/workflows/real-robot-wbt.md)
- [src/holosoma_inference/docs/workflows/sim-to-sim-locomotion.md](../src/holosoma_inference/docs/workflows/sim-to-sim-locomotion.md)
- [src/holosoma_inference/docs/workflows/sim-to-sim-wbt.md](../src/holosoma_inference/docs/workflows/sim-to-sim-wbt.md)
