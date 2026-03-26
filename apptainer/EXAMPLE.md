# Holosoma Command Examples

All commands are run from the `holosoma/` directory unless otherwise specified.

---

## 1. Retargeting

### Robot-only (OMOMO)

```bash
apptainer exec --nv --writable-tmpfs --bind /run apptainer/holosoma.sif bash -c "
    mkdir -p /home/gbesset && \
    ln -sf /root/.holosoma_deps /home/gbesset/.holosoma_deps && \
    export MUJOCO_GL=egl && \
    export PYOPENGL_PLATFORM=egl && \
    unset DISPLAY && \
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hsretargeting && \
    cd src/holosoma_retargeting/holosoma_retargeting && \
    python examples/robot_retarget.py \
        --data_path demo_data/OMOMO_new \
        --task-type robot_only \
        --task-name sub3_largebox_003 \
        --data_format smplh
"
```

### Object interaction (OMOMO)

```bash
apptainer exec --nv --writable-tmpfs --bind /run apptainer/holosoma.sif bash -c "
    mkdir -p /home/gbesset && \
    ln -sf /root/.holosoma_deps /home/gbesset/.holosoma_deps && \
    export MUJOCO_GL=egl && \
    export PYOPENGL_PLATFORM=egl && \
    unset DISPLAY && \
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hsretargeting && \
    cd src/holosoma_retargeting/holosoma_retargeting && \
    python examples/robot_retarget.py \
        --data_path demo_data/OMOMO_new \
        --task-type object_interaction \
        --task-name sub3_largebox_003 \
        --data_format smplh
"
```

### Climbing

```bash
apptainer exec --nv --writable-tmpfs --bind /run apptainer/holosoma.sif bash -c "
    mkdir -p /home/gbesset && \
    ln -sf /root/.holosoma_deps /home/gbesset/.holosoma_deps && \
    export MUJOCO_GL=egl && \
    export PYOPENGL_PLATFORM=egl && \
    unset DISPLAY && \
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hsretargeting && \
    cd src/holosoma_retargeting/holosoma_retargeting && \
    python examples/robot_retarget.py \
        --data_path demo_data/climb \
        --task-type climbing \
        --task-name mocap_climb_seq_0 \
        --data_format mocap \
        --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf
"
```

---

## 2. Data Conversion

### Robot-only

```bash
apptainer exec --nv --writable-tmpfs --bind /run apptainer/holosoma.sif bash -c "
    mkdir -p /home/gbesset && \
    ln -sf /root/.holosoma_deps /home/gbesset/.holosoma_deps && \
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hsretargeting && \
    cd src/holosoma_retargeting/holosoma_retargeting && \
    sed -i 's/viewer = mjv/#viewer = mjv/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.opt/#viewer.opt/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.cam/#viewer.cam/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.sync/#viewer.sync/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.close/#viewer.close/g' data_conversion/convert_data_format_mj.py && \
    python data_conversion/convert_data_format_mj.py \
        --input_file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz \
        --output_fps 50 \
        --output_name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz \
        --data_format smplh \
        --object_name 'ground' \
        --once
"
```

### Robot + object

```bash
apptainer exec --nv --writable-tmpfs --bind /run apptainer/holosoma.sif bash -c "
    mkdir -p /home/gbesset && \
    ln -sf /root/.holosoma_deps /home/gbesset/.holosoma_deps && \
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hsretargeting && \
    cd src/holosoma_retargeting/holosoma_retargeting && \
    sed -i 's/viewer = mjv/#viewer = mjv/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.opt/#viewer.opt/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.cam/#viewer.cam/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.sync/#viewer.sync/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.close/#viewer.close/g' data_conversion/convert_data_format_mj.py && \
    python data_conversion/convert_data_format_mj.py \
        --input_file ./demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz \
        --output_fps 50 \
        --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz \
        --data_format smplh \
        --object_name 'largebox' \
        --has_dynamic_object \
        --once
"
```

---

## 3. Training

### Locomotion (MJWarp)

```bash
apptainer exec --nv --writable-tmpfs \
    --bind /run \
    --bind /dev/shm \
    --bind $SCRATCH/holosoma_logs:/workspace/holosoma/logs \
    --bind $SCRATCH/wandb_cache:/workspace/holosoma/wandb \
    apptainer/holosoma.sif bash -c "
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hssim && \
    export WANDB_INSECURE_DISABLE_SSL=true && \
    export MUJOCO_GL=egl && \
    cd /workspace/holosoma && \
    python src/holosoma/holosoma/train_agent.py \
        exp:g1-29dof-fast-sac \
        simulator:mjwarp \
        logger:wandb \
        --logger.video.enabled=False
"
```

### WBT IsaacSim

```bash
apptainer exec --nv \
    --bind /run \
    --bind /dev/shm \
    --bind $SCRATCH/.isaac_cache:/root/.local/share/ov \
    --bind $SCRATCH/holosoma_logs:/workspace/holosoma/logs \
    --bind $SCRATCH/wandb_cache:/workspace/holosoma/wandb \
    --bind $SCRATCH/isaac_tmp:/tmp \
    --bind $SCRATCH/isaac_pkg_cache:/root/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/isaacsim/kit/cache \
    --bind $SCRATCH/isaac_pkg_data:/root/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/isaacsim/kit/data \
    --bind $SCRATCH/isaac_pkg_logs:/root/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/isaacsim/kit/logs \
    --bind $SCRATCH/holosoma_converted_robots:/workspace/holosoma/src/holosoma/holosoma/data/robots/converted_rank0 \
    apptainer/holosoma.sif bash -c "
    export HTTP_PROXY=\$http_proxy && \
    export HTTPS_PROXY=\$https_proxy && \
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt && \
    export OMNI_KIT_ALLOW_ROOT=1 && \
    export TMPDIR=/tmp && \
    export TEMP=/tmp && \
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hssim && \
    export WANDB_INSECURE_DISABLE_SSL=true && \
    cd /workspace/holosoma && \
    xvfb-run -a python src/holosoma/holosoma/train_agent.py \
        exp:g1-29dof-wbt-isaacsim \
        logger:wandb \
        --logger.video.enabled=False \
        --command.setup-terms.motion-command.params.motion-config.motion-file=\"holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz\"
"
```

### WBT IsaacSim + object

```bash
apptainer exec --nv \
    --bind /run \
    --bind /dev/shm \
    --bind /home/gbesset/WILLOW/holosoma/src:/workspace/holosoma/src \
    --bind $SCRATCH/.isaac_cache:/root/.local/share/ov \
    --bind $SCRATCH/holosoma_logs:/workspace/holosoma/logs \
    --bind $SCRATCH/wandb_cache:/workspace/holosoma/wandb \
    --bind $SCRATCH/isaac_tmp:/tmp \
    --bind $SCRATCH/isaac_pkg_cache:/root/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/isaacsim/kit/cache \
    --bind $SCRATCH/isaac_pkg_data:/root/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/isaacsim/kit/data \
    --bind $SCRATCH/isaac_pkg_logs:/root/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/isaacsim/kit/logs \
    --bind $SCRATCH/holosoma_converted_robots:/workspace/holosoma/src/holosoma/holosoma/data/robots/converted_rank0 \
    apptainer/holosoma.sif bash -c "
    export HTTP_PROXY=\$http_proxy && \
    export HTTPS_PROXY=\$https_proxy && \
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt && \
    export OMNI_KIT_ALLOW_ROOT=1 && \
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hssim && \
    cd /workspace/holosoma && \
    xvfb-run -a python src/holosoma/holosoma/train_agent.py \
        exp:g1-29dof-wbt-isaacsim-w-object \
        logger:wandb \
        --logger.video.enabled=False \
        --training.num-envs=512 \
        --command.setup-terms.motion-command.params.motion-config.motion-file=\"holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz\" \
        --simulator.config.scene.rigid-objects.0.name=\"largebox\" \
        --simulator.config.scene.rigid-objects.0.urdf-path=\"src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf\" \
        --simulator.config.scene.rigid-objects.0.position=\"[1.0,0.0,1.0]\"
"
```

> The object position offset is a safety margin to avoid explosive forces at simulation reset. The actual position is randomized immediately after reset.

### WBT MJWarp

```bash
apptainer exec --nv --writable-tmpfs \
    --bind /run \
    --bind /dev/shm \
    --bind $SCRATCH/holosoma_logs:/workspace/holosoma/logs \
    --bind $SCRATCH/wandb_cache:/workspace/holosoma/wandb \
    --bind /home/gbesset/WILLOW/holosoma/src:/workspace/holosoma/src \
    /home/gbesset/WILLOW/holosoma/apptainer/holosoma.sif bash -c "
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hssim && \
    export MUJOCO_GL=egl && \
    export WANDB_INSECURE_DISABLE_SSL=true && \
    cd /workspace/holosoma && \
    xvfb-run -a python src/holosoma/holosoma/train_agent.py \
        exp:g1-29dof-wbt-mjwarp \
        logger:wandb \
        --logger.video.enabled=False \
        --training.num-envs=512 \
        --command.setup-terms.motion-command.params.motion-config.motion-file=\"holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz\"
"
```

### WBT MJWarp multi-GPU

```bash
apptainer exec --nv --writable-tmpfs \
    --bind /run \
    --bind /dev/shm \
    --bind $SCRATCH/holosoma_logs:/workspace/holosoma/logs \
    --bind $SCRATCH/wandb_cache:/workspace/holosoma/wandb \
    --bind /home/gbesset/WILLOW/holosoma/src:/workspace/holosoma/src \
    /home/gbesset/WILLOW/holosoma/apptainer/holosoma.sif bash -c "
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hssim && \
    export MUJOCO_GL=egl && \
    export WANDB_INSECURE_DISABLE_SSL=true && \
    cd /workspace/holosoma && \
    xvfb-run -a torchrun --nproc_per_node=2 src/holosoma/holosoma/train_agent.py \
        exp:g1-29dof-wbt-mjwarp \
        logger:wandb \
        --logger.video.enabled=False \
        --training.multigpu=True \
        --training.num-envs=8192 \
        --reward.terms.undesired_contacts.weight=-0.1 \
        --command.setup-terms.motion-command.params.motion-config.motion-file=\"holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz\"
"
```

### WBT MJWarp + object

```bash
apptainer exec --nv --writable-tmpfs \
    --bind /run \
    --bind /dev/shm \
    --bind $SCRATCH/holosoma_logs:/workspace/holosoma/logs \
    --bind $SCRATCH/wandb_cache:/workspace/holosoma/wandb \
    --bind /home/gbesset/WILLOW/holosoma/src:/workspace/holosoma/src \
    /home/gbesset/WILLOW/holosoma/apptainer/holosoma.sif bash -c "
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hssim && \
    export MUJOCO_GL=egl && \
    export WANDB_INSECURE_DISABLE_SSL=true && \
    cd /workspace/holosoma && \
    xvfb-run -a python src/holosoma/holosoma/train_agent.py \
        exp:g1-29dof-wbt-mjwarp-w-object \
        logger:wandb \
        terrain:terrain-locomotion-plane \
        --logger.video.enabled=False \
        --training.num-envs=8192 \
        --command.setup-terms.motion-command.params.motion-config.motion-file=\"holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz\" \
        --simulator.config.scene.rigid-objects.0.name=\"largebox\" \
        --simulator.config.scene.rigid-objects.0.urdf-path=\"src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf\" \
        --simulator.config.scene.rigid-objects.0.position=\"[1.0,0.0,1.0]\" \
        --terrain.terrain-term.mesh-type=\"PLANE\" \
        --terrain.terrain-term.obj-file-path=\"\" \
        --simulator.config.mujoco-warp.ccd-iterations=200
"
```

---

## 4. Inference

### Via native SDK (Unitree / Booster)

The default path uses the robot's native SDK directly (DDS for Unitree, SDK2PY for Booster).

```bash
python src/holosoma_inference/holosoma_inference/run_policy.py \
    inference:g1-29dof-loco \
    --task.model-path path/to/model.onnx
```

### Via ROS2 (real robot)

Set `--robot.sdk-type=ros2` to route commands and state through ROS2 topics instead of the native SDK. This requires `rclpy` and a running ROS2 environment.

```bash
python src/holosoma_inference/holosoma_inference/run_policy.py \
    inference:g1-29dof-loco \
    --robot.sdk-type=ros2 \
    --task.model-path path/to/model.onnx
```

A bridge node on the robot side is needed to relay these topics to/from the native SDK.

### Sim2Sim (local policy + MuJoCo)

Run the policy against a local MuJoCo simulation via ROS2. This requires two terminals — one for the simulator with the ROS2 bridge enabled, and one for the policy.

**Terminal 1 — Simulator (env `hsmujoco`):**

```bash
conda activate hsmujoco
source /opt/ros/humble/setup.bash
python src/holosoma/holosoma/run_sim.py \
    simulator:mujoco \
    robot:g1-29dof \
    terrain:terrain-locomotion-plane \
    --robot.bridge.sdk-type=ros2 \
    --simulator.config.bridge.enabled=True \
    --simulator.config.bridge.interface=lo
```

**Terminal 2 — Policy (env `hsinference`):**

```bash
conda activate hsinference
source /opt/ros/humble/setup.bash
python src/holosoma_inference/holosoma_inference/run_policy.py \
    inference:g1-29dof-wbt \
    --robot.sdk-type=ros2 \
    --task.model-path wandb://guibsst-inria/WholeBodyTracking/dcpxirww/model_16000.onnx \
    --task.no-use-joystick \
    --task.use-sim-time \
    --task.rl-rate 50 \
    --task.interface lo
```

Both sides communicate over the following ROS2 topics:

| Direction | Topic | Type | Content |
|-----------|-------|------|---------|
| Sim → Policy | `/holosoma/low_state` | `sensor_msgs/JointState` | Current joint positions, velocities, efforts |
| Sim → Policy | `/holosoma/imu` | `sensor_msgs/Imu` | Orientation quaternion and angular velocity |
| Policy → Sim | `/holosoma/low_cmd` | `sensor_msgs/JointState` | Target positions, velocities, feedforward torques |
| Policy → Sim | `/holosoma/pd_gains` | `sensor_msgs/JointState` | PD gains (KP in position, KD in velocity) |

> Both processes must have access to `rclpy`. If running outside an Apptainer container, make sure to `source /opt/ros/<distro>/setup.bash` before launching each terminal.
