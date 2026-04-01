# Holosoma — Training

All commands are run from the `holosoma/` directory.
Requires the Apptainer image (`deploy/cluster/holosoma.sif`) — see [INSTALL.md](INSTALL.md).

For SLURM jobs, use the pre-configured scripts in `deploy/cluster/slurm/`.

---

## G1 variants: 27-DOF (base) vs 29-DOF (pro)

The G1 robot comes in two variants:

| Variant | DOF | Difference |
|---------|-----|------------|
| **G1 base** (`g1_27dof`) | 27 | `waist_roll` and `waist_pitch` joints are locked |
| **G1 pro** (`g1_29dof`) | 29 | All waist joints are actuated |

All training commands below default to the **29-DOF pro** variant.
To train with the **27-DOF base** variant instead, replace `g1-29dof` by `g1-27dof` in the experiment name:

```
exp:g1-29dof-wbt-isaacsim   →   exp:g1-27dof-wbt-isaacsim
exp:g1-29dof-wbt-mjwarp     →   exp:g1-27dof-wbt-mjwarp
exp:g1-29dof-wbt-fast-sac   →   exp:g1-27dof-wbt-fast-sac
```

The 27-DOF configs reuse the same motion data, reward, observation, and randomization
settings — only the robot model (URDF/XML) and joint configuration differ.

---

## Locomotion (MJWarp)

```bash
apptainer exec --nv --writable-tmpfs \
    --bind /run \
    --bind /dev/shm \
    --bind $SCRATCH/holosoma_logs:/workspace/holosoma/logs \
    --bind $SCRATCH/wandb_cache:/workspace/holosoma/wandb \
    deploy/cluster/holosoma.sif bash -c "
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

---

## Whole-Body Tracking (WBT)

### WBT — IsaacSim

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
    deploy/cluster/holosoma.sif bash -c "
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

### WBT — IsaacSim + object

```bash
apptainer exec --nv \
    --bind /run \
    --bind /dev/shm \
    --bind $(pwd)/src:/workspace/holosoma/src \
    --bind $SCRATCH/.isaac_cache:/root/.local/share/ov \
    --bind $SCRATCH/holosoma_logs:/workspace/holosoma/logs \
    --bind $SCRATCH/wandb_cache:/workspace/holosoma/wandb \
    --bind $SCRATCH/isaac_tmp:/tmp \
    --bind $SCRATCH/isaac_pkg_cache:/root/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/isaacsim/kit/cache \
    --bind $SCRATCH/isaac_pkg_data:/root/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/isaacsim/kit/data \
    --bind $SCRATCH/isaac_pkg_logs:/root/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/isaacsim/kit/logs \
    --bind $SCRATCH/holosoma_converted_robots:/workspace/holosoma/src/holosoma/holosoma/data/robots/converted_rank0 \
    deploy/cluster/holosoma.sif bash -c "
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

### WBT — MJWarp

```bash
apptainer exec --nv --writable-tmpfs \
    --bind /run \
    --bind /dev/shm \
    --bind $SCRATCH/holosoma_logs:/workspace/holosoma/logs \
    --bind $SCRATCH/wandb_cache:/workspace/holosoma/wandb \
    --bind $(pwd)/src:/workspace/holosoma/src \
    deploy/cluster/holosoma.sif bash -c "
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

### WBT — MJWarp multi-GPU

```bash
apptainer exec --nv --writable-tmpfs \
    --bind /run \
    --bind /dev/shm \
    --bind $SCRATCH/holosoma_logs:/workspace/holosoma/logs \
    --bind $SCRATCH/wandb_cache:/workspace/holosoma/wandb \
    --bind $(pwd)/src:/workspace/holosoma/src \
    deploy/cluster/holosoma.sif bash -c "
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

### WBT — MJWarp + object

```bash
apptainer exec --nv --writable-tmpfs \
    --bind /run \
    --bind /dev/shm \
    --bind $SCRATCH/holosoma_logs:/workspace/holosoma/logs \
    --bind $SCRATCH/wandb_cache:/workspace/holosoma/wandb \
    --bind $(pwd)/src:/workspace/holosoma/src \
    deploy/cluster/holosoma.sif bash -c "
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

## Next step

To evaluate a trained checkpoint and export it to ONNX, see [EVALUATION.md](EVALUATION.md).