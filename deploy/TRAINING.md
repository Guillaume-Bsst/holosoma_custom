# Holosoma — Training

All commands are run from the `holosoma/` directory.
See [INSTALL.md](INSTALL.md) for environment setup and Apptainer image build.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Apptainer image | `deploy/cluster/holosoma.sif` (see [INSTALL.md](INSTALL.md)) |
| GPU | NVIDIA with CUDA support |
| Storage | `$SCRATCH` for logs, caches, and W&B data |

Before your first run, create the bind-mount directories:

```bash
mkdir -p $SCRATCH/{holosoma_logs,wandb_cache}
# IsaacSim only:
mkdir -p $SCRATCH/{.isaac_cache,isaac_tmp,isaac_pkg_cache,isaac_pkg_data,isaac_pkg_logs,holosoma_converted_robots}
```

---

## Experiments

### Locomotion

| Experiment | Algorithm | Simulator |
|------------|-----------|-----------|
| `exp:g1-29dof-isaacgym` | PPO | IsaacGym |
| `exp:g1-29dof-fast-sac-isaacgym` | Fast-SAC | IsaacGym |
| `exp:g1-29dof-mjwarp` | PPO | MJWarp |
| `exp:g1-29dof-fast-sac-mjwarp` | Fast-SAC | MJWarp |
| `exp:t1-29dof-isaacgym` | PPO | IsaacGym |
| `exp:t1-29dof-fast-sac-isaacgym` | Fast-SAC | IsaacGym |
| `exp:t1-29dof-mjwarp` | PPO | MJWarp |
| `exp:t1-29dof-fast-sac-mjwarp` | Fast-SAC | MJWarp |

### Whole-Body Tracking (WBT)

Every WBT experiment is a combination of 4 independent axes:

| Axis | Options |
|------|---------|
| **DOF** | `g1-27dof` (G1 base, waist_roll + waist_pitch locked) / `g1-29dof` (G1 pro) |
| **Simulator** | `isaacsim` / `mjwarp` |
| **Algorithm** | PPO (default, omitted) / `fast-sac` |
| **Object** | without (default, omitted) / `w-object` |

Naming convention: `exp:g1-{dof}dof-wbt-[fast-sac-]{sim}[-w-object]`

Full matrix (16 experiments):

| Experiment | DOF | Simulator | Algorithm | Object |
|------------|-----|-----------|-----------|--------|
| `exp:g1-29dof-wbt-isaacsim` | 29 | IsaacSim | PPO | |
| `exp:g1-29dof-wbt-isaacsim-w-object` | 29 | IsaacSim | PPO | yes |
| `exp:g1-29dof-wbt-fast-sac-isaacsim` | 29 | IsaacSim | Fast-SAC | |
| `exp:g1-29dof-wbt-fast-sac-isaacsim-w-object` | 29 | IsaacSim | Fast-SAC | yes |
| `exp:g1-29dof-wbt-mjwarp` | 29 | MJWarp | PPO | |
| `exp:g1-29dof-wbt-mjwarp-w-object` | 29 | MJWarp | PPO | yes |
| `exp:g1-29dof-wbt-fast-sac-mjwarp` | 29 | MJWarp | Fast-SAC | |
| `exp:g1-29dof-wbt-fast-sac-mjwarp-w-object` | 29 | MJWarp | Fast-SAC | yes |
| `exp:g1-27dof-wbt-isaacsim` | 27 | IsaacSim | PPO | |
| `exp:g1-27dof-wbt-isaacsim-w-object` | 27 | IsaacSim | PPO | yes |
| `exp:g1-27dof-wbt-fast-sac-isaacsim` | 27 | IsaacSim | Fast-SAC | |
| `exp:g1-27dof-wbt-fast-sac-isaacsim-w-object` | 27 | IsaacSim | Fast-SAC | yes |
| `exp:g1-27dof-wbt-mjwarp` | 27 | MJWarp | PPO | |
| `exp:g1-27dof-wbt-mjwarp-w-object` | 27 | MJWarp | PPO | yes |
| `exp:g1-27dof-wbt-fast-sac-mjwarp` | 27 | MJWarp | Fast-SAC | |
| `exp:g1-27dof-wbt-fast-sac-mjwarp-w-object` | 27 | MJWarp | Fast-SAC | yes |

Each experiment config is self-contained: simulator, robot model, motion file, object, and
algorithm are all pre-configured. Just pick an experiment name.

---

## Running training

For SLURM jobs, use the pre-configured scripts in `deploy/cluster/slurm/`.

### Locomotion (MJWarp)

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
        --logger.video.enabled=False
"
```

> Replace the `exp:` line with any MJWarp experiment from the matrix, e.g.
> `exp:g1-27dof-wbt-mjwarp`, `exp:g1-29dof-wbt-fast-sac-mjwarp-w-object`, etc.

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
        --training.num-envs=8192
"
```

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
        --logger.video.enabled=False
"
```

> Replace the `exp:` line with any IsaacSim experiment from the matrix, e.g.
> `exp:g1-27dof-wbt-isaacsim`, `exp:g1-29dof-wbt-fast-sac-isaacsim-w-object`, etc.

---

## Overriding defaults

All config values can be overridden on the CLI:

```bash
# Use a different motion file
--command.setup-terms.motion-command.params.motion-config.motion-file="path/to/motion.npz"

# Tweak a reward weight
--reward.terms.undesired_contacts.weight=-0.1

# Change number of environments
--training.num-envs=512
```

---

## Next step

To evaluate a trained checkpoint and export it to ONNX, see [EVALUATION.md](EVALUATION.md).
