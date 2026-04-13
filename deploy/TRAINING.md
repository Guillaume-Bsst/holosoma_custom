# Holosoma — Training (quick reference)

This quick reference is focused on training commands and deploy-side workflows.
For the full training documentation, configuration options, and experiment details, see [src/holosoma/README.md](../src/holosoma/README.md).

All commands are run from the `holosoma/` directory.

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Apptainer image | `deploy/cluster/holosoma.sif` |
| GPU | NVIDIA with CUDA support |
| Storage | `$SCRATCH` for logs, caches, and W&B data |

Create bind-mount directories before your first run:

```bash
mkdir -p $SCRATCH/{holosoma_logs,wandb_cache}
# IsaacSim only:
mkdir -p $SCRATCH/{.isaac_cache,isaac_tmp,isaac_pkg_cache,isaac_pkg_data,isaac_pkg_logs,holosoma_converted_robots}
```

## Experiment matrix

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

| Axis | Options |
|------|---------|
| **DOF** | `g1-27dof` / `g1-29dof` |
| **Simulator** | `isaacsim` / `mjwarp` |
| **Algorithm** | PPO / `fast-sac` |
| **Object** | without / `w-object` |

Naming convention: `exp:g1-{dof}dof-wbt-[fast-sac-]{sim}[-w-object]`

### Training examples

#### Locomotion (MJWarp)

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

#### WBT — MJWarp

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
        --training.num-envs=4096
"
```

#### WBT — IsaacSim

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
    --bind $SCRATCH/holosoma_converted_robots:/workspace/holosoma/src/holosoma_data/holosoma_data/robots/g1/converted_rank0 \
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
        --training.num-envs=4096
"
```

## Notes

- Use the full training docs at [src/holosoma/README.md](../src/holosoma/README.md) for task descriptions, reward weights, and motion file setup.
- Override any config on the CLI as needed with `--training.num-envs`, `--reward.*`, and `--command.*`.

## Next step

After training, evaluate and export ONNX models. See [EVALUATION.md](EVALUATION.md) for the deploy-side quick reference, or [src/holosoma/README.md](../src/holosoma/README.md) for more details.
