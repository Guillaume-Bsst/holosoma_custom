# Holosoma — Policy Evaluation

All commands are run from the `holosoma/` directory.
See [INSTALL.md](INSTALL.md) for environment setup.

Evaluation sits between training and inference: it loads a trained checkpoint, runs the policy
in the same simulator used during training, and optionally exports it to ONNX for deployment.

---

## Activation

The environment to activate depends on the simulator used during training:

| Simulator | Environment | Activation |
|-----------|-------------|------------|
| MJWarp | `hsmujoco` | `source scripts/source_mujoco_setup.sh` |
| IsaacSim | `hssim` | `source scripts/source_isaacsim_setup.sh` |

---

## Basic usage

```bash
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint path/to/model_5000.pt
```

The experiment config is loaded automatically from the checkpoint. No need to re-specify
`exp:` or `simulator:`.

---

## Checkpoint sources

### Local checkpoint

```bash
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint logs/my_run/model_5000.pt
```

### W&B checkpoint (URI format)

```bash
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint wandb://<entity>/<project>/<run_id>/model_5000.pt
```

If no checkpoint name is given, the latest checkpoint from the run is used:

```bash
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint wandb://<entity>/<project>/<run_id>
```

---

## ONNX export

Pass `--training.export-onnx=True` to export the policy as an ONNX file during evaluation.
The file is saved alongside the checkpoint in an `exported/` subdirectory.

```bash
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint wandb://<entity>/<project>/<run_id>/model_5000.pt \
    --training.export-onnx=True
```

The exported file (`model_5000.onnx`) can then be used directly with `run_policy.py` — see [INFERENCE.md](INFERENCE.md).

---

## Config overrides

Any training config key can be overridden on top of the loaded checkpoint config:

```bash
# Reduce number of environments for faster eval
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint wandb://<entity>/<project>/<run_id>/model_5000.pt \
    --training.num-envs=64

# Cap the number of evaluation steps
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint wandb://<entity>/<project>/<run_id>/model_5000.pt \
    --training.max-eval-steps=500
```

---

## Locomotion (MJWarp)

```bash
source scripts/source_mujoco_setup.sh
export MUJOCO_GL=egl
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint wandb://<entity>/<project>/<run_id>/model_5000.pt \
    --training.export-onnx=True
```

---

## Whole-Body Tracking (WBT)

### WBT — MJWarp

```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint wandb://<entity>/<project>/<run_id>/model_5000.pt \
    --training.export-onnx=True
```

### WBT — IsaacSim

```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint wandb://<entity>/<project>/<run_id>/model_5000.pt \
    --training.export-onnx=True
```

---

## On a cluster (Apptainer)

### MJWarp

```bash
apptainer exec --nv --writable-tmpfs \
    --bind /run \
    --bind /dev/shm \
    --bind $SCRATCH/holosoma_logs:/workspace/holosoma/logs \
    --bind $SCRATCH/wandb_cache:/workspace/holosoma/wandb \
    deploy/cluster/holosoma.sif bash -c "
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hsmujoco && \
    export MUJOCO_GL=egl && \
    export WANDB_INSECURE_DISABLE_SSL=true && \
    cd /workspace/holosoma && \
    python src/holosoma/holosoma/eval_agent.py \
        --checkpoint wandb://<entity>/<project>/<run_id>/model_5000.pt \
        --training.export-onnx=True
"
```

### IsaacSim

```bash
apptainer exec --nv \
    --bind /run \
    --bind /dev/shm \
    --bind $SCRATCH/.isaac_cache:/root/.local/share/ov \
    --bind $SCRATCH/holosoma_logs:/workspace/holosoma/logs \
    --bind $SCRATCH/wandb_cache:/workspace/holosoma/wandb \
    --bind $SCRATCH/isaac_tmp:/tmp \
    deploy/cluster/holosoma.sif bash -c "
    export OMNI_KIT_ALLOW_ROOT=1 && \
    export TMPDIR=/tmp && \
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hssim && \
    export WANDB_INSECURE_DISABLE_SSL=true && \
    cd /workspace/holosoma && \
    xvfb-run -a python src/holosoma/holosoma/eval_agent.py \
        --checkpoint wandb://<entity>/<project>/<run_id>/model_5000.pt \
        --training.export-onnx=True
"
```

---

## What happens during evaluation

1. The checkpoint config (`holosoma_config.yaml`) is loaded from the `.pt` file or from W&B.
2. Additional CLI overrides are applied on top.
3. The simulator is initialized with the same settings as training.
4. If `--training.export-onnx=True`, the policy is exported to `<checkpoint_dir>/exported/model_<step>.onnx`.
5. `evaluate_policy()` runs the policy in the simulator for `--training.max-eval-steps` steps.

The ONNX file embeds the control gains (kp/kd) and is ready to pass to `run_policy.py`.

---

## Next step

Once you have an ONNX model, deploy it — see [INFERENCE.md](INFERENCE.md).
