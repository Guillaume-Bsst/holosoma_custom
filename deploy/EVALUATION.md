# Holosoma — Policy Evaluation (quick reference)

This page is a quick reference for evaluating checkpoints and exporting ONNX models.
For full details, see [src/holosoma/README.md](../src/holosoma/README.md).

All commands are run from the `holosoma/` directory.

## Environment activation

| Simulator | Environment | Activation |
|-----------|-------------|------------|
| MJWarp | `hsmujoco` | `source scripts/source_mujoco_setup.sh` |
| IsaacSim | `hssim` | `source scripts/source_isaacsim_setup.sh` |

## Basic usage

```bash
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint path/to/model_5000.pt
```

The experiment config is loaded from the checkpoint automatically.

## Export ONNX

```bash
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint wandb://<entity>/<project>/<run_id>/model_5000.pt \
    --training.export-onnx=True
```

The ONNX model is saved in `exported/` alongside the checkpoint.

## Cluster evaluation

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

## Notes

- Use [src/holosoma/README.md](../src/holosoma/README.md) for full evaluation and export options.
- When the checkpoint includes an embedded config, you do not need to specify `exp:` or `simulator:`.

## Next step

Deploy the ONNX model using [INFERENCE.md](INFERENCE.md) or the full inference docs in [src/holosoma_inference/README.md](../src/holosoma_inference/README.md).
