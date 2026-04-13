# Holosoma — Installation Guide

All commands are run from the `holosoma/` directory unless otherwise specified.

## What to install depending on your use case

| Use case | Environment | Installation |
|---|---|---|
| Retargeting, data conversion | `hsretargeting` | [Local conda](#local-conda-all-use-cases) |
| Policy evaluation — MJWarp | `hsmujoco` | [Local conda](#local-conda-all-use-cases) |
| Policy evaluation — IsaacSim | `hssim` | [Local conda](#local-conda-all-use-cases) |
| Inference on real robot | `hsinference` | [Local conda](#local-conda-all-use-cases) |
| Sim-to-sim (MuJoCo) | `hsmujoco` | [Local conda](#local-conda-all-use-cases) |
| Sim-to-sim (PyBullet) | `hsinference` + `unitree_control_interface` | [Local conda](#local-conda-all-use-cases) + [PyBullet bridge](#pybullet-bridge-sim-to-sim) |
| Training on a cluster (SLURM) | Apptainer container | [Apptainer (cluster)](#apptainer-cluster) |
| Training on a local machine | `hssim` or Apptainer | [Local conda](#local-conda-all-use-cases) or [Apptainer (local)](#apptainer-local-machine) |

---

## Local conda (all use cases)

The individual `scripts/setup_*.sh` may not work out of the box depending on your system.
Use `deploy/local/setup_holosoma.sh` instead, which installs everything in one shot without requiring `sudo`:

```bash
# No sudo required — all dependencies managed via conda
bash deploy/local/setup_holosoma.sh
```

This installs miniconda at `~/.holosoma_deps/miniconda3/` and creates all conda environments:
`hssim`, `hsinference`, `hsmujoco`, `hsretargeting`.

Then activate the relevant environment before running:

```bash
source scripts/source_inference_setup.sh    # activates hsinference
source scripts/source_mujoco_setup.sh       # activates hsmujoco
source scripts/source_retargeting_setup.sh  # activates hsretargeting
```

After installation, use the deploy quick references below, and consult module docs for full details:

- [TRAINING.md](TRAINING.md) — training quick reference and cluster commands
- [EVALUATION.md](EVALUATION.md) — evaluation and ONNX export quick reference
- [INFERENCE.md](INFERENCE.md) — inference quick reference
- [RETARGETING.md](RETARGETING.md) — retargeting quick reference
- [src/holosoma/README.md](../src/holosoma/README.md) — full training and evaluation documentation
- [src/holosoma_inference/README.md](../src/holosoma_inference/README.md) — full inference documentation
- [src/holosoma_retargeting/README.md](../src/holosoma_retargeting/README.md) — full retargeting documentation
- [src/holosoma_data/README.md](../src/holosoma_data/README.md) — full dataset preparation documentation

> **Important:** The `deploy/` docs are meant for quick operational use only.
> For exact config syntax, parameter meanings, and workflow details, consult the
> corresponding `src/` module README.

### PyBullet bridge (sim-to-sim)

The PyBullet sim-to-sim workflow requires a separate ROS2 workspace:
[inria-paris-robotics-lab/unitree_control_interface](https://github.com/inria-paris-robotics-lab/unitree_control_interface).

Follow the installation instructions in that repository. The workspace ends up at
`unitree_ros2/cyclonedds_ws/` alongside `holosoma/`.

---

## Apptainer (cluster)

Intended for SLURM training jobs. Provides a fully reproducible environment across nodes without requiring `sudo` or a specific host OS.

### Requirements

* Linux OS (Ubuntu 22.04+)
* 128 GB RAM (for building)
* NVIDIA GPU with CUDA support
* 150 GB disk space

### Build on a cluster (HPC)

`$SCRATCH` and `$TMP_DIR` are typically set automatically after login. Verify with `echo $SCRATCH`.

From `holosoma/`:

```bash
export APPTAINER_CACHEDIR=$SCRATCH/.apptainer_cache
mkdir -p $APPTAINER_CACHEDIR
export APPTAINER_TMPDIR=$TMP_DIR/.apptainer_tmp
mkdir -p $APPTAINER_TMPDIR
apptainer build --fakeroot --force deploy/cluster/holosoma.sif deploy/cluster/setup_apptainer.def
```

### Apptainer (local machine)

From `holosoma/`:

```bash
export APPTAINER_CACHEDIR=$HOME/.cache/apptainer
mkdir -p $APPTAINER_CACHEDIR
export APPTAINER_TMPDIR=/tmp/apptainer_tmp
mkdir -p $APPTAINER_TMPDIR
apptainer build --fakeroot --force deploy/cluster/holosoma.sif deploy/cluster/setup_apptainer.def
```

Accepts the NVIDIA Isaac Sim EULA. Build time: ~30–60 minutes.

### Run the container

```bash
# Interactive shell
apptainer exec --nv --bind /run deploy/cluster/holosoma.sif bash

# Run a Python script
apptainer exec --nv --bind /run deploy/cluster/holosoma.sif python my_script.py

# With home directory access
apptainer exec --nv --bind /run --bind $HOME:/home/$USER deploy/cluster/holosoma.sif bash
```

### Contents of the image

* Isaac Sim 5.1.0 + Isaac Lab
* MuJoCo 3.5.0
* Holosoma + tools
