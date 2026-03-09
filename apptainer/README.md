# Holosoma Apptainer Container Installation

Containerized environment for Holosoma full pipeline with IsaacSim & MuJoCo

## Requirements

* Linux OS (Ubuntu 22.04+)
* 128 GB RAM (for building)
* NVIDIA GPU with CUDA support
* 150 GB disk space

## Building

From `holosoma-main/` directory:

```bash
export APPTAINER_CACHEDIR=$SCRATCH/.apptainer_cache
mkdir -p $APPTAINER_CACHEDIR
export APPTAINER_TMPDIR=$TMP_DIR/.apptainer_tmp
mkdir -p $APPTAINER_TMPDIR
apptainer build --fakeroot --force apptainer/holosoma.sif apptainer/setup_apptainer.def

```

This accepts the NVIDIA Isaac Sim EULA. Build time: ~30-60 minutes

## Running

```bash
# Interactive shell
apptainer exec --nv --bind /run apptainer/holosoma.sif bash

# Run Python script
apptainer exec --nv --bind /run apptainer/holosoma.sif python my_script.py

# With home directory access
apptainer exec --nv --bind /run --bind $HOME:/home/$USER apptainer/holosoma.sif bash

```

Other example usages in EXAMPLE.md

## Installed

* Isaac Sim 5.1.0 + Isaac Lab
* MuJoCo 3.5.0
* Holosoma + tools

---

## Important: Required Source Patches for MuJoCo (MJWarp)

Holosoma is heavily optimized for IsaacSim by default. To use MJWarp via the CLI and train policies successfully with advanced features (like Whole Body Tracking and Domain Randomization), you must patch the local source code.

**Note:** You MUST use the `--bind` command when running the container (e.g., `--bind src:/workspace/holosoma/src`) to inject this patched code at runtime.

### 1. Tyro CLI Parsing Fix (`src/holosoma/holosoma/config_types/simulator.py`)

Fixes an initialization issue with Tyro configuration parsing.

* Locate the `SceneConfig` class.
* Update `scene_files` and `rigid_objects` to use a `default_factory`:
```python
scene_files: list[SceneFileConfig] = field(default_factory=list)
rigid_objects: list[RigidObjectConfig] = field(default_factory=list)

```

### 2. Forward Kinematics for Default Pose (`src/holosoma/holosoma/managers/command/terms/wbt.py`)

Fixes the `_capture_body_states` method which natively relies on IsaacSim's GPU state writes, causing crashes in MuJoCo.

* Added a `SimulatorType.MUJOCO` branch.
* Implemented a static body mapping cache (`_mj_body_ids`) to avoid redundant name lookups.
* Used the `mujoco.MjData` and `mujoco.mj_kinematics` API to compute forward kinematics purely on the CPU model (`simulator.backend.model`).
* Utilized vectorized NumPy slicing to extract positions, quaternions (with `wxyz` to `xyzw` conversion), and velocities efficiently to PyTorch tensors.

### 3. Domain Randomization Compatibility (`src/holosoma/holosoma/managers/randomization/terms/locomotion.py`)

Bypasses IsaacSim-specific physics material constraints (e.g., `RandomizerNotSupportedError`) to allow friction randomization in MuJoCo without using `--randomization.ignore-unsupported=True`.

* In `randomize_robot_rigid_body_material_startup`, added an early exit branch for MuJoCo.
* Cached target collision geometries specifically for the robot (`geom_names.startswith("robot_")`) to avoid altering the floor's friction.
* Applied the `dynamic_friction_range` directly to `mj_model.geom_friction[..., 0]` (sliding friction).
* Forced the physics engine to recompute its internal cache using `mujoco.mj_setConst(mj_model, ...)`.