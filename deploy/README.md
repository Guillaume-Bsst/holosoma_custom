# Holosoma — deploy/

This folder contains everything needed to run Holosoma: environment setup, container definition,
SLURM job scripts, and usage guides.

## Documentation

| File | Contents |
|---|---|
| [INSTALL.md](INSTALL.md) | What to install depending on your use case (local conda or Apptainer) |
| [RETARGETING.md](RETARGETING.md) | How to run retargeting and data conversion (27-DOF / 29-DOF) |
| [TRAINING.md](TRAINING.md) | How to run training jobs (MJWarp, IsaacSim, multi-GPU, with objects) |
| [EVALUATION.md](EVALUATION.md) | How to evaluate a trained checkpoint and export to ONNX |
| [INFERENCE.md](INFERENCE.md) | How to run inference on the real robot or in simulation (MuJoCo, PyBullet) |

## Folder structure

```
deploy/
├── README.md                  ← this file
├── INSTALL.md                 ← installation guide
├── RETARGETING.md             ← retargeting & data conversion guide
├── TRAINING.md                ← training guide
├── EVALUATION.md              ← policy evaluation & ONNX export guide
├── INFERENCE.md               ← inference & sim-to-sim guide
├── local/
│   ├── setup_holosoma.sh      ← full local install (no sudo)
│   ├── setup_isaacsim.sh
│   ├── setup_isaacgym.sh
│   ├── setup_mujoco.sh
│   ├── setup_inference.sh
│   ├── setup_retargeting.sh
│   └── source_common.sh
└── cluster/
    ├── setup_apptainer.def    ← Apptainer container definition
    ├── holosoma.sif           ← built container image (not versioned)
    └── slurm/
        ├── train_locomotion_mujoco.slurm
        ├── train_wbt_isaacsim.slurm
        ├── train_wbt_isaacsim_w_obj.slurm
        ├── train_wbt_mujoco.slurm
        └── train_wbt_mujoco_w_obj.slurm
```

## Quick start

**Local use (inference, sim-to-sim, retargeting):**

```bash
bash deploy/local/setup_holosoma.sh
```

**Cluster training (SLURM):** build the Apptainer image first — see [INSTALL.md](INSTALL.md),
then submit a job from `deploy/cluster/slurm/`.
