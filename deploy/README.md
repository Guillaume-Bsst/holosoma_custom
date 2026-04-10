# Holosoma — deploy/

This folder contains everything needed to install and run Holosoma: environment setup, container definition, SLURM job scripts, and operational guides.

## Documentation

| File | Contents |
|---|---|
| [INSTALL.md](INSTALL.md) | Environment setup — local conda or Apptainer |
| [RETARGETING.md](RETARGETING.md) | Retargeting and data conversion commands |

For training, evaluation, and inference guides, see the per-module READMEs:

| Module | Guide |
|---|---|
| Training | [src/holosoma/README.md](../src/holosoma/README.md) |
| Inference & deployment | [src/holosoma_inference/README.md](../src/holosoma_inference/README.md) |
| Retargeting (full reference) | [src/holosoma_retargeting/holosoma_retargeting/README.md](../src/holosoma_retargeting/holosoma_retargeting/README.md) |
| Dataset preparation | [src/holosoma_data/README.md](../src/holosoma_data/README.md) |

## Folder structure

```
deploy/
├── README.md                  ← this file
├── INSTALL.md                 ← installation guide
├── RETARGETING.md             ← retargeting & data conversion quick reference
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
