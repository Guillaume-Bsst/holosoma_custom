# Holosoma — deploy/

This folder contains installation and runtime quick-reference guides for Holosoma.
It is not the primary user manual: the source-module readmes hold the full details.

## Documentation

| File | Contents |
|---|---|
| [INSTALL.md](INSTALL.md) | Environment setup — local conda or Apptainer |
| [TRAINING.md](TRAINING.md) | Training quick reference and cluster commands |
| [EVALUATION.md](EVALUATION.md) | Checkpoint evaluation and ONNX export |
| [INFERENCE.md](INFERENCE.md) | Real robot and sim-to-sim inference quick reference |
| [RETARGETING.md](RETARGETING.md) | Retargeting and data conversion quick reference |

For the full module documentation, see:

| Module | Guide |
|---|---|
| Training | [src/holosoma/README.md](../src/holosoma/README.md) |
| Inference & deployment | [src/holosoma_inference/README.md](../src/holosoma_inference/README.md) |
| Retargeting | [src/holosoma_retargeting/README.md](../src/holosoma_retargeting/README.md) |
| Dataset preparation | [src/holosoma_data/README.md](../src/holosoma_data/README.md) |

## Folder structure

```
deploy/
├── README.md                  ← this file
├── INSTALL.md                 ← installation guide
├── TRAINING.md                ← training quick reference
├── EVALUATION.md              ← policy evaluation quick reference
├── INFERENCE.md               ← inference quick reference
├── RETARGETING.md             ← retargeting quick reference
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

**Cluster training (SLURM):** build the Apptainer image first — see [INSTALL.md](INSTALL.md), then submit a job from `deploy/cluster/slurm/`.
