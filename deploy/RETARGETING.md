# Holosoma — Retargeting & Data Conversion (quick reference)

This page is a concise reference for retargeting and data conversion commands.
For the full retargeting documentation, see [src/holosoma_retargeting/README.md](../src/holosoma_retargeting/README.md).
For dataset preparation, see [src/holosoma_data/README.md](../src/holosoma_data/README.md).

All commands are run from `src/holosoma_retargeting/holosoma_retargeting/` after sourcing the retargeting environment from `holosoma/`:

```bash
source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting
```

## DOF mode (27 / 29)

The retargeting and data conversion pipelines support both **29-DOF** (default) and **27-DOF** modes.
Add `--robot-config.robot-dof 27` or `29` to any retargeting or conversion command.

## 1. Retargeting

### Robot-only (OMOMO)

```bash
python examples/robot_retarget.py \
    --data_path holosoma_data/datasets/omomo \
    --task-type robot_only \
    --task-name sub3_largebox_003 \
    --data_format smplx \
    --retargeter.debug \
    --retargeter.visualize
```

### Object interaction (OMOMO)

```bash
python examples/robot_retarget.py \
    --data_path holosoma_data/datasets/omomo \
    --task-type object_interaction \
    --task-name sub3_largebox_003 \
    --data_format smplx \
    --retargeter.debug \
    --retargeter.visualize
```

### Robot-only (SFU / AMASS)

```bash
python examples/robot_retarget.py \
    --data_path holosoma_data/datasets/sfu \
    --task-type robot_only \
    --task-name SFU_0005_0005_2FeetJump001_stageii \
    --data_format smplx \
    --task-config.ground-range -10 10 \
    --save_dir holosoma_data/pipeline/retargeted/g1/robot_only/sfu \
    --retargeter.debug \
    --retargeter.visualize
```

### Robot-only (LAFAN)

```bash
python examples/robot_retarget.py \
    --data_path holosoma_data/datasets/lafan \
    --task-type robot_only \
    --task-name dance2_subject1 \
    --data_format lafan \
    --task-config.ground-range -10 10 \
    --save_dir holosoma_data/pipeline/retargeted/g1/robot_only/lafan \
    --retargeter.debug \
    --retargeter.visualize \
    --retargeter.foot-sticking-tolerance 0.02
```

### Climbing

```bash
python examples/robot_retarget.py \
    --data_path holosoma_data/datasets/climb \
    --task-type climbing \
    --task-name mocap_climb_seq_0 \
    --data_format mocap \
    --robot-config.robot-urdf-file holosoma_data/robots/g1/g1_29dof_spherehand_retargeting.urdf \
    --retargeter.debug \
    --retargeter.visualize
```

## 2. Batch retargeting

```bash
python examples/parallel_robot_retarget.py \
    --data-dir holosoma_data/datasets/omomo \
    --task-type robot_only \
    --data_format smplx \
    --save_dir holosoma_data/pipeline/retargeted/g1/robot_only/omomo \
    --task-config.object-name ground
```

## 3. Data conversion

### Robot-only

```bash
python data_conversion/convert_data_format_mj.py \
    --input_file holosoma_data/pipeline/retargeted/g1/robot_only/omomo/sub3_largebox_003_original.npz \
    --output_fps 50 \
    --output_name holosoma_data/pipeline/converted/robot_only/sub3_largebox_003_mj_fps50.npz \
    --data_format smplx \
    --object_name ground \
    --once
```

## Notes

- This file is a quick reference. Use [src/holosoma_retargeting/README.md](../src/holosoma_retargeting/README.md) for the full retargeting pipeline.
- Use [src/holosoma_data/README.md](../src/holosoma_data/README.md) for dataset preparation and format details.
