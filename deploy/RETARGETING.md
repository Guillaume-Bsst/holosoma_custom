# Holosoma — Retargeting & Data Conversion

All commands are run from `src/holosoma_retargeting/holosoma_retargeting/` after sourcing the retargeting environment from `holosoma/`:

```bash
source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting
```

For dataset preparation, see [src/holosoma_data/README.md](../src/holosoma_data/README.md).
For the full retargeting reference, see [src/holosoma_retargeting/holosoma_retargeting/README.md](../src/holosoma_retargeting/holosoma_retargeting/README.md).

---

## DOF mode (27 / 29)

The retargeting and data conversion pipelines support both **29-DOF** (default) and **27-DOF** modes.

To select the DOF mode, add `--robot-config.robot-dof 27` (or `29`) to any retargeting or
data-conversion command. This automatically adjusts:

- The robot URDF/XML model (`g1_27dof` vs `g1_29dof`)
- Joint limit bounds and cost weights
- Default output directory (e.g. `holosoma_data/pipeline/retargeted/g1_27dof/...`)

> **Important:** The DOF mode must be consistent across the whole pipeline —
> retargeting, data conversion, and training must all use the same DOF setting.

---

## 1. Retargeting

The pipeline supports three retargeting algorithms, selected via `--retargeter-method`:

| Method | Flag | Task types | Description |
| --- | --- | --- | --- |
| `omniretarget` | *(default)* | robot_only, object_interaction, climbing | Interaction Mesh + SQP |
| `gmr` | `--retargeter-method gmr` | robot_only | IK-based (mink/mujoco, requires GMR library) |
| `test` | `--retargeter-method test` | robot_only | Native mink IK (no GMR dependency) |

### Robot-only (OMOMO) — OmniRetarget

```bash
python examples/robot_retarget.py \
    --data_path holosoma_data/datasets/omomo \
    --task-type robot_only \
    --task-name sub3_largebox_003 \
    --data_format smplx \
    --retargeter.debug \
    --retargeter.visualize
```

### Object interaction (OMOMO) — OmniRetarget

```bash
python examples/robot_retarget.py \
    --data_path holosoma_data/datasets/omomo \
    --task-type object_interaction \
    --task-name sub3_largebox_003 \
    --data_format smplx \
    --retargeter.debug \
    --retargeter.visualize
```

### Robot-only (SFU / AMASS) — OmniRetarget

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

### Robot-only (SFU / AMASS) — GMR

Requires the GMR library (expected at `../GMR`, alongside the `holosoma/` folder).
Install once with `pip install -e ../GMR`.

```bash
python examples/robot_retarget.py \
    --retargeter-method gmr \
    --task-type robot_only \
    --data_path holosoma_data/datasets/sfu \
    --task-name SFU_0005_0005_2FeetJump001_stageii \
    --data_format smplx \
    --gmr.src_human smplx
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

### Climbing — OmniRetarget

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

### 27-DOF

Add `--robot-config.robot-dof 27` to any command above.

---

## 2. Batch Retargeting

```bash
# OMOMO robot-only
python examples/parallel_robot_retarget.py \
    --data-dir holosoma_data/datasets/omomo \
    --task-type robot_only \
    --data_format smplx \
    --save_dir holosoma_data/pipeline/retargeted/g1/robot_only/omomo \
    --task-config.object-name ground

# OMOMO object interaction
python examples/parallel_robot_retarget.py \
    --data-dir holosoma_data/datasets/omomo \
    --task-type object_interaction \
    --data_format smplx \
    --save_dir holosoma_data/pipeline/retargeted/g1/object_interaction/omomo \
    --task-config.object-name largebox

# LAFAN robot-only
python examples/parallel_robot_retarget.py \
    --data-dir holosoma_data/datasets/lafan \
    --task-type robot_only \
    --data_format lafan \
    --save_dir holosoma_data/pipeline/retargeted/g1/robot_only/lafan \
    --task-config.object-name ground \
    --task-config.ground-range -10 10 \
    --retargeter.foot-sticking-tolerance 0.02
```

---

## 3. Visualizing Results

```bash
# Robot-only (29-DOF)
python viser_player.py \
    --robot_urdf holosoma_data/robots/g1/g1_29dof_retargeting.urdf \
    --qpos_npz holosoma_data/pipeline/retargeted/g1/robot_only/omomo/sub3_largebox_003_original.npz

# Object interaction (29-DOF)
python viser_player.py \
    --robot_urdf holosoma_data/robots/g1/g1_29dof_retargeting.urdf \
    --object_urdf holosoma_data/objects/largebox/largebox.urdf \
    --qpos_npz holosoma_data/pipeline/retargeted/g1/object_interaction/omomo/sub3_largebox_003_original.npz

# 27-DOF
python viser_player.py \
    --robot_urdf holosoma_data/robots/g1/g1_27dof_retargeting.urdf \
    --qpos_npz holosoma_data/pipeline/retargeted/g1/robot_only/omomo/sub3_largebox_003_original.npz
```

---

## 4. Data Conversion

Converts retargeted `.npz` files to the format required for RL training.

### Robot-only

```bash
python data_conversion/convert_data_format_mj.py \
    --input_file holosoma_data/pipeline/retargeted/g1/robot_only/omomo/sub3_largebox_003_original.npz \
    --output_fps 50 \
    --output_name holosoma_data/pipeline/converted/robot_only/sub3_largebox_003_mj_fps50.npz \
    --data_format smplx \
    --object_name ground \
    --once

python data_conversion/convert_data_format_mj.py \
    --input_file holosoma_data/pipeline/retargeted/g1/robot_only/lafan/dance2_subject1.npz \
    --output_fps 50 \
    --output_name holosoma_data/pipeline/converted/robot_only/dance2_subject1_mj_fps50.npz \
    --data_format lafan \
    --object_name ground \
    --once
```

### Robot + object

```bash
python data_conversion/convert_data_format_mj.py \
    --input_file holosoma_data/pipeline/retargeted/g1/object_interaction/omomo/sub3_largebox_003_original.npz \
    --output_fps 50 \
    --output_name holosoma_data/pipeline/converted/object_interaction/sub3_largebox_003_mj_w_obj.npz \
    --data_format smplx \
    --object_name largebox \
    --has_dynamic_object \
    --once
```

### 27-DOF

Add `--robot-config.robot-dof 27` to any conversion command.
