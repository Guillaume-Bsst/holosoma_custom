# Holosoma — Retargeting & Data Conversion

All commands are run from the `holosoma/` directory.

---

## DOF mode (27 / 29)

The retargeting and data conversion pipelines support both **29-DOF** (G1 pro, default) and
**27-DOF** (G1 base) modes. The 27-DOF mode locks `waist_roll_joint` and `waist_pitch_joint`,
keeping only `waist_yaw_joint` in the torso — matching the physical G1 base robot.

To select the DOF mode, add `--robot-config.robot-dof 27` (or `29`) to any retargeting or
data-conversion command. This automatically adjusts:

- The robot URDF/XML model (`g1_27dof` vs `g1_29dof`)
- Joint limit bounds and cost weights
- Joint name lists for data conversion
- Default output directory (e.g. `demo_results/g1_27dof/...` vs `demo_results/g1_29dof/...`)

> **Important:** The DOF mode must be consistent across the whole pipeline —
> retargeting, data conversion, and training must all use the same DOF setting.
> A `.npz` retargeted with 27-DOF must be converted and trained with 27-DOF configs.

---

## 1. Retargeting

### Robot-only (OMOMO)

```bash
source source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting
python examples/robot_retarget.py \
    --data_path demo_data/OMOMO_new \
    --task-type robot_only \
    --task-name sub3_largebox_003 \
    --data_format smplh \
    --retargeter.debug \
    --retargeter.visualize
```

### Object interaction (OMOMO)

```bash
source source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting
python examples/robot_retarget.py \
    --data_path demo_data/OMOMO_new \
    --task-type object_interaction \
    --task-name sub3_largebox_003 \
    --data_format smplh \
    --retargeter.debug \
    --retargeter.visualize
```

### Robot-only (SFU / AMASS, SMPL-X format)

```bash
source source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting
python examples/robot_retarget.py \
    --data_path demo_data/SFU/ \
    --task-type robot_only \
    --task-name SFU_0005_0005_2FeetJump001_stageii \
    --data_format smplx \
    --task-config.ground-range -10 10 \
    --save_dir demo_results/g1/robot_only/SFU \
    --retargeter.debug \
    --retargeter.visualize
```

### Climbing

```bash
source source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting
python examples/robot_retarget.py \
    --data_path demo_data/climb \
    --task-type climbing \
    --task-name mocap_climb_seq_0 \
    --data_format mocap \
    --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf \
    --retargeter.debug \
    --retargeter.visualize
```

### 27-DOF examples

Any of the above commands can be run in 27-DOF mode by adding `--robot-config.robot-dof 27`:

```bash
# Robot-only, 27-DOF
source source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting
python examples/robot_retarget.py \
    --data_path demo_data/OMOMO_new \
    --task-type robot_only \
    --task-name sub3_largebox_003 \
    --data_format smplh \
    --robot-config.robot-dof 27 \
    --retargeter.debug \
    --retargeter.visualize

# Object interaction, 27-DOF
python examples/robot_retarget.py \
    --data_path demo_data/OMOMO_new \
    --task-type object_interaction \
    --task-name sub3_largebox_003 \
    --data_format smplh \
    --robot-config.robot-dof 27 \
    --retargeter.debug \
    --retargeter.visualize
```

---

## 2. Data Conversion

### Robot-only

```bash
source source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting
python data_conversion/convert_data_format_mj.py \
    --input_file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz \
    --output_fps 50 \
    --output_name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz \
    --data_format smplh \
    --object_name 'ground' \
    --once
```

### Robot + object

```bash
source source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting
python data_conversion/convert_data_format_mj.py \
    --input_file ./demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz \
    --output_fps 50 \
    --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz \
    --data_format smplh \
    --object_name 'largebox' \
    --has_dynamic_object \
    --once
```

### 27-DOF data conversion

Add `--robot-config.robot-dof 27` to convert data retargeted in 27-DOF mode:

```bash
# Robot-only, 27-DOF
source source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting
python data_conversion/convert_data_format_mj.py \
    --input_file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz \
    --output_fps 50 \
    --output_name converted_res/robot_only/sub3_largebox_003_27dof_mj_fps50.npz \
    --data_format smplh \
    --object_name 'ground' \
    --robot-config.robot-dof 27 \
    --once

# Robot + object, 27-DOF
python data_conversion/convert_data_format_mj.py \
    --input_file ./demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz \
    --output_fps 50 \
    --output_name converted_res/object_interaction/sub3_largebox_003_27dof_mj_w_obj.npz \
    --data_format smplh \
    --object_name 'largebox' \
    --has_dynamic_object \
    --robot-config.robot-dof 27 \
    --once
```

The converted `.npz` files can then be used for training with the matching 27-DOF experiment
configs (e.g. `exp:g1-27dof-wbt-mjwarp`) — see [TRAINING.md](TRAINING.md).