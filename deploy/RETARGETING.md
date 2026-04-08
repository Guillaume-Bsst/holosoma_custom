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

The pipeline supports three retargeting algorithms, selected via `--retargeter-method`:

| Method | Flag | Task types | Description |
| --- | --- | --- | --- |
| `omniretarget` | *(default)* | robot_only, object_interaction, climbing | Interaction Mesh + SQP |
| `gmr` | `--retargeter-method gmr` | robot_only | IK-based (mink/mujoco, requires GMR library) |
| `test` | `--retargeter-method test` | robot_only | Native mink IK in holosoma (no GMR dependency) |

Each algorithm has its own nested config namespace:

- OmniRetarget params: `--retargeter.<param>`
- GMR params: `--gmr.<param>`
- Test params: `--test.<param>`

### Robot-only (OMOMO) — OmniRetarget

```bash
source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting
python examples/robot_retarget.py \
    --data_path holosoma_data/datasets/OMOMO \
    --task-type robot_only \
    --task-name sub3_largebox_003 \
    --data_format smplh \
    --retargeter.debug \
    --retargeter.visualize
```

### Object interaction (OMOMO) — OmniRetarget

```bash
source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting
python examples/robot_retarget.py \
    --data_path holosoma_data/datasets/OMOMO \
    --task-type object_interaction \
    --task-name sub3_largebox_003 \
    --data_format smplh \
    --retargeter.debug \
    --retargeter.visualize
```

### Robot-only (SFU / AMASS, SMPL-X format) — OmniRetarget

```bash
source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting
python examples/robot_retarget.py \
    --data_path holosoma_data/datasets/SFU_demo/ \
    --task-type robot_only \
    --task-name SFU_0005_0005_2FeetJump001_stageii \
    --data_format smplx \
    --task-config.ground-range -10 10 \
    --save_dir demo_results/g1/robot_only/SFU \
    --retargeter.debug \
    --retargeter.visualize
```

### Robot-only (SMPL-X format) — GMR

Requires the GMR library (expected at `../GMR`, alongside the `holosoma/` folder).
Install it once with:

```bash
pip install -e ../GMR
```

Uses IK-based retargeting with a two-pass damped least squares solver.

GMR natively supports `smplx` and `bvh` formats only (not SMPLH/InterMimic `.pt` files).
Use OmniRetarget for OMOMO data.

```bash
source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting
python examples/robot_retarget.py \
    --retargeter-method gmr \
    --task-type robot_only \
    --data_path holosoma_data/datasets/SFU_demo \
    --task-name my_sequence \
    --data_format smplx \
    --gmr.src_human smplx
```

### Robot-only — Test (native mink IK)

Drop-in replacement for GMR without the external library dependency. Work in progress toward
GMR parity — see `retargeters/TODO.md` for remaining steps.

```bash
source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting
python examples/robot_retarget.py \
    --retargeter-method test \
    --task-type robot_only \
    --data_path holosoma_data/datasets/OMOMO \
    --task-name sub3_largebox_003 \
    --data_format smplh
```

### Climbing — OmniRetarget

```bash
source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting
python examples/robot_retarget.py \
    --data_path holosoma_data/datasets/climb \
    --task-type climbing \
    --task-name mocap_climb_seq_0 \
    --data_format mocap \
    --robot-config.robot-urdf-file holosoma_data/robots/g1/g1_29dof_spherehand.urdf \
    --retargeter.debug \
    --retargeter.visualize
```

### 27-DOF examples

Any of the above commands can be run in 27-DOF mode by adding `--robot-config.robot-dof 27`:

```bash
# Robot-only, 27-DOF
source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting
python examples/robot_retarget.py \
    --data_path holosoma_data/datasets/OMOMO \
    --task-type robot_only \
    --task-name sub3_largebox_003 \
    --data_format smplh \
    --robot-config.robot-dof 27 \
    --retargeter.debug \
    --retargeter.visualize

# Object interaction, 27-DOF
python examples/robot_retarget.py \
    --data_path holosoma_data/datasets/OMOMO \
    --task-type object_interaction \
    --task-name sub3_largebox_003 \
    --data_format smplh \
    --robot-config.robot-dof 27 \
    --retargeter.debug \
    --retargeter.visualize
```

### Visualizing results (viser_player)

All retargeting outputs (OmniRetarget, GMR, Test) produce a `.npz` file with a `qpos` key.
The `viser_player.py` script replays any result in the browser via [viser](https://viser.studio).

```bash
cd src/holosoma_retargeting/holosoma_retargeting

# OmniRetarget result (object pose included in qpos)
python viser_player.py \
    --qpos-npz demo_results/g1/robot_only/omomo/sub3_largebox_003.npz \
    --robot-urdf holosoma_data/robots/g1/g1_29dof.urdf

# GMR result (no object in qpos)
python viser_player.py \
    --qpos-npz holosoma_data/pipeline/retargeted/g1_29dof/gmr/sub3_largebox_003/retargeted.npz \
    --robot-urdf holosoma_data/robots/g1/g1_29dof.urdf \
    --no-assume-object-in-qpos

# 27-DOF
python viser_player.py \
    --qpos-npz demo_results/g1_27dof/robot_only/omomo/sub3_largebox_003.npz \
    --robot-urdf holosoma_data/robots/g1/g1_27dof.urdf
```

Then open the URL printed in the terminal (e.g. `http://localhost:8080`) in a browser.

> OmniRetarget also supports an inline mode: add `--retargeter.visualize` to the retargeting
> command to open the viewer directly at the end of the computation.

---

### Adding a new retargeting algorithm

1. Create `retargeters/my_method.py` with a class inheriting `BaseRetargeter`
2. Create `config_types/retargeters/my_method.py` with a frozen dataclass config
3. Register in `retargeters/registry.py`: `RETARGETER_REGISTRY["my_method"] = "...MyMethodRetargeter"`
4. Add `my_method: MyMethodConfig` to `RetargetingConfig` in `config_types/retargeting.py`
5. Extend `retargeter_method: Literal[..., "my_method"]`

See `retargeters/base.py` for the `BaseRetargeter` interface.

---

## 2. Data Conversion

### Robot-only

```bash
source source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting
python data_conversion/convert_data_format_mj.py \
    --input_file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz \
    --output_fps 50 \
    --output_name holosoma_data/pipeline/converted/robot_only/sub3_largebox_003_mj_fps50.npz \
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
    --output_name holosoma_data/pipeline/converted/object_interaction/sub3_largebox_003_mj_w_obj.npz \
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
    --output_name holosoma_data/pipeline/converted/robot_only/sub3_largebox_003_27dof_mj_fps50.npz \
    --data_format smplh \
    --object_name 'ground' \
    --robot-config.robot-dof 27 \
    --once

# Robot + object, 27-DOF
python data_conversion/convert_data_format_mj.py \
    --input_file ./demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz \
    --output_fps 50 \
    --output_name holosoma_data/pipeline/converted/object_interaction/sub3_largebox_003_27dof_mj_w_obj.npz \
    --data_format smplh \
    --object_name 'largebox' \
    --has_dynamic_object \
    --robot-config.robot-dof 27 \
    --once
```

The converted `.npz` files can then be used for training with the matching 27-DOF experiment
configs (e.g. `exp:g1-27dof-wbt-mjwarp`) — see [TRAINING.md](TRAINING.md).