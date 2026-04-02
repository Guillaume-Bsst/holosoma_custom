# Holosoma — Retargeting & Data Conversion

All commands are run from the `holosoma/` directory.

---

## 1. Retargeting

### Robot-only (OMOMO)

```bash
source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && conda activate hsretargeting
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
source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && conda activate hsretargeting
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
source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && conda activate hsretargeting
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
source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && conda activate hsretargeting
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

---

## 2. Data Conversion

### Robot-only

```bash
source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && conda activate hsretargeting
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
source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && conda activate hsretargeting
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