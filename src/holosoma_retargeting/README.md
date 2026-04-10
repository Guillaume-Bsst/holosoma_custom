# Holosoma Motion Retargeting

This repository provides tools for retargeting human motion data to humanoid robots. It supports multiple data formats (smplx, mocap, lafan) and task types including robot-only motion, object interaction, and climbing.

**Working directory**: All commands in this guide are run from `src/holosoma_retargeting/holosoma_retargeting/` after sourcing the retargeting environment:

```bash
source scripts/source_retargeting_setup.sh    # run from holosoma/ root
cd src/holosoma_retargeting/holosoma_retargeting
```

**Data Requirements**: The retargeting pipeline requires motion data in world joint positions. For custom data, you need to prepare world joint positions in shape `(T, J, 3)` where T is the number of frames and J is the number of joints, and modify `demo_joints` and `joints_mapping` defined in `config_types/data_type.py`.

**Dataset uniformity**: All datasets (OMOMO, AMASS/SFU, LAFAN) are stored as pre-processed files in `holosoma_data/datasets/`. Each dataset has a preparation script in `data_utils/` that converts raw downloaded data into the format expected by the retargeting pipeline. All retargeters (OmniRetarget, GMR, Test) can process any dataset. See [Dataset Preparation](../../holosoma_data/README.md).

All datasets and pipeline outputs are stored in `holosoma_data/` (the shared data package). Default paths below assume the working directory is `src/holosoma_retargeting/holosoma_retargeting/`.

## DOF Mode Selection (27-DOF / 29-DOF)

The retargeting pipeline supports both **29-DOF** (default) and **27-DOF** modes for the G1 robot. The 27-DOF mode removes the `waist_roll_joint` and `waist_pitch_joint`, keeping only `waist_yaw_joint` in the torso.

To select the DOF mode, use `--robot-config.robot-dof`:

```bash
# 29-DOF (default, can be omitted)
python examples/robot_retarget.py --robot-config.robot-dof 29 ...

# 27-DOF
python examples/robot_retarget.py --robot-config.robot-dof 27 ...
```

This flag automatically adjusts:
- The robot URDF/XML model (`g1_29dof_retargeting.urdf` or `g1_27dof_retargeting.urdf`)
- Joint limit bounds (manual lower/upper bounds)
- Cost weights (waist joints)
- Default output directory (e.g. `holosoma_data/pipeline/retargeted/g1_27dof/...`)

The flag works with all commands: single retargeting, batch processing, evaluation, and data conversion.

## Single Sequence Motion Retargeting

```bash
# Robot-only (OMOMO)
python examples/robot_retarget.py --data_path holosoma_data/datasets/omomo --task-type robot_only --task-name sub3_largebox_003 --data_format smplx --retargeter.debug --retargeter.visualize

# Object interaction (OMOMO)
python examples/robot_retarget.py --data_path holosoma_data/datasets/omomo --task-type object_interaction --task-name sub3_largebox_003 --data_format smplx --retargeter.debug --retargeter.visualize

# Climbing
python examples/robot_retarget.py --data_path holosoma_data/datasets/climb --task-type climbing --task-name mocap_climb_seq_0 --data_format mocap --robot-config.robot-urdf-file holosoma_data/robots/g1/g1_29dof_spherehand_retargeting.urdf --retargeter.debug --retargeter.visualize

# 27-DOF example (robot-only)
python examples/robot_retarget.py --data_path holosoma_data/datasets/omomo --task-type robot_only --task-name sub3_largebox_003 --data_format smplx --robot-config.robot-dof 27 --retargeter.debug --retargeter.visualize
```

**Note**: Add `--augmentation` to run sequences with augmentation. You must first run the original sequence before adding augmentation.

## Batch Processing for Motion Retargeting

```bash
# Robot-only (OMOMO)
python examples/parallel_robot_retarget.py --data-dir holosoma_data/datasets/omomo --task-type robot_only --data_format smplx --save_dir holosoma_data/pipeline/retargeted/g1/robot_only/omomo --task-config.object-name ground

# Object interaction (OMOMO)
python examples/parallel_robot_retarget.py --data-dir holosoma_data/datasets/omomo --task-type object_interaction --data_format smplx --save_dir holosoma_data/pipeline/retargeted/g1/object_interaction/omomo --task-config.object-name largebox

# Climbing
python examples/parallel_robot_retarget.py --data-dir holosoma_data/datasets/climb --task-type climbing --data_format mocap --robot-config.robot-urdf-file holosoma_data/robots/g1/g1_29dof_spherehand_retargeting.urdf --task-config.object-name multi_boxes --save_dir holosoma_data/pipeline/retargeted/g1/climbing/mocap_climb
```

**Note**: Add `--augmentation` to run original sequences and sequences with augmentation (for object interaction and climbing tasks).

## Data Preparation

Datasets live in `holosoma_data/datasets/`. To test on more motion sequences, follow the instructions below.

### OMOMO

#### Download the Original OMOMO Data

1. Download the [OMOMO dataset](https://drive.google.com/file/d/1tZVqLB7II0whI-Qjz-z-AU3ponSEyAmm/view) (`data/` folder)
2. Download the [SMPL-H model](https://mano.is.tue.mpg.de/download.php) (select "Extended SMPL+H model for AMASS") and place it in `holosoma_data/datasets/smplh_models/` (structure: `smplh/male/model.npz`, `smplh/female/model.npz`, `smplh/neutral/model.npz`)

#### Convert the Original OMOMO Data Format for Motion Retargeting

```bash
cd data_utils/
python prep_omomo_for_rt.py \
    --omomo-root-folder /path/to/omomo/data \
    --output-folder holosoma_data/datasets/omomo \
    --smplh-root-folder holosoma_data/datasets/smplh_models \
    --objects-output-folder holosoma_data/objects
cd ..
```

This converts the raw OMOMO `.p` files to `.npz` format with global joint positions and height — the same format as AMASS/SFU.

### LAFAN

#### Download the Original LAFAN Data

1. Download [lafan1.zip](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip) by clicking "View Raw"
2. Put `lafan1.zip` in your designated data folder and uncompress it to `DATA_FOLDER_PATH/lafan`
3. The file structure should be `DATA_FOLDER_PATH/lafan/*.bvh`

#### Convert the Original LAFAN Data Format for Motion Retargeting

We need some data processing files from the [LAFAN GitHub repo](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).

```bash
cd data_utils/
# Clone processing scripts if not already present:
git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git
mv ubisoft-laforge-animation-dataset/lafan1 .

python extract_global_positions.py --input_dir DATA_FOLDER_PATH/lafan
# output goes to holosoma_data/datasets/lafan/ by default
cd ..
```

This will convert the BVH files to `.npy` format with global joint positions.

**Note**: For LAFAN data, you need to relax the foot sticking constraint by setting `--retargeter.foot-sticking-tolerance` (default is stricter). You can adjust this tolerance number based on your data quality and retargeting results.

#### Single Sequence Retargeting on LAFAN

```bash
python examples/robot_retarget.py --data_path holosoma_data/datasets/lafan --task-type robot_only --task-name dance2_subject1 --data_format lafan --task-config.ground-range -10 10 --save_dir holosoma_data/pipeline/retargeted/g1/robot_only/lafan --retargeter.debug --retargeter.visualize --retargeter.foot-sticking-tolerance 0.02
```

#### Batch Processing for Motion Retargeting on LAFAN

```bash
python examples/parallel_robot_retarget.py --data-dir holosoma_data/datasets/lafan --task-type robot_only --data_format lafan --save_dir holosoma_data/pipeline/retargeted/g1/robot_only/lafan --task-config.object-name ground --task-config.ground-range -10 10 --retargeter.foot-sticking-tolerance 0.02
```

### AMASS SMPL-X

#### Download the Original AMASS Data

1. Follow the [AMASS](https://amass.is.tue.mpg.de/) instructions to download the original AMASS data
2. The AMASS data structure should be `/path/to/amass/dataset_name/subject_name/*.npz`

#### Download SMPL-X Models

1. Follow the [SMPL-X](https://smpl-x.is.tue.mpg.de/index.html) instructions to download SMPL-X models
2. For AMASS data, we tested on SMPL-X N (neutral) format
3. Place the models in `holosoma_data/datasets/smplx_models/` — structure: `smplx_models/models/smplx/SMPLX_NEUTRAL.npz`

#### Convert the Original AMASS SMPL-X Data Format for Motion Retargeting

We provide `data_utils/prep_amass_smplx_for_rt.py` for converting AMASS SMPLX data to the format required for motion retargeting.

```bash
cd data_utils/
# Install dependencies if needed:
# git clone https://github.com/nghorbani/human_body_prior.git
# pip install tqdm dotmap PyYAML omegaconf loguru
# cd human_body_prior && python setup.py develop && cd ..

python prep_amass_smplx_for_rt.py \
  --amass-root-folder /path/to/amass \
  --output-folder holosoma_data/datasets/sfu \
  --model-root-folder holosoma_data/datasets/smplx_models
cd ..
```

This will convert the AMASS `.npz` files to `.npz` format with global joint positions and height information.

**Note**: You can optionally specify `--subdataset-folder` to process only a specific subdataset (e.g., `HumanEva`). If not specified, it will process all datasets recursively.

#### Single Sequence Retargeting on AMASS SMPL-X

```bash
python examples/robot_retarget.py --data_path holosoma_data/datasets/sfu --task-type robot_only --task-name HumanEva_S3_Jog_1_stageii --data_format smplx --task-config.ground-range -10 10 --save_dir holosoma_data/pipeline/retargeted/g1/robot_only/amass_smplx --retargeter.debug --retargeter.visualize
```

#### Batch Processing for Motion Retargeting on AMASS SMPL-X

```bash
python examples/parallel_robot_retarget.py --data-dir holosoma_data/datasets/sfu --task-type robot_only --data_format smplx --save_dir holosoma_data/pipeline/retargeted/g1/robot_only/amass_smplx --task-config.object-name ground --task-config.ground-range -10 10
```

## Check Visualizations of Saved Retargeting Results

Use `--robot_urdf` to select the correct URDF matching the DOF mode used during retargeting.

```bash
# Visualize object-interaction results (29-DOF)
python viser_player.py --robot_urdf holosoma_data/robots/g1/g1_29dof_retargeting.urdf \
    --object_urdf holosoma_data/objects/largebox/largebox.urdf \
    --qpos_npz holosoma_data/pipeline/retargeted/g1/object_interaction/omomo/sub3_largebox_003_original.npz

# Visualize climbing results
python viser_player.py --robot_urdf holosoma_data/robots/g1/g1_29dof_spherehand_retargeting.urdf \
    --object_urdf holosoma_data/datasets/climb/mocap_climb_seq_0/multi_boxes.urdf \
    --qpos_npz holosoma_data/pipeline/retargeted/g1/climbing/mocap_climb/mocap_climb_seq_0_original.npz

python viser_player.py --robot_urdf holosoma_data/robots/g1/g1_29dof_spherehand_retargeting.urdf \
    --object_urdf holosoma_data/datasets/climb/mocap_climb_seq_0/multi_boxes_scaled_0.74_0.74_0.89.urdf \
    --qpos_npz holosoma_data/pipeline/retargeted/g1/climbing/mocap_climb/mocap_climb_seq_0_z_scale_1.2.npz

# Visualize robot only results (29-DOF)
python viser_player.py --robot_urdf holosoma_data/robots/g1/g1_29dof_retargeting.urdf \
    --qpos_npz holosoma_data/pipeline/retargeted/g1/robot_only/omomo/sub3_largebox_003_original.npz

# Visualize robot only results (27-DOF)
python viser_player.py --robot_urdf holosoma_data/robots/g1/g1_27dof_retargeting.urdf \
    --qpos_npz holosoma_data/pipeline/retargeted/g1/robot_only/omomo/sub3_largebox_003_original.npz

# Visualize LAFAN robot only results
python viser_player.py --robot_urdf holosoma_data/robots/g1/g1_29dof_retargeting.urdf \
    --qpos_npz holosoma_data/pipeline/retargeted/g1/robot_only/lafan/dance2_subject1.npz

# Visualize AMASS results
python viser_player.py --robot_urdf holosoma_data/robots/g1/g1_29dof_retargeting.urdf \
    --qpos_npz holosoma_data/pipeline/retargeted/g1/robot_only/amass_smplx/HumanEva_S3_Jog_1_stageii.npz
```

## Quantitative Evaluation

```bash
# Evaluate robot-object interaction
python evaluation/eval_retargeting.py --res_dir holosoma_data/pipeline/retargeted/g1/object_interaction/omomo --data_dir holosoma_data/datasets/omomo --data_type "robot_object"

# Evaluate climbing sequence
python evaluation/eval_retargeting.py --res_dir holosoma_data/pipeline/retargeted/g1/climbing/mocap_climb --data_dir holosoma_data/datasets/climb --data_type "robot_terrain" --robot-config.robot-urdf-file holosoma_data/robots/g1/g1_29dof_spherehand_retargeting.urdf

# Evaluate robot only (OMOMO)
python evaluation/eval_retargeting.py --res_dir holosoma_data/pipeline/retargeted/g1/robot_only/omomo --data_dir holosoma_data/datasets/omomo --data_type "robot_only"
```

## Prepare Data for Training RL Whole-Body Tracking Policy

To prepare data for training RL whole-body tracking policies, you need to follow a two-step process:

1. **First, run retargeting** to obtain `.npz` files containing the retargeted robot motion. Use the retargeting commands shown in the sections above.

2. **Then, run the data conversion code** below to convert the retargeted `.npz` files into the format required for RL training.

**Note**: If you run this code on Mac, please use `mjpython` instead of `python`.

### Mac (using mjpython)

```bash
mjpython data_conversion/convert_data_format_mj.py --input_file holosoma_data/pipeline/retargeted/g1/robot_only/omomo/sub3_largebox_003.npz --output_fps 50 --output_name holosoma_data/pipeline/converted/robot_only/sub3_largebox_003_mj_fps50.npz --data_format smplx --object_name "ground" --once

mjpython data_conversion/convert_data_format_mj.py --input_file holosoma_data/pipeline/retargeted/g1/object_interaction/omomo/sub3_largebox_003_original.npz --output_fps 50 --output_name holosoma_data/pipeline/converted/object_interaction/sub3_largebox_003_mj_w_obj.npz --data_format smplx --object_name "largebox" --has_dynamic_object --once
```

### Robot-Only Setting

```bash
python data_conversion/convert_data_format_mj.py --input_file holosoma_data/pipeline/retargeted/g1/robot_only/omomo/sub3_largebox_003.npz --output_fps 50 --output_name holosoma_data/pipeline/converted/robot_only/sub3_largebox_003_mj_fps50.npz --data_format smplx --object_name "ground" --once

python data_conversion/convert_data_format_mj.py --input_file holosoma_data/pipeline/retargeted/g1/robot_only/lafan/dance2_subject1.npz --output_fps 50 --output_name holosoma_data/pipeline/converted/robot_only/dance2_subject1_mj_fps50.npz --data_format lafan --object_name "ground" --once
```

### Robot-Object Setting

```bash
python data_conversion/convert_data_format_mj.py --input_file holosoma_data/pipeline/retargeted/g1/object_interaction/omomo/sub3_largebox_003_original.npz --output_fps 50 --output_name holosoma_data/pipeline/converted/object_interaction/sub3_largebox_003_mj_w_obj.npz --data_format smplx --object_name "largebox" --has_dynamic_object --once
```

### OmniRetarget Data

For OmniRetarget data downloaded from HuggingFace, please add `--use_omniretarget_data` for data conversion.

```bash
python data_conversion/convert_data_format_mj.py --input_file OmniRetarget/robot-object/sub3_largebox_003_original.npz --output_fps 50 --output_name holosoma_data/pipeline/converted/object_interaction/sub3_largebox_003_mj_w_obj_omnirt.npz --data_format smplx --object_name "largebox" --has_dynamic_object --use_omniretarget_data --once
```

## Custom Human Motion Data Format
Please see the instructions for custom human motion data formats: [ADD_MOTION_FORMAT_README.md](ADD_MOTION_FORMAT_README.md)

## Custom Robot Type
Please see the instructions for retargeting custom robot types: [ADD_ROBOT_TYPE_README.md](ADD_ROBOT_TYPE_README.md)
