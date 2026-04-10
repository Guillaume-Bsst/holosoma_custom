# Holosoma Data

Shared data package used by all Holosoma modules. Contains datasets, robot models, object assets, and pipeline outputs.

All commands below are run from `src/holosoma_retargeting/holosoma_retargeting/` (where `holosoma_data/` is available as a symlink).

## Structure

```
holosoma_data/
├── datasets/
│   ├── omomo/           # OMOMO sequences — .npz (global_joint_positions + height + object_poses)
│   ├── sfu/             # SFU/AMASS sequences — .npz (global_joint_positions + height)
│   ├── lafan/           # LAFAN sequences — .npy (global_joint_positions)
│   ├── climb/           # Climbing sequences — .npy (global_joint_positions)
│   ├── smplx_models/    # SMPL-X neutral model for AMASS processing
│   ├── smplh_models/    # SMPL-H models (male/female/neutral) for OMOMO processing
│   └── backup/          # Raw/old formats (OMOMO .pt, SFU_demo, SFU_raw, LAFAN .bvh)
├── objects/             # Object assets (.obj + .urdf + .xml) for object interaction
│   ├── largebox/
│   ├── smallbox/
│   └── ...
├── robots/              # Robot URDF/XML files
│   └── g1/
├── pipeline/
│   ├── retargeted/      # Retargeted robot motion (.npz with qpos)
│   └── converted/       # Converted for RL training (.npz with physics state)
└── policies/            # Trained ONNX policies
```

## Dataset Format

All datasets use a **uniform `.npz` format** with at minimum:
- `global_joint_positions`: `(T, J, 3)` — joint positions in world frame (metres)
- `height`: `float` — subject height in metres

Object-interaction sequences additionally contain:
- `object_poses`: `(T, 7)` — `[qw, qx, qy, qz, x, y, z]`

The exception is LAFAN (`.npy` format, `(T, J, 3)`) and climbing (`.npy`, `(T, J, 3)`).

---

## Dataset Preparation

### OMOMO

**Download:**
1. Download the [OMOMO dataset](https://drive.google.com/file/d/1tZVqLB7II0whI-Qjz-z-AU3ponSEyAmm/view) (`data/` folder)
2. Download the [SMPL-H model](https://mano.is.tue.mpg.de/download.php) (select "Extended SMPL+H model for AMASS") and place it in `holosoma_data/datasets/smplh_models/` with structure:
   ```
   smplh_models/
   ├── male/model.npz
   ├── female/model.npz
   └── neutral/model.npz
   ```

**Convert:**
```bash
source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting/data_utils/
python prep_omomo_for_rt.py \
    --omomo-root-folder /path/to/omomo/data \
    --output-folder holosoma_data/datasets/omomo \
    --smplh-root-folder holosoma_data/datasets/smplh_models \
    --objects-output-folder holosoma_data/objects
```

This runs SMPL-H forward kinematics to produce `(T, 22, 3)` global joint positions + height + object poses. Object assets (`.urdf`, `.xml`) are generated automatically from the `.obj` files in `captured_objects/`.

---

### SFU / AMASS SMPL-X

**Download:**
1. Follow [AMASS](https://amass.is.tue.mpg.de/) instructions to download AMASS data
2. Download [SMPL-X models](https://smpl-x.is.tue.mpg.de/) (tested with SMPL-X N neutral format)
3. Place models in `holosoma_data/datasets/smplx_models/` — structure: `smplx_models/models/smplx/SMPLX_NEUTRAL.npz`

**Convert:**
```bash
source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting/data_utils/
# Install human_body_prior if needed:
# git clone https://github.com/nghorbani/human_body_prior.git
# pip install tqdm dotmap PyYAML omegaconf loguru && cd human_body_prior && python setup.py develop && cd ..

python prep_amass_smplx_for_rt.py \
    --amass-root-folder /path/to/amass \
    --output-folder holosoma_data/datasets/sfu \
    --model-root-folder holosoma_data/datasets/smplx_models
```

---

### LAFAN

**Download:**
1. Download [lafan1.zip](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip) (click "View Raw")
2. Uncompress to `DATA_FOLDER_PATH/lafan/` — structure: `DATA_FOLDER_PATH/lafan/*.bvh`

**Convert:**
```bash
source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting/data_utils/
# Clone processing scripts if not present:
# git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git
# mv ubisoft-laforge-animation-dataset/lafan1 .

python extract_global_positions.py --input_dir DATA_FOLDER_PATH/lafan
# output goes to holosoma_data/datasets/lafan/ by default
```

**Note:** LAFAN outputs `.npy` (not `.npz`). Use `--data_format lafan` when retargeting.
