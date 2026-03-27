# Holosoma — Retargeting & Data Conversion

All commands are run from the `holosoma/` directory.
Requires the Apptainer image (`deploy/cluster/holosoma.sif`) — see [INSTALL.md](INSTALL.md).

---

## 1. Retargeting

### Robot-only (OMOMO)

```bash
apptainer exec --nv --writable-tmpfs --bind /run deploy/cluster/holosoma.sif bash -c "
    mkdir -p /home/gbesset && \
    ln -sf /root/.holosoma_deps /home/gbesset/.holosoma_deps && \
    export MUJOCO_GL=egl && \
    export PYOPENGL_PLATFORM=egl && \
    unset DISPLAY && \
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hsretargeting && \
    cd src/holosoma_retargeting/holosoma_retargeting && \
    python examples/robot_retarget.py \
        --data_path demo_data/OMOMO_new \
        --task-type robot_only \
        --task-name sub3_largebox_003 \
        --data_format smplh
"
```

### Object interaction (OMOMO)

```bash
apptainer exec --nv --writable-tmpfs --bind /run deploy/cluster/holosoma.sif bash -c "
    mkdir -p /home/gbesset && \
    ln -sf /root/.holosoma_deps /home/gbesset/.holosoma_deps && \
    export MUJOCO_GL=egl && \
    export PYOPENGL_PLATFORM=egl && \
    unset DISPLAY && \
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hsretargeting && \
    cd src/holosoma_retargeting/holosoma_retargeting && \
    python examples/robot_retarget.py \
        --data_path demo_data/OMOMO_new \
        --task-type object_interaction \
        --task-name sub3_largebox_003 \
        --data_format smplh
"
```

### Climbing

```bash
apptainer exec --nv --writable-tmpfs --bind /run deploy/cluster/holosoma.sif bash -c "
    mkdir -p /home/gbesset && \
    ln -sf /root/.holosoma_deps /home/gbesset/.holosoma_deps && \
    export MUJOCO_GL=egl && \
    export PYOPENGL_PLATFORM=egl && \
    unset DISPLAY && \
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hsretargeting && \
    cd src/holosoma_retargeting/holosoma_retargeting && \
    python examples/robot_retarget.py \
        --data_path demo_data/climb \
        --task-type climbing \
        --task-name mocap_climb_seq_0 \
        --data_format mocap \
        --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf
"
```

---

## 2. Data Conversion

### Robot-only

```bash
apptainer exec --nv --writable-tmpfs --bind /run deploy/cluster/holosoma.sif bash -c "
    mkdir -p /home/gbesset && \
    ln -sf /root/.holosoma_deps /home/gbesset/.holosoma_deps && \
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hsretargeting && \
    cd src/holosoma_retargeting/holosoma_retargeting && \
    sed -i 's/viewer = mjv/#viewer = mjv/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.opt/#viewer.opt/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.cam/#viewer.cam/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.sync/#viewer.sync/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.close/#viewer.close/g' data_conversion/convert_data_format_mj.py && \
    python data_conversion/convert_data_format_mj.py \
        --input_file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz \
        --output_fps 50 \
        --output_name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz \
        --data_format smplh \
        --object_name 'ground' \
        --once
"
```

### Robot + object

```bash
apptainer exec --nv --writable-tmpfs --bind /run deploy/cluster/holosoma.sif bash -c "
    mkdir -p /home/gbesset && \
    ln -sf /root/.holosoma_deps /home/gbesset/.holosoma_deps && \
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hsretargeting && \
    cd src/holosoma_retargeting/holosoma_retargeting && \
    sed -i 's/viewer = mjv/#viewer = mjv/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.opt/#viewer.opt/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.cam/#viewer.cam/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.sync/#viewer.sync/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.close/#viewer.close/g' data_conversion/convert_data_format_mj.py && \
    python data_conversion/convert_data_format_mj.py \
        --input_file ./demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz \
        --output_fps 50 \
        --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz \
        --data_format smplh \
        --object_name 'largebox' \
        --has_dynamic_object \
        --once
"
```