"""Parallel batch retargeting entry point.

Processes multiple motion files in parallel using ProcessPoolExecutor.
Each worker calls the same pipeline as robot_retarget.py with all augmentations.

Usage examples:

    # OmniRetarget — robot_only (OMOMO / SMPLH)
    python examples/parallel_robot_retarget.py \
        --data-dir holosoma_data/datasets/OMOMO \
        --task-type robot_only \
        --data_format smplh \
        --save_dir demo_results_parallel/g1/robot_only/omomo \
        --task-config.object-name ground

    # OmniRetarget — object interaction
    python examples/parallel_robot_retarget.py \
        --data-dir holosoma_data/datasets/OMOMO \
        --task-type object_interaction \
        --data_format smplh \
        --save_dir demo_results_parallel/g1/object_interaction/omomo \
        --task-config.object-name largebox

    # GMR — robot_only (SMPLX)
    python examples/parallel_robot_retarget.py \
        --retargeter-method gmr \
        --data-dir holosoma_data/datasets/SFU_demo \
        --task-type robot_only \
        --data_format smplx

See holosoma_retargeting/retargeters/registry.py to add new algorithms.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.data_type import MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeting import ParallelRetargetingConfig  # noqa: E402
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.pipeline.data_loading import load_motion_data, setup_object_data  # noqa: E402
from holosoma_retargeting.pipeline.preprocessing import (  # noqa: E402
    convert_object_poses_to_mujoco_order,
    get_foot_sticking_sequences,
    preprocess_motion_data,
)
from holosoma_retargeting.pipeline.run import (  # noqa: E402
    _DEFAULT_DATA_FORMATS,
    _OBJECT_SCALE_AUGMENTED,
    _AUGMENTATION_TRANSLATION,
    _PIPELINE_DATA_DIR,
    _get_retargeter_cfg,
    create_task_constants,
    determine_output_path,
    initialize_robot_pose,
)
from holosoma_retargeting.retargeters.registry import build_retargeter  # noqa: E402

# ----------------------------- File discovery -----------------------------


def find_files(data_dir: Path, data_format: str, object_name: str | None = None) -> list[str]:
    """Find motion files based on data format.

    Args:
        data_dir: Directory to search.
        data_format: Data format name.
        object_name: Optional object name filter (smplh only).

    Returns:
        Sorted list of file paths.
    """
    data_dir = Path(data_dir)
    if data_format == "lafan":
        return sorted(str(p) for p in data_dir.glob("*.npy"))
    if data_format == "smplh":
        pattern = f"*{object_name}*.pt" if object_name else "*.pt"
        return sorted(str(p) for p in data_dir.glob(pattern))
    if data_format == "mocap":
        return sorted(str(p) for p in data_dir.glob("*/*.npy"))
    # smplx and custom formats
    return sorted(str(p) for p in data_dir.glob("*.npz"))


# ----------------------------- Augmentation configs -----------------------------


def generate_augmentation_configs(task_type: str, augmentation: bool = True) -> list[dict]:
    """Generate augmentation configurations for the given task type."""
    if task_type == "robot_only":
        return [{"name": "original"}]

    if task_type == "object_interaction":
        augmentations = [{"name": "original", "translation": np.array([0.0, 0.0, 0.0]), "rotation": 0.0}]
        if augmentation:
            for i, trans in enumerate([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, -0.2, 0.0]]):
                augmentations.append({"name": f"trans_{i}", "translation": np.array(trans), "rotation": 0.0})
            for i, rot in enumerate([np.pi / 4, -np.pi / 4]):
                augmentations.append({
                    "name": f"rot_{i}",
                    "translation": np.array([0.0, 0.2 * (-1) ** i, 0.0]),
                    "rotation": rot,
                })
        return augmentations

    if task_type == "climbing":
        configs = [{"name": "original", "scale": np.array([1, 1, 1])}]
        if augmentation:
            configs.extend(
                {"name": f"z_scale_{z}", "scale": np.array([1, 1, z])} for z in [0.8, 0.9, 1.1, 1.2]
            )
        return configs

    raise ValueError(f"Invalid task type: {task_type}")


# ----------------------------- Per-task worker -----------------------------


def process_single_task(args: tuple) -> None:
    """Process one motion file with all augmentations (runs in a worker process)."""
    (
        file_path,
        save_dir,
        task_type,
        data_format,
        robot_config,
        motion_data_config,
        task_config,
        retargeter_method,
        retargeter_cfg,
        augmentation,
    ) = args

    if task_type == "climbing":
        file_path = "/".join(file_path.split("/")[:-1])
    task_name = Path(file_path).stem

    task_save_dir = Path(save_dir) / task_name
    os.makedirs(task_save_dir, exist_ok=True)
    print(f"Processing: {task_name}")

    if task_type == "climbing" and task_config.object_dir is None:
        task_config = replace(task_config, object_dir=Path(file_path))

    constants = create_task_constants(robot_config, motion_data_config, task_config, task_type)
    demo_joints: list[str] = motion_data_config.resolved_demo_joints
    toe_names: list[str] = motion_data_config.toe_names

    human_joints, object_poses, smpl_scale = load_motion_data(
        task_type, data_format, Path(file_path).parent, task_name, constants, motion_data_config
    )

    human_joints_original = human_joints.copy()
    object_poses_original = object_poses.copy()

    augmentations = generate_augmentation_configs(task_type, augmentation)
    print(f"  Number of augmentations: {len(augmentations)}")

    for k, aug_config in enumerate(augmentations):
        human_joints = human_joints_original.copy()
        object_poses = object_poses_original.copy()
        is_augmentation_run = k > 0

        file_name = determine_output_path(task_type, task_save_dir, task_name, is_augmentation_run)
        if Path(file_name).exists():
            continue

        print(f"  Processing augmentation: {aug_config['name']}")

        # Object geometry
        aug_scale = aug_config.get("scale", _OBJECT_SCALE_AUGMENTED)
        object_local_pts, object_local_pts_demo, object_urdf_path = setup_object_data(
            task_type,
            constants,
            task_config.object_dir,
            smpl_scale,
            task_config,
            augmentation=is_augmentation_run,
            object_scale_augmented=aug_scale,
        )
        if object_urdf_path is not None:
            constants.OBJECT_URDF_FILE = object_urdf_path

        # Build retargeter
        retargeter = build_retargeter(retargeter_method, retargeter_cfg, constants)

        if task_type not in retargeter.supported_task_types:
            raise ValueError(
                f"Retargeter '{retargeter_method}' does not support task type '{task_type}'."
            )

        # Preprocess
        if task_type == "robot_only":
            human_joints = preprocess_motion_data(demo_joints, human_joints, toe_names, smpl_scale)
        else:
            human_joints, object_poses, _ = preprocess_motion_data(
                demo_joints, human_joints, toe_names, scale=smpl_scale, object_poses=object_poses
            )

        # Init robot pose
        aug_translation = aug_config.get("translation", _AUGMENTATION_TRANSLATION)
        aug_rotation = aug_config.get("rotation", 0.0)
        q_init, q_nominal, object_poses_augmented, human_joints, object_poses = initialize_robot_pose(
            task_type,
            data_format,
            human_joints,
            object_poses,
            constants,
            demo_joints,
            task_config,
            is_augmentation_run,
            task_save_dir,
            task_name,
            augmentation_translation=aug_translation,
            augmentation_rotation=aug_rotation,
        )

        # Foot sticking
        foot_sticking_sequences = get_foot_sticking_sequences(human_joints, demo_joints, toe_names)
        if task_type == "object_interaction":
            foot_sticking_sequences[0][toe_names[0]] = False
            foot_sticking_sequences[0][toe_names[1]] = False

        # Retarget
        retargeter.retarget_motion(
            human_joint_motions=human_joints,
            object_poses=object_poses,
            object_poses_augmented=object_poses_augmented,
            object_points_local_demo=object_local_pts_demo,
            object_points_local=object_local_pts,
            foot_sticking_sequences=foot_sticking_sequences,
            q_a_init=q_init,
            q_nominal_list=q_nominal,
            original=(k == 0),
            dest_res_path=file_name,
        )


# ----------------------------- Main -----------------------------


def main(cfg: ParallelRetargetingConfig) -> None:
    """Main parallel retargeting pipeline."""
    robot = cfg.robot
    task_type = cfg.task_type
    retargeter_method = cfg.retargeter_method
    robot_name = cfg.robot_config.ROBOT_NAME

    data_format: str = cfg.data_format or _DEFAULT_DATA_FORMATS[task_type]

    if cfg.save_dir is not None:
        save_dir = cfg.save_dir
    else:
        save_dir = _PIPELINE_DATA_DIR / robot_name / retargeter_method
    data_dir = cfg.data_dir

    os.makedirs(save_dir, exist_ok=True)
    print(f"Task type: {task_type} | Format: {data_format} | Method: {retargeter_method}")
    print(f"Data dir: {data_dir} | Save dir: {save_dir}")

    if cfg.robot_config.robot_type != robot:
        cfg.robot_config = RobotConfig(robot_type=robot)
    if cfg.motion_data_config.robot_type != robot or cfg.motion_data_config.data_format != data_format:
        cfg.motion_data_config = MotionDataConfig(data_format=data_format, robot_type=robot)

    files = find_files(
        data_dir,
        data_format,
        cfg.task_config.object_name if task_type != "robot_only" else None,
    )
    print(f"Found {len(files)} files for task type: {task_type}")

    retargeter_cfg = _get_retargeter_cfg(cfg, retargeter_method)

    process_args = [
        (
            file_path,
            save_dir,
            task_type,
            data_format,
            cfg.robot_config,
            cfg.motion_data_config,
            cfg.task_config,
            retargeter_method,
            retargeter_cfg,
            cfg.augmentation,
        )
        for file_path in files
    ]

    max_workers = cfg.max_workers or mp.cpu_count()
    print(f"Using {max_workers} parallel workers")

    start_time = time.time()
    successful = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_single_task, arg): arg[0] for arg in process_args}
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                future.result()
                print(f"Completed: {file_path}")
                successful += 1
            except Exception as e:
                import traceback
                print(f"Failed {file_path}: {e}")
                traceback.print_exc()
                failed += 1

    elapsed = time.time() - start_time
    print(f"\n=== Summary ===")
    print(f"Task type: {task_type} | Method: {retargeter_method}")
    print(f"Total: {len(files)} | Successful: {successful} | Failed: {failed}")
    print(f"Total time: {elapsed:.2f}s")
    if files:
        print(f"Avg time/file: {elapsed / len(files):.2f}s")
    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    cfg = tyro.cli(ParallelRetargetingConfig)
    main(cfg)
