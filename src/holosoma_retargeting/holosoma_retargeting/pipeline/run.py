"""Algorithm-agnostic retargeting pipeline orchestration.

This module is the single entry point for running retargeting. It:
  1. Loads and validates configuration.
  2. Loads motion data (format-agnostic).
  3. Sets up object geometry.
  4. Builds the retargeter via the registry (algorithm selected by cfg.retargeter_method).
  5. Preprocesses human joints (decoupled from the retargeter).
  6. Initialises the robot pose.
  7. Extracts foot-sticking sequences.
  8. Calls retargeter.retarget_motion().

Adding a new algorithm: see retargeters/registry.py.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeting import RetargetingConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.pipeline.data_loading import load_motion_data, setup_object_data
from holosoma_retargeting.pipeline.preprocessing import (
    convert_object_poses_to_mujoco_order,
    get_foot_sticking_sequences,
    preprocess_motion_data,
)
from holosoma_retargeting.retargeters.registry import build_retargeter
from holosoma_retargeting.src.utils import (
    augment_object_poses,
    estimate_human_orientation,
    extract_object_first_moving_frame,
    transform_from_human_to_world,
)

logger = logging.getLogger(__name__)

TaskType = Literal["robot_only", "object_interaction", "climbing"]

# Default save-dir template: <pipeline_data_dir>/<robot_name>/<method>/<task_name>/
_PIPELINE_DATA_DIR = Path(__file__).resolve().parents[3] / "holosoma" / "holosoma" / "data" / "pipeline"
_SAVE_DIR_TEMPLATE = "{robot_name}/{retargeter_method}/{task_name}"

# Constants for augmentation
_OBJECT_SCALE_AUGMENTED = np.array([1.0, 1.0, 1.2])
_AUGMENTATION_TRANSLATION = np.array([0.2, 0.0, 0.0])


# ------------------------------------------------------------------
# Public helpers (reusable from parallel_robot_retarget.py)
# ------------------------------------------------------------------

def create_task_constants(
    robot_config: RobotConfig,
    motion_data_config: MotionDataConfig,
    task_config: TaskConfig,
    task_type: str,
) -> SimpleNamespace:
    """Build a mutable SimpleNamespace of task constants for the pipeline.

    Combines robot config, motion data config, and task type into a single
    namespace that retargeters can read via uppercase attribute names.

    Args:
        robot_config: Robot configuration.
        motion_data_config: Motion data format configuration.
        task_config: Task-specific configuration.
        task_type: One of "robot_only", "object_interaction", "climbing".

    Returns:
        SimpleNamespace with all task constants.
    """
    task_constants = SimpleNamespace()

    for attr in dir(robot_config):
        if attr.isupper() and not attr.startswith("_"):
            setattr(task_constants, attr, getattr(robot_config, attr))

    for attr, value in motion_data_config.legacy_constants().items():
        setattr(task_constants, attr, value)

    # Store task_type so retargeters can read it via from_config
    task_constants._task_type = task_type

    if task_type == "robot_only":
        obj_name = task_config.object_name or "ground"
        task_constants.OBJECT_NAME = obj_name
        task_constants.OBJECT_URDF_FILE = None
        task_constants.OBJECT_MESH_FILE = None

    elif task_type == "object_interaction":
        obj_name = task_config.object_name or "largebox"
        task_constants.OBJECT_NAME = obj_name
        task_constants.OBJECT_URDF_FILE = f"models/{obj_name}/{obj_name}.urdf"
        task_constants.OBJECT_MESH_FILE = f"models/{obj_name}/{obj_name}.obj"
        task_constants.OBJECT_URDF_TEMPLATE = f"models/templates/{obj_name}.urdf.jinja"

    elif task_type == "climbing":
        obj_name = task_config.object_name or "multi_boxes"
        task_constants.OBJECT_NAME = obj_name
        object_dir = task_config.object_dir
        task_constants.OBJECT_DIR = str(object_dir) if object_dir else ""
        task_constants.OBJECT_URDF_FILE = (
            str(object_dir / f"{obj_name}.urdf") if object_dir else f"{obj_name}.urdf"
        )
        task_constants.OBJECT_MESH_FILE = (
            str(object_dir / f"{obj_name}.obj") if object_dir else f"{obj_name}.obj"
        )
        task_constants.SCENE_XML_FILE = ""

    return task_constants


def determine_output_path(task_type: TaskType, save_dir: Path, task_name: str, augmentation: bool) -> str:
    """Determine the output .npz path.

    Args:
        task_type: Task type.
        save_dir: Save directory (already includes robot_name/method/task_name).
        task_name: Task name.
        augmentation: Whether this is an augmented run.

    Returns:
        Absolute path string for the output .npz file.
    """
    if task_type == "robot_only":
        return str(save_dir / "retargeted.npz")
    if task_type in ("object_interaction", "climbing"):
        suffix = "_augmented" if augmentation else "_original"
        return str(save_dir / f"retargeted_w_obj{suffix}.npz")
    raise ValueError(f"Unknown task type: {task_type}")


def validate_config(cfg: RetargetingConfig) -> None:
    """Validate configuration consistency.

    Raises:
        ValueError: If configuration is invalid.
    """
    from holosoma_retargeting.config_types.data_type import DEMO_JOINTS_REGISTRY

    if cfg.data_format is not None and cfg.data_format not in DEMO_JOINTS_REGISTRY:
        available = ", ".join(sorted(DEMO_JOINTS_REGISTRY.keys()))
        raise ValueError(
            f"Unknown data_format: '{cfg.data_format}'. "
            f"Available formats: {available}. "
            f"Add your format to DEMO_JOINTS_REGISTRY in config_types/data_type.py"
        )
    if cfg.task_type == "climbing" and cfg.data_format not in (None, "mocap"):
        raise ValueError("Climbing task requires 'mocap' data format")
    if cfg.task_type == "object_interaction" and cfg.data_format not in (None, "smplh"):
        raise ValueError("Object interaction requires 'smplh' data format")


# ------------------------------------------------------------------
# Robot pose initialisation (decoupled from retargeter object)
# ------------------------------------------------------------------

def _compute_q_init_base(
    task_type: TaskType,
    data_format: str,
    human_joints: np.ndarray,
    object_poses: np.ndarray,
    constants: SimpleNamespace,
    demo_joints: list[str],
) -> np.ndarray:
    """Compute initial robot pose (q_init) in MuJoCo order.

    Args:
        task_type: Type of task.
        data_format: Data format name.
        human_joints: (T, J, 3) preprocessed joint positions.
        object_poses: (T, 7) object poses in [qw,qx,qy,qz,x,y,z].
        constants: Task constants.
        demo_joints: Ordered joint name list (from MotionDataConfig).

    Returns:
        (7 + robot_dof,) initial qpos in MuJoCo order [pos(3), quat(4), dof(N)].
    """
    if task_type == "robot_only":
        if data_format == "lafan":
            spine_joint_idx = demo_joints.index("Spine1")
            human_quat_init = estimate_human_orientation(human_joints, demo_joints)
            return np.concatenate(
                [human_joints[0, spine_joint_idx, :3], human_quat_init, np.zeros(constants.ROBOT_DOF)]
            )
        else:
            _, human_quat_init = transform_from_human_to_world(
                human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
            )
            return np.concatenate([human_joints[0, 0, :3], human_quat_init, np.zeros(constants.ROBOT_DOF)])

    if task_type == "object_interaction":
        _, human_quat_init = transform_from_human_to_world(
            human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
        )
        return np.concatenate([human_joints[0, 0, :3], human_quat_init, np.zeros(constants.ROBOT_DOF)])

    if task_type == "climbing":
        _, human_quat_init = transform_from_human_to_world(
            human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
        )
        spine_joint_idx = demo_joints.index("Spine1")
        return np.concatenate(
            [human_joints[0, spine_joint_idx], human_quat_init, np.zeros(constants.ROBOT_DOF)]
        )

    raise ValueError(f"Unknown task type: {task_type}")


def initialize_robot_pose(
    task_type: TaskType,
    data_format: str,
    human_joints: np.ndarray,
    object_poses: np.ndarray,
    constants: SimpleNamespace,
    demo_joints: list[str],
    task_config: TaskConfig,
    augmentation: bool,
    save_dir: Path,
    task_name: str,
    augmentation_translation: np.ndarray | None = None,
    augmentation_rotation: float = 0.0,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    """Initialise robot pose (q_init, q_nominal) and finalize object poses.

    This function replaces the old version that took a retargeter instance.
    demo_joints is now passed directly from MotionDataConfig.

    Args:
        task_type: Type of task.
        data_format: Data format name.
        human_joints: (T, J, 3) preprocessed joint positions.
        object_poses: (T, 7) object poses in [qw,qx,qy,qz,x,y,z].
        constants: Task constants.
        demo_joints: Ordered joint name list (from MotionDataConfig).
        task_config: Task-specific configuration.
        augmentation: Whether this is an augmentation run.
        save_dir: Save directory.
        task_name: Task name.
        augmentation_translation: 3D translation for object augmentation.
        augmentation_rotation: Rotation angle (rad) for object augmentation.

    Returns:
        (q_init, q_nominal, object_poses_augmented, human_joints, object_poses).
    """
    if augmentation_translation is None:
        augmentation_translation = _AUGMENTATION_TRANSLATION

    if task_type == "robot_only":
        q_init = _compute_q_init_base(task_type, data_format, human_joints, object_poses, constants, demo_joints)
        object_poses = convert_object_poses_to_mujoco_order(object_poses)
        return q_init, None, object_poses, human_joints, object_poses

    if task_type == "object_interaction":
        if augmentation:
            object_moving_frame_idx = extract_object_first_moving_frame(object_poses)
            object_poses_augmented = augment_object_poses(
                object_poses,
                object_moving_frame_idx,
                human_joints[0, 0, :],
                augmentation_translation,
                augmentation_rotation,
            )
            object_poses_augmented = convert_object_poses_to_mujoco_order(object_poses_augmented)
            object_poses = convert_object_poses_to_mujoco_order(object_poses)
            original_path = save_dir / "retargeted_w_obj_original.npz"
            if not original_path.exists():
                raise FileNotFoundError(
                    f"Original file not found: {original_path}. Run without --augmentation first."
                )
            data = np.load(str(original_path))
            q_nominal = data["qpos"]
            return q_nominal[0], q_nominal, object_poses_augmented, human_joints, object_poses

        q_init = _compute_q_init_base(task_type, data_format, human_joints, object_poses, constants, demo_joints)
        object_poses = convert_object_poses_to_mujoco_order(object_poses)
        object_poses_augmented = object_poses.copy()
        return q_init, None, object_poses_augmented, human_joints, object_poses

    if task_type == "climbing":
        if augmentation:
            original_path = save_dir / "retargeted_w_obj_original.npz"
            if not original_path.exists():
                raise FileNotFoundError(
                    f"Original file not found: {original_path}. Run without --augmentation first."
                )
            data = np.load(str(original_path))
            q_nominal = data["qpos"]
            object_poses = convert_object_poses_to_mujoco_order(object_poses)
            return q_nominal[0], q_nominal, object_poses, human_joints, object_poses

        q_init = _compute_q_init_base(task_type, data_format, human_joints, object_poses, constants, demo_joints)
        object_poses = convert_object_poses_to_mujoco_order(object_poses)
        return q_init, None, object_poses, human_joints, object_poses

    raise ValueError(f"Unknown task type: {task_type}")


# ------------------------------------------------------------------
# Main pipeline entry point
# ------------------------------------------------------------------

def run_retargeting(cfg: RetargetingConfig) -> None:
    """Main retargeting pipeline — algorithm-agnostic.

    Selects the retargeter via cfg.retargeter_method and delegates the
    frame-level computation to it. Everything else (data loading, preprocessing,
    pose init, foot sticking) is algorithm-independent and runs here.

    Args:
        cfg: Top-level retargeting configuration (from tyro CLI or Python).
    """
    validate_config(cfg)

    robot = cfg.robot
    task_name = cfg.task_name
    task_type = cfg.task_type
    retargeter_method = cfg.retargeter_method
    robot_name = cfg.robot_config.ROBOT_NAME

    data_format: str = cfg.data_format or _DEFAULT_DATA_FORMATS[task_type]

    if cfg.save_dir is not None:
        save_dir = cfg.save_dir
    else:
        save_dir = _PIPELINE_DATA_DIR / _SAVE_DIR_TEMPLATE.format(
            robot_name=robot_name,
            retargeter_method=retargeter_method,
            task_name=task_name,
        )
    data_path = cfg.data_path
    os.makedirs(save_dir, exist_ok=True)

    logger.info("Task: %s | Type: %s | Format: %s | Method: %s", task_name, task_type, data_format, retargeter_method)
    logger.info("Data path: %s | Save dir: %s", data_path, save_dir)

    # Sync nested configs if top-level flags differ
    if cfg.robot_config.robot_type != robot:
        cfg.robot_config = RobotConfig(robot_type=robot)
    if cfg.motion_data_config.robot_type != robot or cfg.motion_data_config.data_format != data_format:
        cfg.motion_data_config = MotionDataConfig(data_format=data_format, robot_type=robot)
    if task_type == "climbing" and cfg.task_config.object_dir is None:
        cfg.task_config = replace(cfg.task_config, object_dir=data_path / task_name)

    # Build constants namespace
    constants = create_task_constants(cfg.robot_config, cfg.motion_data_config, cfg.task_config, task_type)
    demo_joints: list[str] = cfg.motion_data_config.resolved_demo_joints
    toe_names: list[str] = cfg.motion_data_config.toe_names

    # Load motion data
    human_joints, object_poses, smpl_scale = load_motion_data(
        task_type, data_format, data_path, task_name, constants, cfg.motion_data_config
    )

    # Setup object geometry
    object_local_pts, object_local_pts_demo, object_urdf_path = setup_object_data(
        task_type,
        constants,
        cfg.task_config.object_dir,
        smpl_scale,
        cfg.task_config,
        cfg.augmentation,
        object_scale_augmented=_OBJECT_SCALE_AUGMENTED,
    )

    # Set object_urdf_path in constants so retargeters can access it
    if object_urdf_path is not None:
        constants.OBJECT_URDF_FILE = object_urdf_path

    # Build retargeter via factory — only place the method name is resolved
    retargeter_cfg = _get_retargeter_cfg(cfg, retargeter_method)
    retargeter = build_retargeter(retargeter_method, retargeter_cfg, constants)

    # Validate task type support
    if task_type not in retargeter.supported_task_types:
        raise ValueError(
            f"Retargeter '{retargeter_method}' does not support task type '{task_type}'. "
            f"Supported: {sorted(retargeter.supported_task_types)}"
        )

    # Preprocess motion data (demo_joints replaces retargeter.demo_joints)
    if task_type == "robot_only":
        human_joints = preprocess_motion_data(demo_joints, human_joints, toe_names, smpl_scale)
    else:
        human_joints, object_poses, _ = preprocess_motion_data(
            demo_joints, human_joints, toe_names, scale=smpl_scale, object_poses=object_poses
        )

    # Initialise robot pose (demo_joints replaces retargeter dependency)
    q_init, q_nominal, object_poses_augmented, human_joints, object_poses = initialize_robot_pose(
        task_type,
        data_format,
        human_joints,
        object_poses,
        constants,
        demo_joints,
        cfg.task_config,
        cfg.augmentation,
        save_dir,
        task_name,
        augmentation_translation=_AUGMENTATION_TRANSLATION,
    )

    # Foot sticking sequences
    foot_sticking_sequences = get_foot_sticking_sequences(human_joints, demo_joints, toe_names)
    if task_type == "object_interaction":
        foot_sticking_sequences[0][toe_names[0]] = False
        foot_sticking_sequences[0][toe_names[1]] = False

    # Output path
    dest_res_path = determine_output_path(task_type, save_dir, task_name, cfg.augmentation)

    # Run retargeting
    logger.info("Starting retargeting with method '%s'...", retargeter_method)
    retargeter.retarget_motion(
        human_joint_motions=human_joints,
        object_poses=object_poses,
        object_poses_augmented=object_poses_augmented,
        object_points_local_demo=object_local_pts_demo,
        object_points_local=object_local_pts,
        foot_sticking_sequences=foot_sticking_sequences,
        q_a_init=q_init,
        q_nominal_list=q_nominal,
        original=not cfg.augmentation,
        dest_res_path=dest_res_path,
    )
    logger.info("Retargeting complete. Results saved to: %s", dest_res_path)

    # Debug pause (OmniRetarget only)
    if retargeter_method == "omniretarget" and hasattr(retargeter_cfg, "debug") and retargeter_cfg.debug:
        input("Press Enter to exit ...")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

_DEFAULT_DATA_FORMATS: dict[str, str] = {
    "robot_only": "smplh",
    "object_interaction": "smplh",
    "climbing": "mocap",
}


def _get_retargeter_cfg(cfg: RetargetingConfig, method: str):
    """Return the algorithm-specific config from the top-level RetargetingConfig."""
    if method == "omniretarget":
        return cfg.retargeter
    if method == "gmr":
        return cfg.gmr
    if method == "test":
        return cfg.test
    raise ValueError(
        f"Unknown retargeter method '{method}'. "
        f"Add a cfg field and a branch here in pipeline/run.py._get_retargeter_cfg()."
    )
