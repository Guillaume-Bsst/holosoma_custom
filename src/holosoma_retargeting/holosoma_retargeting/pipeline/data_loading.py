"""Data loading utilities for the retargeting pipeline.

These functions are algorithm-agnostic: they load raw motion data and
set up object geometry regardless of which retargeter will be used.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.src.utils import (
    calculate_scale_factor,
    create_new_scene_xml_file,
    create_scaled_multi_boxes_urdf,
    create_scaled_multi_boxes_xml,
    load_intermimic_data,
    load_object_data,
    transform_y_up_to_z_up,
)

logger = logging.getLogger(__name__)

TaskType = Literal["robot_only", "object_interaction", "climbing"]


# ------------------------------------------------------------------
# Ground grid helper
# ------------------------------------------------------------------

def create_ground_points(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    size: int,
) -> np.ndarray:
    """Create a ground point meshgrid.

    Args:
        x_range: (min, max) x-coordinate range.
        y_range: (min, max) y-coordinate range.
        size: Number of points per dimension.

    Returns:
        (N, 3) array of ground points at z=0.
    """
    x = np.linspace(x_range[0], x_range[1], size)
    y = np.linspace(y_range[0], y_range[1], size)
    X, Y = np.meshgrid(x, y)
    return np.stack([X.flatten(), Y.flatten(), np.zeros_like(X.flatten())], axis=1)


# ------------------------------------------------------------------
# Motion data loading
# ------------------------------------------------------------------

def load_motion_data(
    task_type: TaskType,
    data_format: str,
    data_path: Path,
    task_name: str,
    constants: SimpleNamespace,
    motion_data_config: MotionDataConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load motion data based on task type and format.

    Args:
        task_type: Type of task.
        data_format: Data format ("lafan", "smplh", "mocap", "smplx", ...).
        data_path: Path to data directory.
        task_name: Name of the task/sequence.
        constants: Task constants (DEMO_JOINTS, ROBOT_HEIGHT, …).
        motion_data_config: Motion data configuration.

    Returns:
        Tuple of (human_joints, object_poses, smpl_scale):
            human_joints : (T, J, 3) joint positions in world-frame.
            object_poses : (T, 7) object poses [qw, qx, qy, qz, x, y, z].
            smpl_scale   : Scaling factor (robot_height / human_height).

    Raises:
        FileNotFoundError: If required data files are not found.
    """
    logger.info("Loading motion data for task: %s, format: %s", task_name, data_format)

    if task_type == "robot_only":
        if data_format == "lafan":
            npy_path = data_path / f"{task_name}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(f"LAFAN data file not found: {npy_path}")
            human_joints = np.load(str(npy_path))
            human_joints = transform_y_up_to_z_up(human_joints)
            spine_joint_idx = constants.DEMO_JOINTS.index("Spine1")
            human_joints[:, spine_joint_idx, -1] -= 0.06
            smpl_scale = motion_data_config.default_scale_factor or 1.0

        elif data_format == "smplh":
            pt_path = data_path / f"{task_name}.pt"
            if not pt_path.exists():
                raise FileNotFoundError(f"InterMimic data file not found: {pt_path}")
            human_joints, object_poses = load_intermimic_data(str(pt_path))
            smpl_scale = calculate_scale_factor(task_name, constants.ROBOT_HEIGHT)

        elif data_format == "mocap":
            downsample = 4
            npy_file = data_path / f"{task_name}.npy"
            if not npy_file.exists():
                raise FileNotFoundError(f"MOCAP data file not found: {npy_file}")
            human_joints = np.load(str(npy_file))[::downsample]
            default_human_height = motion_data_config.default_human_height or 1.78
            smpl_scale = constants.ROBOT_HEIGHT / default_human_height

        else:
            # smplx and custom formats: expect .npz with global_joint_positions + height
            npz_file = data_path / f"{task_name}.npz"
            human_data = np.load(str(npz_file))
            human_joints = human_data["global_joint_positions"]
            human_height = human_data["height"]
            smpl_scale = constants.ROBOT_HEIGHT / human_height

        # Dummy object poses for robot_only
        num_frames = human_joints.shape[0]
        object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]], dtype=float), (num_frames, 1))

    elif task_type == "object_interaction":
        npz_path = data_path / f"{task_name}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Motion data file not found: {npz_path}")
        human_data = np.load(str(npz_path))
        human_joints = human_data["global_joint_positions"]
        human_height = human_data["height"]
        smpl_scale = constants.ROBOT_HEIGHT / human_height
        if "object_poses" not in human_data:
            raise KeyError(
                f"'object_poses' key missing in {npz_path}. "
                "Re-run prep_omomo_for_rt.py to regenerate the file."
            )
        object_poses = human_data["object_poses"]

    elif task_type == "climbing":
        task_dir = data_path / task_name
        npy_files = list(task_dir.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy file found in {task_dir}")
        downsample = 4
        human_joints = np.load(str(npy_files[0]))[::downsample]
        num_frames = human_joints.shape[0]
        object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]], dtype=float), (num_frames, 1))
        default_human_height = motion_data_config.default_human_height or 1.78
        smpl_scale = constants.ROBOT_HEIGHT / default_human_height

    else:
        raise ValueError(f"Unknown task type: {task_type}")

    logger.debug("Loaded %d frames, scale factor: %.4f", human_joints.shape[0], smpl_scale)
    return human_joints, object_poses, smpl_scale


# ------------------------------------------------------------------
# Object geometry setup
# ------------------------------------------------------------------

def setup_object_data(
    task_type: TaskType,
    constants: SimpleNamespace,
    object_dir: Path | None,
    smpl_scale: float,
    task_config: TaskConfig,
    augmentation: bool,
    object_scale_augmented: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Setup object geometry (ground mesh, object mesh, climbing terrain).

    Args:
        task_type: Type of task.
        constants: Task constants (OBJECT_NAME, OBJECT_MESH_FILE, …). May be
            mutated to update SCENE_XML_FILE for climbing tasks.
        object_dir: Directory containing object files (climbing only).
        smpl_scale: SMPL scaling factor.
        task_config: Task-specific configuration.
        augmentation: Whether this is an augmentation run.
        object_scale_augmented: Scale applied to object in augmented runs.
            Defaults to [1.0, 1.0, 1.2] (z-scale up).

    Returns:
        Tuple of (object_local_pts, object_local_pts_demo, object_urdf_path).
        object_local_pts and object_local_pts_demo are None for robot_only
        tasks (ground points are returned instead).
        object_urdf_path is None for robot_only tasks.
    """
    _object_scale_normal = np.array([1.0, 1.0, 1.0])
    if object_scale_augmented is None:
        object_scale_augmented = np.array([1.0, 1.0, 1.2])

    logger.info("Setting up object data for task: %s", task_type)

    if task_type == "robot_only":
        ground_pts = create_ground_points(
            task_config.ground_range, task_config.ground_range, task_config.ground_size
        )
        return ground_pts, ground_pts, None

    if task_type == "object_interaction":
        if constants.OBJECT_MESH_FILE is None:
            raise ValueError("OBJECT_MESH_FILE not set for object_interaction task")
        object_local_pts, object_local_pts_demo = load_object_data(
            constants.OBJECT_MESH_FILE, smpl_scale=smpl_scale, sample_count=100
        )
        return object_local_pts, object_local_pts_demo, constants.OBJECT_URDF_FILE

    if task_type == "climbing":
        if object_dir is None:
            raise ValueError("object_dir must be provided for climbing task")

        box_asset_xml = object_dir / "box_assets.xml"
        scene_xml_name = (
            Path(constants.ROBOT_URDF_FILE).name.replace(".urdf", f"_w_{constants.OBJECT_NAME}.xml")
        )
        scene_xml_file = object_dir / scene_xml_name
        constants.SCENE_XML_FILE = str(scene_xml_file)

        np.random.seed(0)
        object_local_pts, object_local_pts_demo_original = load_object_data(
            constants.OBJECT_MESH_FILE,
            smpl_scale=smpl_scale,
            surface_weights=lambda p: (
                task_config.surface_weight_high
                if p[2] > task_config.surface_weight_threshold
                else task_config.surface_weight_low
            ),
            sample_count=100,
        )

        if augmentation:
            ground_pts = create_ground_points(
                task_config.climbing_ground_range,
                task_config.climbing_ground_range,
                task_config.climbing_ground_size,
            )
            object_local_pts_demo = np.concatenate([object_local_pts_demo_original, ground_pts], axis=0)
            object_scale = object_scale_augmented
            object_local_pts = object_scale * object_local_pts_demo
        else:
            object_scale = _object_scale_normal
            object_local_pts_demo = object_local_pts_demo_original
            object_local_pts = object_local_pts_demo

        scale_factors = tuple(float(v) for v in (object_scale * smpl_scale))
        object_urdf_file = create_scaled_multi_boxes_urdf(constants.OBJECT_URDF_FILE, scale_factors)
        object_asset_xml_path = create_scaled_multi_boxes_xml(str(box_asset_xml), scale_factors)
        new_scene_xml_path = create_new_scene_xml_file(str(scene_xml_file), scale_factors, object_asset_xml_path)
        constants.SCENE_XML_FILE = new_scene_xml_path

        return object_local_pts, object_local_pts_demo, object_urdf_file

    raise ValueError(f"Unknown task type: {task_type}")
