"""Preprocessing utilities for the retargeting pipeline.

These are algorithm-agnostic: they operate on raw joint arrays and
derive context from MotionDataConfig (demo_joints), not from the retargeter.
"""

from __future__ import annotations

import numpy as np

from holosoma_retargeting.src.utils import (
    augment_object_poses,
    extract_foot_sticking_sequence_velocity,
    extract_object_first_moving_frame,
    transform_from_human_to_world,
)


def preprocess_motion_data(
    demo_joints: list[str],
    human_joints: np.ndarray,
    foot_names: list[str],
    scale: float = 1.0,
    mat_height: float = 0.1,
    object_poses: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, int]:
    """Normalise and scale human joint data.

    Normalises the floor height (z_min → 0), applies scale factor, and
    optionally normalises object poses. No retargeter object is required;
    the joint-name-to-index lookup uses demo_joints directly.

    Args:
        demo_joints: Ordered list of joint names matching the J axis of human_joints.
            Use MotionDataConfig.resolved_demo_joints.
        human_joints: (T, J, 3) joint positions in world frame, before scaling.
        foot_names: [left_toe_name, right_toe_name] in demo_joints ordering.
        scale: Scale factor to apply (robot_height / human_height).
        mat_height: Mat/platform height offset to subtract before z-normalisation.
        object_poses: Optional (T, 7) object poses. If provided, returned scaled.

    Returns:
        If object_poses is None: scaled human_joints (T, J, 3).
        If object_poses is provided: (human_joints, object_poses, object_moving_frame_idx).
    """
    toe_indices = [
        demo_joints.index(foot_names[0]),
        demo_joints.index(foot_names[1]),
    ]
    z_min = human_joints[:, toe_indices, 2].min()
    if z_min >= mat_height:
        z_min -= mat_height
    human_joints[:, :, 2] -= z_min
    human_joints = human_joints * scale

    if object_poses is not None:
        object_poses[:, -3:-1] = object_poses[:, -3:-1] * scale
        object_z0 = object_poses[0, -1]
        dz_scale = (object_poses[:, -1] - object_z0) * scale
        object_poses[:, -1] = object_z0 + dz_scale
        object_moving_frame_idx = extract_object_first_moving_frame(object_poses)
        return human_joints, object_poses, object_moving_frame_idx

    return human_joints


def get_foot_sticking_sequences(
    human_joints: np.ndarray,
    demo_joints: list[str],
    foot_names: list[str],
) -> list[dict[str, bool]]:
    """Extract foot-sticking contact sequences from joint velocity.

    Thin wrapper around extract_foot_sticking_sequence_velocity that
    takes demo_joints directly (no retargeter object needed).

    Args:
        human_joints: (T, J, 3) scaled joint positions.
        demo_joints: Ordered joint name list (from MotionDataConfig).
        foot_names: [left_toe_name, right_toe_name].

    Returns:
        List of T dicts: {foot_name: bool} where True = foot is sticking.
    """
    return extract_foot_sticking_sequence_velocity(human_joints, demo_joints, foot_names)


def convert_object_poses_to_mujoco_order(object_poses: np.ndarray) -> np.ndarray:
    """Convert object poses from [qw, qx, qy, qz, x, y, z] to MuJoCo [x, y, z, qw, qx, qy, qz].

    Args:
        object_poses: (T, 7) in [qw, qx, qy, qz, x, y, z] format.

    Returns:
        (T, 7) in MuJoCo order [x, y, z, qw, qx, qy, qz].
    """
    return object_poses[:, [4, 5, 6, 0, 1, 2, 3]]
