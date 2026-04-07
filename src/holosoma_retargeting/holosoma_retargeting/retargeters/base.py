"""Base interface that every retargeting algorithm must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import SimpleNamespace

import numpy as np


class BaseRetargeter(ABC):
    """Common interface for all retargeting algorithms.

    A retargeter receives preprocessed human motion data (already scaled,
    z-normalised, in the pipeline's joint format) and writes the retargeted
    robot qpos to disk as a .npz file.

    Pipeline contract
    -----------------
    Input joint array ``human_joint_motions`` is always shape ``(T, J, 3)``
    in world-frame Z-up coordinates, scaled to robot proportions.

    Output .npz must contain at minimum:
        qpos               : (T, 7 + robot_dof)  — MuJoCo order [pos(3), quat_wxyz(4), dof]
        fps                : scalar
        _meta_robot_dof    : scalar int
        _meta_robot_name   : str
        _meta_step         : "retargeting"

    Adding a new algorithm
    ----------------------
    1. Create ``config_types/retargeters/<name>.py`` with a frozen dataclass.
    2. Create ``retargeters/<name>.py`` subclassing BaseRetargeter.
    3. Register in ``retargeters/registry.py``.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def supported_task_types(self) -> frozenset[str]:
        """Task types this algorithm can handle.

        Must be a subset of {"robot_only", "object_interaction", "climbing"}.
        The pipeline validates this before calling retarget_motion.

        Example:
            return frozenset({"robot_only"})  # GMR: robot-only
            return frozenset({"robot_only", "object_interaction", "climbing"})  # OmniRetarget
        """
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, cfg, constants: SimpleNamespace) -> "BaseRetargeter":
        """Construct a retargeter from its algorithm-specific config dataclass.

        Args:
            cfg: The algorithm config (e.g. OmniRetargeterConfig, GMRRetargeterConfig).
            constants: SimpleNamespace built by pipeline.run.create_task_constants().
                       Provides ROBOT_DOF, ROBOT_NAME, ROBOT_URDF_FILE, JOINTS_MAPPING, etc.

        Returns:
            Initialised retargeter instance.
        """
        ...

    @abstractmethod
    def retarget_motion(
        self,
        human_joint_motions: np.ndarray,
        object_poses: np.ndarray,
        object_poses_augmented: np.ndarray,
        object_points_local_demo: np.ndarray | None,
        object_points_local: np.ndarray | None,
        foot_sticking_sequences: list[dict[str, bool]],
        q_a_init: np.ndarray | None,
        q_nominal_list: np.ndarray | None,
        original: bool,
        dest_res_path: str,
    ) -> None:
        """Retarget a full motion sequence and save the result to disk.

        Args:
            human_joint_motions: (T, J, 3) joint positions, world-frame Z-up,
                scaled to robot proportions, z-normalised (floor at 0).
            object_poses: (T, 7) object poses in MuJoCo order [x,y,z, qw,qx,qy,qz].
                For robot_only tasks, all rows are identity [0,0,0, 1,0,0,0].
            object_poses_augmented: (T, 7) augmented object poses (same as object_poses
                for non-augmented runs or robot_only tasks).
            object_points_local_demo: (N, 3) demo object surface points in local frame.
                None for algorithms that don't use interaction mesh (e.g. GMR).
            object_points_local: (N, 3) current object surface points in local frame.
                None for algorithms that don't use interaction mesh (e.g. GMR).
            foot_sticking_sequences: list of T dicts mapping foot-joint-name → bool.
                True means the foot should be sticking (velocity-based heuristic).
                Algorithms that don't enforce foot sticking can ignore this.
            q_a_init: (7 + robot_dof,) initial robot configuration in MuJoCo order.
                None to use robot's default pose.
            q_nominal_list: (T, 7 + robot_dof) nominal configurations for each frame.
                Used by augmentation runs to stay close to a reference trajectory.
                None for first-pass / original runs.
            original: True for first-pass (non-augmented) runs. Controls some
                algorithm-specific behaviours (e.g. nominal tracking decay).
            dest_res_path: Absolute path for the output .npz file.
        """
        ...
