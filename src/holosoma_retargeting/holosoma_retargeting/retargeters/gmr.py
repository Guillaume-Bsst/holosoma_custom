"""GMR retargeter: wraps General Motion Retargeting (IK-based).

GMR uses mink/mujoco IK and currently supports robot_only tasks only.
See: https://github.com/YanjieZe/GMR

Input format: human_joint_motions (T, J, 3) in pipeline's joint order.
GMR internally uses a dict-per-frame format; this adapter converts between them.

The joint mapping from pipeline format to GMR body names is defined by
the IK config in GMR (smplx_to_<robot>.json). The adapter extracts body
positions from the joint array using the demo_joints ordering from MotionDataConfig.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from holosoma_retargeting.config_types.retargeters.gmr import GMRRetargeterConfig
from holosoma_retargeting.retargeters.base import BaseRetargeter


# Mapping from pipeline SMPLX joint names to GMR SMPLX body names.
# GMR uses the SMPLX body model naming convention.
_PIPELINE_TO_GMR_JOINT_NAMES: dict[str, str] = {
    "Pelvis": "pelvis",
    "L_Hip": "left_hip",
    "R_Hip": "right_hip",
    "Spine1": "spine1",
    "L_Knee": "left_knee",
    "R_Knee": "right_knee",
    "Spine2": "spine2",
    "L_Ankle": "left_ankle",
    "R_Ankle": "right_ankle",
    "Spine3": "spine3",
    "L_Foot": "left_foot",
    "R_Foot": "right_foot",
    "Neck": "neck",
    "L_Collar": "left_collar",
    "R_Collar": "right_collar",
    "Head": "head",
    "L_Shoulder": "left_shoulder",
    "R_Shoulder": "right_shoulder",
    "L_Elbow": "left_elbow",
    "R_Elbow": "right_elbow",
    "L_Wrist": "left_wrist",
    "R_Wrist": "right_wrist",
}

# FPS used when saving the output (matches GMR's default)
_OUTPUT_FPS = 30


class GMRRetargeter(BaseRetargeter):
    """IK-based retargeter using GMR (General Motion Retargeting).

    Only supports robot_only tasks. Does not use interaction mesh, foot sticking,
    object interaction, or augmentation.
    """

    def __init__(
        self,
        gmr_instance,
        demo_joints: list[str],
        robot_name: str,
        robot_dof: int,
        robot_urdf_file: str,
    ) -> None:
        self._gmr = gmr_instance
        self._demo_joints = demo_joints
        self._robot_name = robot_name
        self._robot_dof = robot_dof
        self._robot_urdf_file = robot_urdf_file

    @property
    def supported_task_types(self) -> frozenset[str]:
        return frozenset({"robot_only"})

    @classmethod
    def from_config(cls, cfg: GMRRetargeterConfig, constants: SimpleNamespace) -> "GMRRetargeter":
        try:
            from general_motion_retargeting import GeneralMotionRetargeting as GMR
        except ImportError as e:
            raise ImportError(
                "GMR is not installed. Install it with: pip install -e /path/to/GMR\n"
                "See https://github.com/YanjieZe/GMR"
            ) from e

        # Map holosoma robot_type to GMR robot name
        # GMR uses slightly different naming conventions
        robot_type = constants.ROBOT_NAME.split("_")[0] + "_" + constants.ROBOT_NAME.split("_")[1]
        # e.g. "g1_29dof" → "unitree_g1"
        _HOLOSOMA_TO_GMR_ROBOT: dict[str, str] = {
            "g1": "unitree_g1",
            "t1": "booster_t1",
        }
        robot_key = constants.ROBOT_NAME.split("_")[0]
        tgt_robot = _HOLOSOMA_TO_GMR_ROBOT.get(robot_key)
        if tgt_robot is None:
            available = list(_HOLOSOMA_TO_GMR_ROBOT.keys())
            raise ValueError(
                f"GMR does not have a mapping for robot '{robot_key}'. "
                f"Add it to _HOLOSOMA_TO_GMR_ROBOT in retargeters/gmr.py. "
                f"Currently mapped: {available}"
            )

        gmr_instance = GMR(
            src_human=cfg.src_human,
            tgt_robot=tgt_robot,
            actual_human_height=cfg.actual_human_height,
            solver=cfg.solver,
            damping=cfg.damping,
            use_velocity_limit=cfg.use_velocity_limit,
        )

        return cls(
            gmr_instance=gmr_instance,
            demo_joints=constants.DEMO_JOINTS,
            robot_name=constants.ROBOT_NAME,
            robot_dof=constants.ROBOT_DOF,
            robot_urdf_file=constants.ROBOT_URDF_FILE,
        )

    def _frame_to_gmr_format(self, joint_positions: np.ndarray) -> dict:
        """Convert a single frame (J, 3) to the GMR per-frame dict format.

        GMR expects: {body_name: (translation_3D, rotation_quaternion_wxyz)}
        For the pipeline → GMR bridge, we provide positions as translations
        with identity rotations. GMR's IK solver computes the rotations itself.

        Args:
            joint_positions: (J, 3) joint positions for one frame.

        Returns:
            Dict mapping GMR body names to (pos, rot) tuples.
        """
        frame_data = {}
        identity_rot = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz identity quaternion
        for pipeline_name, gmr_name in _PIPELINE_TO_GMR_JOINT_NAMES.items():
            if pipeline_name in self._demo_joints:
                idx = self._demo_joints.index(pipeline_name)
                pos = joint_positions[idx]
                frame_data[gmr_name] = (pos, identity_rot)
        return frame_data

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
        """Retarget frame by frame using GMR IK solver."""
        num_frames = human_joint_motions.shape[0]
        qpos_list = []

        for i in range(num_frames):
            frame_data = self._frame_to_gmr_format(human_joint_motions[i])
            qpos = self._gmr.retarget(frame_data)
            qpos_list.append(qpos)

        qpos_array = np.array(qpos_list)  # (T, 7 + robot_dof)

        np.savez(
            dest_res_path,
            qpos=qpos_array,
            fps=_OUTPUT_FPS,
            _meta_robot_name=self._robot_name,
            _meta_robot_dof=self._robot_dof,
            _meta_robot_urdf=self._robot_urdf_file,
            _meta_step="retargeting",
        )
