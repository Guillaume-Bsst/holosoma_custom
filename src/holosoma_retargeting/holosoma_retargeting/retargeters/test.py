"""Test retargeter: native mink IK, no external GMR dependency.

Replicates GMR's single-pass IK logic using:
  - mink.FrameTask  (position targets only — no SMPLX rotations needed)
  - mink.solve_ik   (damped least squares QP)
  - mink.ConfigurationLimit / VelocityLimit

Input: (T, J, 3) joint positions from the holosoma pipeline.
The human-to-robot joint mapping comes from constants.JOINTS_MAPPING
(defined in config_types/data_type.py), the same table used by OmniRetarget.

This is the starting point for mixing GMR-style IK with OmniRetarget constraints.
"""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np

from holosoma_retargeting.config_types.retargeters.test import TestRetargeterConfig
from holosoma_retargeting.retargeters.base import BaseRetargeter

_OUTPUT_FPS = 30


class TestRetargeter(BaseRetargeter):
    """Position-based IK retargeter implemented natively with mink.

    Architecture
    ------------
    For each frame t:
      1. Set mink.FrameTask targets from human joint positions.
      2. Call mink.solve_ik → joint velocity v* (DLS with joint-limit constraints).
      3. Integrate: q ← q + v* · dt.
      4. Refine up to max_iter times while error decreases.

    The IK objective per frame is:
        min_v  Σᵢ position_cost · ‖ Jᵢ(q)·v·dt − (pᵢ_target − pᵢ(q)) ‖²
                + damping · ‖v‖²
        s.t.   q_min ≤ q + v·dt ≤ q_max         (ConfigurationLimit)
               |v| ≤ 3π  (if use_velocity_limit)  (VelocityLimit)

    Extension points
    ----------------
    Override _set_frame_targets() to add orientation costs (needs rotation data).
    Override _build_tasks()       to change which joints are tracked.
    Override retarget_motion()    to inject OmniRetarget post-processing.
    """

    def __init__(
        self,
        configuration,
        tasks: list,
        joint_to_task: dict[str, object],
        demo_joints: list[str],
        ik_limits: list,
        robot_name: str,
        robot_dof: int,
        robot_urdf_file: str,
        cfg: TestRetargeterConfig,
    ) -> None:
        self._configuration = configuration
        self._tasks = tasks
        self._joint_to_task = joint_to_task   # human_joint_name → mink.FrameTask
        self._demo_joints = demo_joints
        self._ik_limits = ik_limits
        self._robot_name = robot_name
        self._robot_dof = robot_dof
        self._robot_urdf_file = robot_urdf_file
        self._cfg = cfg

    # ------------------------------------------------------------------
    # BaseRetargeter interface
    # ------------------------------------------------------------------

    @property
    def supported_task_types(self) -> frozenset[str]:
        return frozenset({"robot_only"})

    @classmethod
    def from_config(cls, cfg: TestRetargeterConfig, constants: SimpleNamespace) -> "TestRetargeter":
        """Build a TestRetargeter from config + task constants.

        Loads the MuJoCo robot model, creates one mink.FrameTask per mapped
        joint pair (from constants.JOINTS_MAPPING), and wires up the IK limits.
        """
        import mink

        # Load MuJoCo model (.xml is next to .urdf)
        robot_xml = constants.ROBOT_URDF_FILE.replace(".urdf", ".xml")
        model = mujoco.MjModel.from_xml_path(robot_xml)
        configuration = mink.Configuration(model)

        # IK limits
        ik_limits = [mink.ConfigurationLimit(model)]
        if cfg.use_velocity_limit:
            motor_names = {
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                for i in range(model.nu)
            }
            ik_limits.append(mink.VelocityLimit(model, {n: 3 * np.pi for n in motor_names}))

        # Build one FrameTask per human→robot joint pair
        tasks = []
        joint_to_task: dict[str, object] = {}

        for human_name, robot_body_name in constants.JOINTS_MAPPING.items():
            task = mink.FrameTask(
                frame_name=robot_body_name,
                frame_type="body",
                position_cost=cfg.position_cost,
                orientation_cost=0.0,   # position only — no SMPLX rotations available
                lm_damping=1,
            )
            tasks.append(task)
            joint_to_task[human_name] = task

        return cls(
            configuration=configuration,
            tasks=tasks,
            joint_to_task=joint_to_task,
            demo_joints=constants.DEMO_JOINTS,
            ik_limits=ik_limits,
            robot_name=constants.ROBOT_NAME,
            robot_dof=constants.ROBOT_DOF,
            robot_urdf_file=constants.ROBOT_URDF_FILE,
            cfg=cfg,
        )

    # ------------------------------------------------------------------
    # Core retargeting
    # ------------------------------------------------------------------

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
        """Retarget (T, J, 3) joint positions frame-by-frame with mink IK."""
        import mink

        num_frames = human_joint_motions.shape[0]
        dt = self._configuration.model.opt.timestep
        qpos_list = []

        for i in range(num_frames):
            self._set_frame_targets(human_joint_motions[i])
            self._solve_with_refinement(dt)
            qpos_list.append(self._configuration.data.qpos.copy())

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

    # ------------------------------------------------------------------
    # Extension points
    # ------------------------------------------------------------------

    def _set_frame_targets(self, joint_positions: np.ndarray) -> None:
        """Set mink FrameTask targets for one frame.

        Args:
            joint_positions: (J, 3) joint positions for the current frame.

        Override this to add orientation targets or apply per-body offsets.
        """
        import mink

        for human_name, task in self._joint_to_task.items():
            idx = self._demo_joints.index(human_name)
            pos = joint_positions[idx]
            # Position-only target: identity orientation
            task.set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3.identity(), pos)
            )

    def _solve_with_refinement(self, dt: float) -> None:
        """Run IK with adaptive refinement until convergence or max_iter.

        Override to change the solving strategy (e.g. two-pass like GMR).
        """
        import mink

        curr_error = self._total_error()
        vel = mink.solve_ik(
            self._configuration, self._tasks, dt, self._cfg.solver, self._cfg.damping, self._ik_limits
        )
        self._configuration.integrate_inplace(vel, dt)
        next_error = self._total_error()

        for _ in range(self._cfg.max_iter):
            if curr_error - next_error <= self._cfg.convergence_threshold:
                break
            curr_error = next_error
            vel = mink.solve_ik(
                self._configuration, self._tasks, dt, self._cfg.solver, self._cfg.damping, self._ik_limits
            )
            self._configuration.integrate_inplace(vel, dt)
            next_error = self._total_error()

    def _total_error(self) -> float:
        """L2 norm of all task errors at current configuration."""
        errors = np.concatenate(
            [task.compute_error(self._configuration) for task in self._tasks]
        )
        return float(np.linalg.norm(errors))
