"""Configuration types for retargeting (top-level config)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeter import RetargeterConfig  # backward compat alias
from holosoma_retargeting.config_types.retargeters.gmr import GMRRetargeterConfig
from holosoma_retargeting.config_types.retargeters.omniretarget import OmniRetargeterConfig
from holosoma_retargeting.config_types.retargeters.test import TestRetargeterConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig

try:
    from holosoma_data import DATASETS_DIR as _hd_datasets_dir
    _DEFAULT_DATA_PATH = _hd_datasets_dir / "OMOMO"
except ImportError:
    _DEFAULT_DATA_PATH = Path("demo_data/OMOMO_new")


@dataclass
class RetargetingConfig:
    """Top-level retargeting configuration used by the Tyro CLI.

    This combines all configuration types needed for retargeting.

    Algorithm selection
    -------------------
    Set ``retargeter_method`` to choose the algorithm:
        - ``"omniretarget"`` (default): Interaction Mesh + SQP. Supports all task types.
        - ``"gmr"``: IK-based (mink/mujoco). Supports robot_only only.

    Each method has its own nested config namespace:
        - OmniRetarget params: ``--retargeter.<param>``
        - GMR params:          ``--gmr.<param>``
    """

    # --- Task type selection ---
    task_type: Literal["robot_only", "object_interaction", "climbing"] = "object_interaction"
    """Type of retargeting task."""

    # --- Algorithm selection ---
    retargeter_method: Literal["omniretarget", "gmr", "test"] = "omniretarget"
    """Retargeting algorithm to use.
    'omniretarget': Interaction Mesh + SQP (original, all task types).
    'gmr':          IK-based General Motion Retargeting (robot_only only).
    'test':         Native mink IK in holosoma (robot_only, no GMR dependency)."""

    # --- top-level run knobs ---
    robot: str = "g1"
    """Robot type. Use str to allow dynamic robot types via _ROBOT_DEFAULTS."""

    data_format: str | None = None
    """Motion data format. Auto-determined by task_type if None.
    Can be any format registered in DEMO_JOINTS_REGISTRY
    (e.g., 'lafan', 'smplh', 'mocap', 'smplx', or custom formats)."""

    task_name: str = "sub3_largebox_003"
    """Name of the task/sequence."""

    data_path: Path = field(default_factory=lambda: _DEFAULT_DATA_PATH)
    """Path to data directory."""

    save_dir: Path | None = None
    """Directory to save results. Auto-determined if None."""

    augmentation: bool = False
    """Whether to use augmentation."""

    # --- Nested configs ---
    robot_config: RobotConfig = field(default_factory=lambda: RobotConfig(robot_type="g1"))
    """Robot configuration (nested - can override robot_urdf_file, robot_dof, etc.
    via --robot-config.robot-urdf-file)."""

    motion_data_config: MotionDataConfig = field(
        default_factory=lambda: MotionDataConfig(data_format="smplh", robot_type="g1")
    )
    """Motion data configuration (nested - can override demo_joints, joints_mapping, etc.
    via --motion-data-config.demo-joints).
    Note: data_format default will be set based on task_type in main()."""

    task_config: TaskConfig = field(default_factory=TaskConfig)
    """Task-specific configuration (nested - can override ground_size, surface_weight_threshold, etc.
    via --task-config.ground-size)."""

    retargeter: OmniRetargeterConfig = field(default_factory=OmniRetargeterConfig)
    """OmniRetarget (Interaction Mesh + SQP) configuration.
    Only used when retargeter_method='omniretarget'.
    Override via --retargeter.<param>."""

    gmr: GMRRetargeterConfig = field(default_factory=GMRRetargeterConfig)
    """GMR (IK-based) configuration.
    Only used when retargeter_method='gmr'.
    Override via --gmr.<param>."""

    test: TestRetargeterConfig = field(default_factory=TestRetargeterConfig)
    """Test (native mink IK) configuration.
    Only used when retargeter_method='test'.
    Override via --test.<param>."""


@dataclass
class ParallelRetargetingConfig(RetargetingConfig):
    """Extended retargeting config for parallel processing.

    Adds parallel-specific fields while inheriting all retargeting config fields.
    This config is used for processing multiple files in parallel.
    """

    # Parallel processing specific fields
    data_dir: Path = field(default_factory=lambda: _DEFAULT_DATA_PATH)
    """Directory containing input data files for parallel processing.
    This overrides data_path from RetargetingConfig when processing multiple files."""

    max_workers: int | None = None
    """Maximum number of parallel workers. Auto-determined if None."""
