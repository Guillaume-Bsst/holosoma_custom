"""holosoma_data — centralized data store for all Holosoma shared assets.

Provides a single source of truth for:
  - robots/    : URDF + mesh files for all supported robots
  - objects/   : URDF + mesh files for interactive objects
  - datasets/  : raw source motion datasets (OMOMO, SFU, climb, ...)
  - policies/  : trained ONNX policy weights (loco, wbt, ...)
  - pipeline/  : outputs from retargeting and data conversion steps
    - retargeted/ : .npz files output by holosoma_retargeting
    - converted/  : .npz files output by data_conversion step
"""

from __future__ import annotations

from pathlib import Path

# Absolute path to the holosoma_data package root.
# Use this constant instead of hard-coded relative paths.
HOLOSOMA_DATA_ROOT: Path = Path(__file__).resolve().parent

ROBOTS_DIR: Path = HOLOSOMA_DATA_ROOT / "robots"
OBJECTS_DIR: Path = HOLOSOMA_DATA_ROOT / "objects"
DATASETS_DIR: Path = HOLOSOMA_DATA_ROOT / "datasets"
POLICIES_DIR: Path = HOLOSOMA_DATA_ROOT / "policies"
PIPELINE_DIR: Path = HOLOSOMA_DATA_ROOT / "pipeline"
RETARGETED_DIR: Path = PIPELINE_DIR / "retargeted"
CONVERTED_DIR: Path = PIPELINE_DIR / "converted"


def robot_urdf(robot_type: str, dof: int, variant: str = "retargeting") -> Path:
    """Return the URDF path for a robot variant.

    Args:
        robot_type: e.g. "g1", "t1"
        dof: number of degrees of freedom, e.g. 29 or 27
        variant: "retargeting" (default) or "training"

    Returns:
        Absolute path to the URDF file.

    Example:
        >>> robot_urdf("g1", 29)
        PosixPath('.../holosoma_data/robots/g1/g1_29dof_retargeting.urdf')
        >>> robot_urdf("g1", 29, "training")
        PosixPath('.../holosoma_data/robots/g1/g1_29dof_training.urdf')
    """
    return ROBOTS_DIR / robot_type / f"{robot_type}_{dof}dof_{variant}.urdf"


def object_urdf(object_name: str) -> Path:
    """Return the canonical URDF path for an object.

    Args:
        object_name: e.g. "largebox"

    Returns:
        Absolute path to the URDF file.
    """
    return OBJECTS_DIR / object_name / f"{object_name}.urdf"
