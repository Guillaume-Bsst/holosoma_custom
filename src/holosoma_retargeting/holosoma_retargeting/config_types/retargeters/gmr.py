"""Configuration for the GMR (General Motion Retargeting) method.

GMR uses an IK-based solver (mink/mujoco) and supports robot_only tasks.
See: https://github.com/YanjieZe/GMR

Requires GMR to be installed:
    pip install -e /path/to/GMR
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GMRRetargeterConfig:
    """Configuration for GMR (General Motion Retargeting, IK-based).

    Can be overridden via CLI with --gmr.<param>.
    """

    src_human: str = "smplx"
    """Human motion source format. One of: 'smplx', 'bvh'.
    Must match the data_format used in the pipeline."""

    solver: str = "daqp"
    """IK solver backend. 'daqp' (default) or 'quadprog'."""

    damping: float = 5e-1
    """IK solver damping. Higher = smoother but less accurate."""

    actual_human_height: float | None = None
    """Actual subject height in metres. If None, uses GMR config default (~1.7m).
    Used to scale the IK targets to match the robot proportions."""

    max_iter: int = 10
    """Maximum IK refinement iterations per frame."""

    use_velocity_limit: bool = False
    """Whether to enforce motor velocity limits during IK."""
