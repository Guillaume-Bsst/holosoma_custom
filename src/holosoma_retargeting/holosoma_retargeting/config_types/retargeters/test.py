"""Configuration for the Test retargeter.

Position-based IK using mink/mujoco natively in holosoma_retargeting.
This is the base for experimenting with hybrid GMR+OmniRetarget approaches.

Currently replicates GMR's single-pass IK logic but operates directly on the
(T, J, 3) joint positions produced by the holosoma pipeline, without needing
the external GMR library or the SMPLX body model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TestRetargeterConfig:
    """Configuration for the Test (native mink IK) retargeter.

    Can be overridden via CLI with --test.<param>.
    """

    solver: str = "daqp"
    """IK solver backend passed to mink.solve_ik. 'daqp' or 'quadprog'."""

    damping: float = 5e-1
    """Tikhonov regularisation weight (λ·‖v‖²). Higher = smoother / less accurate."""

    position_cost: float = 100.0
    """Weight on position error for each FrameTask."""

    max_iter: int = 10
    """Maximum IK refinement iterations per frame (stops early on convergence)."""

    convergence_threshold: float = 1e-3
    """Stop refining when error decrease per iteration < this value."""

    use_velocity_limit: bool = False
    """Enforce motor velocity limits (3π rad/s) during IK."""
