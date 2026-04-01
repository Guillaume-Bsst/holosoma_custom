from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import (
    action,
    algo,
    command,
    curriculum,
    observation,
    randomization,
    reward,
    robot,
    simulator,
    termination,
    terrain,
)

_SHARED = dict(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
)

_algo_ppo = replace(algo.ppo, config=replace(algo.ppo.config, num_learning_iterations=25000, use_symmetry=True))
_algo_fast_sac = replace(algo.fast_sac, config=replace(algo.fast_sac.config, num_learning_iterations=50000, use_symmetry=True))

_nightly_ppo = NightlyConfig(
    iterations=5000,
    metrics={"Episode/rew_tracking_ang_vel": [0.7, "inf"], "Episode/rew_tracking_lin_vel": [0.55, "inf"]},
)
_nightly_fast_sac = NightlyConfig(
    iterations=50000,
    metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.95, "inf"]},
)

# ── IsaacGym ────────────────────────────────────────────────────────────────

g1_29dof_isaacgym = ExperimentConfig(
    **_SHARED,
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_isaacgym_manager"),
    algo=_algo_ppo,
    simulator=simulator.isaacgym,
    reward=reward.g1_29dof_loco,
    curriculum=curriculum.g1_29dof_curriculum,
    nightly=_nightly_ppo,
)

g1_29dof_fast_sac_isaacgym = ExperimentConfig(
    **_SHARED,
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_fast_sac_isaacgym_manager"),
    algo=_algo_fast_sac,
    simulator=simulator.isaacgym,
    curriculum=curriculum.g1_29dof_curriculum_fast_sac,
    reward=reward.g1_29dof_loco_fast_sac,
    nightly=_nightly_fast_sac,
)

# ── MJWarp ──────────────────────────────────────────────────────────────────

g1_29dof_mjwarp = ExperimentConfig(
    **_SHARED,
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_mjwarp_manager"),
    algo=_algo_ppo,
    simulator=simulator.mjwarp,
    reward=reward.g1_29dof_loco,
    curriculum=curriculum.g1_29dof_curriculum,
    nightly=_nightly_ppo,
)

g1_29dof_fast_sac_mjwarp = ExperimentConfig(
    **_SHARED,
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_fast_sac_mjwarp_manager"),
    algo=_algo_fast_sac,
    simulator=simulator.mjwarp,
    curriculum=curriculum.g1_29dof_curriculum_fast_sac,
    reward=reward.g1_29dof_loco_fast_sac,
    nightly=_nightly_fast_sac,
)

# ── Backward-compatible aliases ─────────────────────────────────────────────

g1_29dof = g1_29dof_isaacgym
g1_29dof_fast_sac = g1_29dof_fast_sac_isaacgym

__all__ = [
    "g1_29dof_isaacgym",
    "g1_29dof_fast_sac_isaacgym",
    "g1_29dof_mjwarp",
    "g1_29dof_fast_sac_mjwarp",
    # Aliases
    "g1_29dof",
    "g1_29dof_fast_sac",
]
