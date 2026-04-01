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
    robot=robot.t1_29dof_waist_wrist,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.t1_29dof_loco_single_wolinvel,
    action=action.t1_29dof_joint_pos,
    termination=termination.t1_29dof_termination,
    randomization=randomization.t1_29dof_randomization,
    command=command.t1_29dof_command,
)

_algo_ppo = replace(algo.ppo, config=replace(algo.ppo.config, num_learning_iterations=25000, use_symmetry=True))
_algo_fast_sac = replace(
    algo.fast_sac, config=replace(algo.fast_sac.config, num_learning_iterations=100000, use_symmetry=True)
)

_nightly_ppo = NightlyConfig(
    iterations=10000,
    metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.75, "inf"]},
)
_nightly_fast_sac = NightlyConfig(
    iterations=50000,
    metrics={"Episode/rew_tracking_ang_vel": [0.65, "inf"], "Episode/rew_tracking_lin_vel": [0.9, "inf"]},
)

# ── IsaacGym ────────────────────────────────────────────────────────────────

t1_29dof_isaacgym = ExperimentConfig(
    **_SHARED,
    training=TrainingConfig(project="hv-t1-manager", name="t1_29dof_isaacgym_manager"),
    algo=_algo_ppo,
    simulator=simulator.isaacgym,
    reward=reward.t1_29dof_loco,
    curriculum=curriculum.t1_29dof_curriculum,
    nightly=_nightly_ppo,
)

t1_29dof_fast_sac_isaacgym = ExperimentConfig(
    **_SHARED,
    training=TrainingConfig(project="hv-t1-manager", name="t1_29dof_fast_sac_isaacgym_manager"),
    algo=_algo_fast_sac,
    simulator=simulator.isaacgym,
    curriculum=curriculum.t1_29dof_curriculum_fast_sac,
    reward=reward.t1_29dof_loco_fast_sac,
    nightly=_nightly_fast_sac,
)

# ── MJWarp ──────────────────────────────────────────────────────────────────

t1_29dof_mjwarp = ExperimentConfig(
    **_SHARED,
    training=TrainingConfig(project="hv-t1-manager", name="t1_29dof_mjwarp_manager"),
    algo=_algo_ppo,
    simulator=simulator.mjwarp,
    reward=reward.t1_29dof_loco,
    curriculum=curriculum.t1_29dof_curriculum,
    nightly=_nightly_ppo,
)

t1_29dof_fast_sac_mjwarp = ExperimentConfig(
    **_SHARED,
    training=TrainingConfig(project="hv-t1-manager", name="t1_29dof_fast_sac_mjwarp_manager"),
    algo=_algo_fast_sac,
    simulator=simulator.mjwarp,
    curriculum=curriculum.t1_29dof_curriculum_fast_sac,
    reward=reward.t1_29dof_loco_fast_sac,
    nightly=_nightly_fast_sac,
)

# ── Backward-compatible aliases ─────────────────────────────────────────────

t1_29dof = t1_29dof_isaacgym
t1_29dof_fast_sac = t1_29dof_fast_sac_isaacgym

__all__ = [
    "t1_29dof_isaacgym",
    "t1_29dof_fast_sac_isaacgym",
    "t1_29dof_mjwarp",
    "t1_29dof_fast_sac_mjwarp",
    # Aliases
    "t1_29dof",
    "t1_29dof_fast_sac",
]
