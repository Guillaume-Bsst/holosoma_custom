"""Whole Body Tracking experiment presets for the G1 robot.

All 16 combinations of:
  DOF:       27 (G1 base) | 29 (G1 pro)
  Simulator: IsaacSim     | MJWarp
  Algorithm: PPO          | Fast-SAC
  Object:    without      | with

Naming convention:
  g1_{dof}dof_wbt_{algo}_{sim}[_w_object]

Where {algo} is omitted for PPO (default) and {sim} is always explicit.
"""

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


# ═════════════════════════════════════════════════════════════════════════════
# Building blocks — simulator configs
# ═════════════════════════════════════════════════════════════════════════════

_sim_isaacsim = replace(
    simulator.isaacsim,
    config=replace(
        simulator.isaacsim.config,
        sim=replace(simulator.isaacsim.config.sim, max_episode_length_s=10.0),
    ),
)

_sim_isaacsim_w_object = replace(
    simulator.isaacsim,
    config=replace(
        simulator.isaacsim.config,
        sim=replace(simulator.isaacsim.config.sim, max_episode_length_s=10.0),
        scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0),
    ),
)

_sim_mjwarp = replace(
    simulator.mjwarp,
    config=replace(
        simulator.mjwarp.config,
        sim=replace(simulator.mjwarp.config.sim, max_episode_length_s=10.0),
    ),
)

_sim_mjwarp_w_object = replace(
    simulator.mjwarp_w_object,
    config=replace(
        simulator.mjwarp_w_object.config,
        sim=replace(simulator.mjwarp_w_object.config.sim, max_episode_length_s=10.0),
        scene=replace(simulator.mjwarp_w_object.config.scene, env_spacing=0.0),
    ),
)


# ═════════════════════════════════════════════════════════════════════════════
# Building blocks — algorithm configs
# ═════════════════════════════════════════════════════════════════════════════

_algo_ppo = replace(
    algo.ppo,
    config=replace(
        algo.ppo.config,
        num_learning_iterations=40000,
        save_interval=4000,
        entropy_coef=0.005,
        init_noise_std=1.0,
        init_at_random_ep_len=False,
        use_symmetry=False,
        actor_optimizer=replace(algo.ppo.config.actor_optimizer, weight_decay=0.000),
        critic_optimizer=replace(algo.ppo.config.critic_optimizer, weight_decay=0.000),
    ),
)

_algo_fast_sac = replace(
    algo.fast_sac,
    config=replace(
        algo.fast_sac.config,
        num_learning_iterations=400000,
        v_max=20.0,
        v_min=-20.0,
        gamma=0.99,
        num_steps=1,
        num_updates=4,
        num_atoms=501,
        policy_frequency=2,
        target_entropy_ratio=0.5,
        tau=0.05,
        use_symmetry=False,
    ),
)


# ═════════════════════════════════════════════════════════════════════════════
# Building blocks — per-DOF robot/command/termination
# ═════════════════════════════════════════════════════════════════════════════

_DOF_CONFIGS = {
    29: dict(
        robot_cfg=replace(
            robot.g1_29dof,
            control=replace(robot.g1_29dof.control, action_scale=1.0),
            asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
            init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
        ),
        robot_w_object_cfg=replace(
            robot.g1_29dof_w_object,
            asset=replace(robot.g1_29dof_w_object.asset, enable_self_collisions=True),
            object=replace(
                robot.g1_29dof_w_object.object,
                object_urdf_path="holosoma_data/pipeline/converted/g1_29dof/whole_body_tracking/objects_largebox.urdf",
            ),
            init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
        ),
        command_cfg=command.g1_29dof_wbt_command,
        command_w_object_cfg=command.g1_29dof_wbt_command_w_object,
        termination_cfg=termination.g1_29dof_wbt_termination,
    ),
    27: dict(
        robot_cfg=replace(
            robot.g1_27dof,
            control=replace(robot.g1_27dof.control, action_scale=1.0),
            asset=replace(robot.g1_27dof.asset, enable_self_collisions=True),
            init_state=replace(robot.g1_27dof.init_state, pos=[0.0, 0.0, 0.76]),
        ),
        robot_w_object_cfg=replace(
            robot.g1_27dof_w_object,
            asset=replace(robot.g1_27dof_w_object.asset, enable_self_collisions=True),
            object=replace(
                robot.g1_27dof_w_object.object,
                object_urdf_path="holosoma_data/pipeline/converted/g1_29dof/whole_body_tracking/objects_largebox.urdf",
            ),
            init_state=replace(robot.g1_27dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
        ),
        command_cfg=command.g1_27dof_wbt_command,
        command_w_object_cfg=command.g1_27dof_wbt_command_w_object,
        termination_cfg=termination.g1_27dof_wbt_termination,
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
# Shared configs (DOF-agnostic)
# ═════════════════════════════════════════════════════════════════════════════

_SHARED = dict(
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    terrain=terrain.terrain_locomotion_plane,
    action=action.g1_29dof_joint_pos,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
)


# ═════════════════════════════════════════════════════════════════════════════
# Generator
# ═════════════════════════════════════════════════════════════════════════════

def _make_experiment(
    dof: int,
    sim_name: str,
    algo_name: str,
    with_object: bool,
) -> ExperimentConfig:
    """Build a single WBT experiment from the 4 axes."""
    d = _DOF_CONFIGS[dof]
    is_sac = algo_name == "fast_sac"

    # Name
    parts = [f"g1_{dof}dof_wbt"]
    if is_sac:
        parts.append("fast_sac")
    parts.append(sim_name)
    if with_object:
        parts.append("w_object")
    name = "_".join(parts)

    # Algo
    algo_cfg = _algo_fast_sac if is_sac else _algo_ppo

    # Simulator + randomization
    if sim_name == "isaacsim":
        if with_object:
            sim_cfg = _sim_isaacsim_w_object
            rand_cfg = randomization.g1_29dof_wbt_randomization_isaacsim_w_object
        else:
            sim_cfg = _sim_isaacsim
            rand_cfg = randomization.g1_29dof_wbt_randomization_isaacsim
    else:  # mjwarp
        if with_object:
            sim_cfg = _sim_mjwarp_w_object
            rand_cfg = randomization.g1_29dof_wbt_randomization_mjwarp_w_object
        else:
            sim_cfg = _sim_mjwarp
            rand_cfg = randomization.g1_29dof_wbt_randomization_mjwarp

    # Robot / command / reward / observation (object vs not)
    if with_object:
        robot_cfg = d["robot_w_object_cfg"]
        command_cfg = d["command_w_object_cfg"]
        reward_cfg = reward.g1_29dof_wbt_reward_w_object
        obs_cfg = observation.g1_29dof_wbt_observation_w_object
    else:
        robot_cfg = d["robot_cfg"]
        command_cfg = d["command_cfg"]
        reward_cfg = reward.g1_29dof_wbt_fast_sac_reward if is_sac else reward.g1_29dof_wbt_reward
        obs_cfg = observation.g1_29dof_wbt_observation

    return ExperimentConfig(
        **_SHARED,
        training=TrainingConfig(
            project="WholeBodyTracking",
            name=f"{name}_manager",
            num_envs=8192,
        ),
        algo=algo_cfg,
        simulator=sim_cfg,
        robot=robot_cfg,
        command=command_cfg,
        termination=d["termination_cfg"],
        randomization=rand_cfg,
        reward=reward_cfg,
        observation=obs_cfg,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Generate all 16 combinations
# ═════════════════════════════════════════════════════════════════════════════

# fmt: off
# ── 29-DOF ───────────────────────────────────────────────────────────────────
g1_29dof_wbt_isaacsim              = _make_experiment(29, "isaacsim", "ppo",      False)
g1_29dof_wbt_isaacsim_w_object     = _make_experiment(29, "isaacsim", "ppo",      True)
g1_29dof_wbt_fast_sac_isaacsim     = _make_experiment(29, "isaacsim", "fast_sac", False)
g1_29dof_wbt_fast_sac_isaacsim_w_object = _make_experiment(29, "isaacsim", "fast_sac", True)
g1_29dof_wbt_mjwarp                = _make_experiment(29, "mjwarp",   "ppo",      False)
g1_29dof_wbt_mjwarp_w_object       = _make_experiment(29, "mjwarp",   "ppo",      True)
g1_29dof_wbt_fast_sac_mjwarp       = _make_experiment(29, "mjwarp",   "fast_sac", False)
g1_29dof_wbt_fast_sac_mjwarp_w_object = _make_experiment(29, "mjwarp", "fast_sac", True)

# ── 27-DOF ───────────────────────────────────────────────────────────────────
g1_27dof_wbt_isaacsim              = _make_experiment(27, "isaacsim", "ppo",      False)
g1_27dof_wbt_isaacsim_w_object     = _make_experiment(27, "isaacsim", "ppo",      True)
g1_27dof_wbt_fast_sac_isaacsim     = _make_experiment(27, "isaacsim", "fast_sac", False)
g1_27dof_wbt_fast_sac_isaacsim_w_object = _make_experiment(27, "isaacsim", "fast_sac", True)
g1_27dof_wbt_mjwarp                = _make_experiment(27, "mjwarp",   "ppo",      False)
g1_27dof_wbt_mjwarp_w_object       = _make_experiment(27, "mjwarp",   "ppo",      True)
g1_27dof_wbt_fast_sac_mjwarp       = _make_experiment(27, "mjwarp",   "fast_sac", False)
g1_27dof_wbt_fast_sac_mjwarp_w_object = _make_experiment(27, "mjwarp", "fast_sac", True)
# fmt: on


# ═════════════════════════════════════════════════════════════════════════════
# Backward-compatible aliases (old names without explicit simulator for SAC)
# ═════════════════════════════════════════════════════════════════════════════

g1_29dof_wbt_fast_sac = g1_29dof_wbt_fast_sac_isaacsim
g1_29dof_wbt_fast_sac_w_object = g1_29dof_wbt_fast_sac_isaacsim_w_object
g1_27dof_wbt_fast_sac = g1_27dof_wbt_fast_sac_isaacsim


__all__ = [
    # 29-DOF
    "g1_29dof_wbt_isaacsim",
    "g1_29dof_wbt_isaacsim_w_object",
    "g1_29dof_wbt_fast_sac_isaacsim",
    "g1_29dof_wbt_fast_sac_isaacsim_w_object",
    "g1_29dof_wbt_mjwarp",
    "g1_29dof_wbt_mjwarp_w_object",
    "g1_29dof_wbt_fast_sac_mjwarp",
    "g1_29dof_wbt_fast_sac_mjwarp_w_object",
    # 27-DOF
    "g1_27dof_wbt_isaacsim",
    "g1_27dof_wbt_isaacsim_w_object",
    "g1_27dof_wbt_fast_sac_isaacsim",
    "g1_27dof_wbt_fast_sac_isaacsim_w_object",
    "g1_27dof_wbt_mjwarp",
    "g1_27dof_wbt_mjwarp_w_object",
    "g1_27dof_wbt_fast_sac_mjwarp",
    "g1_27dof_wbt_fast_sac_mjwarp_w_object",
    # Aliases
    "g1_29dof_wbt_fast_sac",
    "g1_29dof_wbt_fast_sac_w_object",
    "g1_27dof_wbt_fast_sac",
]


"""
Naming convention:
  exp:g1-{dof}dof-wbt-[fast-sac-]{sim}[-w-object]

Full matrix (16 experiments):

  exp:g1-29dof-wbt-isaacsim
  exp:g1-29dof-wbt-isaacsim-w-object
  exp:g1-29dof-wbt-fast-sac-isaacsim
  exp:g1-29dof-wbt-fast-sac-isaacsim-w-object
  exp:g1-29dof-wbt-mjwarp
  exp:g1-29dof-wbt-mjwarp-w-object
  exp:g1-29dof-wbt-fast-sac-mjwarp
  exp:g1-29dof-wbt-fast-sac-mjwarp-w-object

  exp:g1-27dof-wbt-isaacsim
  exp:g1-27dof-wbt-isaacsim-w-object
  exp:g1-27dof-wbt-fast-sac-isaacsim
  exp:g1-27dof-wbt-fast-sac-isaacsim-w-object
  exp:g1-27dof-wbt-mjwarp
  exp:g1-27dof-wbt-mjwarp-w-object
  exp:g1-27dof-wbt-fast-sac-mjwarp
  exp:g1-27dof-wbt-fast-sac-mjwarp-w-object
"""
