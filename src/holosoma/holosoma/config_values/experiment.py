import tyro
from typing_extensions import Annotated

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.loco.g1.experiment import (
    g1_29dof_isaacgym,
    g1_29dof_fast_sac_isaacgym,
    g1_29dof_mjwarp,
    g1_29dof_fast_sac_mjwarp,
    # Aliases
    g1_29dof,
    g1_29dof_fast_sac,
)
from holosoma.config_values.loco.t1.experiment import (
    t1_29dof_isaacgym,
    t1_29dof_fast_sac_isaacgym,
    t1_29dof_mjwarp,
    t1_29dof_fast_sac_mjwarp,
    # Aliases
    t1_29dof,
    t1_29dof_fast_sac,
)
from holosoma.config_values.wbt.g1.experiment import (
    # 29-DOF
    g1_29dof_wbt_isaacsim,
    g1_29dof_wbt_isaacsim_w_object,
    g1_29dof_wbt_fast_sac_isaacsim,
    g1_29dof_wbt_fast_sac_isaacsim_w_object,
    g1_29dof_wbt_mjwarp,
    g1_29dof_wbt_mjwarp_w_object,
    g1_29dof_wbt_fast_sac_mjwarp,
    g1_29dof_wbt_fast_sac_mjwarp_w_object,
    # 27-DOF
    g1_27dof_wbt_isaacsim,
    g1_27dof_wbt_isaacsim_w_object,
    g1_27dof_wbt_fast_sac_isaacsim,
    g1_27dof_wbt_fast_sac_isaacsim_w_object,
    g1_27dof_wbt_mjwarp,
    g1_27dof_wbt_mjwarp_w_object,
    g1_27dof_wbt_fast_sac_mjwarp,
    g1_27dof_wbt_fast_sac_mjwarp_w_object,
)

DEFAULTS = {
    # Locomotion — G1
    "g1_29dof_isaacgym": g1_29dof_isaacgym,
    "g1_29dof_fast_sac_isaacgym": g1_29dof_fast_sac_isaacgym,
    "g1_29dof_mjwarp": g1_29dof_mjwarp,
    "g1_29dof_fast_sac_mjwarp": g1_29dof_fast_sac_mjwarp,
    # Locomotion — T1
    "t1_29dof_isaacgym": t1_29dof_isaacgym,
    "t1_29dof_fast_sac_isaacgym": t1_29dof_fast_sac_isaacgym,
    "t1_29dof_mjwarp": t1_29dof_mjwarp,
    "t1_29dof_fast_sac_mjwarp": t1_29dof_fast_sac_mjwarp,
    # Locomotion — Aliases (backward compat)
    "g1_29dof": g1_29dof,
    "g1_29dof_fast_sac": g1_29dof_fast_sac,
    "t1_29dof": t1_29dof,
    "t1_29dof_fast_sac": t1_29dof_fast_sac,
    # WBT 29-DOF
    "g1_29dof_wbt_isaacsim": g1_29dof_wbt_isaacsim,
    "g1_29dof_wbt_isaacsim_w_object": g1_29dof_wbt_isaacsim_w_object,
    "g1_29dof_wbt_fast_sac_isaacsim": g1_29dof_wbt_fast_sac_isaacsim,
    "g1_29dof_wbt_fast_sac_isaacsim_w_object": g1_29dof_wbt_fast_sac_isaacsim_w_object,
    "g1_29dof_wbt_mjwarp": g1_29dof_wbt_mjwarp,
    "g1_29dof_wbt_mjwarp_w_object": g1_29dof_wbt_mjwarp_w_object,
    "g1_29dof_wbt_fast_sac_mjwarp": g1_29dof_wbt_fast_sac_mjwarp,
    "g1_29dof_wbt_fast_sac_mjwarp_w_object": g1_29dof_wbt_fast_sac_mjwarp_w_object,
    # WBT 27-DOF
    "g1_27dof_wbt_isaacsim": g1_27dof_wbt_isaacsim,
    "g1_27dof_wbt_isaacsim_w_object": g1_27dof_wbt_isaacsim_w_object,
    "g1_27dof_wbt_fast_sac_isaacsim": g1_27dof_wbt_fast_sac_isaacsim,
    "g1_27dof_wbt_fast_sac_isaacsim_w_object": g1_27dof_wbt_fast_sac_isaacsim_w_object,
    "g1_27dof_wbt_mjwarp": g1_27dof_wbt_mjwarp,
    "g1_27dof_wbt_mjwarp_w_object": g1_27dof_wbt_mjwarp_w_object,
    "g1_27dof_wbt_fast_sac_mjwarp": g1_27dof_wbt_fast_sac_mjwarp,
    "g1_27dof_wbt_fast_sac_mjwarp_w_object": g1_27dof_wbt_fast_sac_mjwarp_w_object,
}

AnnotatedExperimentConfig = Annotated[
    ExperimentConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults(
            {f"exp:{k.replace('_', '-')}": v for k, v in DEFAULTS.items()}
        )
    ),
]
