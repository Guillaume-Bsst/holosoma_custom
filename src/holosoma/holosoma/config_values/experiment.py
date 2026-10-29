import tyro
from typing_extensions import Annotated

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.loco.g1.experiment import g1_27dof, g1_27dof_fast_sac, g1_29dof, g1_29dof_fast_sac
from holosoma.config_values.loco.t1.experiment import t1_29dof, t1_29dof_fast_sac
from holosoma.config_values.wbt.g1.experiment import (
    g1_27dof_wbt,
    g1_27dof_wbt_fast_sac,
    g1_27dof_wbt_fast_sac_w_object,
    g1_27dof_wbt_w_object,
    g1_27dof_wbt_w_object_actor,
    g1_29dof_wbt,
    g1_29dof_wbt_fast_sac,
    g1_29dof_wbt_fast_sac_w_object,
    g1_29dof_wbt_w_object,
    g1_29dof_wbt_w_object_actor,
    g1_29dof_wbt_w_object_actor_dynaretarget,
    g1_29dof_wbt_w_object_actor_dynaretarget_dr,
    g1_29dof_wbt_w_object_actor_dynaretarget_dr_sbto,
    g1_29dof_wbt_w_object_actor_dynaretarget_dr_sbto_hdmi,
    g1_29dof_wbt_w_object_actor_grip_force,
    g1_29dof_wbt_w_object_actor_grip_force_dyn,
    g1_29dof_wbt_w_object_actor_grip_force_dyn_approach,
    g1_29dof_wbt_w_object_actor_grip_force_dyn_both,
    g1_29dof_wbt_w_object_actor_grip_force_dyn_qref,
    g1_29dof_wbt_w_object_actor_grip_force_dyn_refobj,
    g1_29dof_wbt_w_object_actor_grip_force_dyn_stab,
    g1_29dof_wbt_w_object_actor_grip_force_dyn_stab2,
    g1_29dof_wbt_w_object_actor_dyn_nogrip,
    g1_29dof_wbt_w_object_actor_dyn_nogrip_ff08,
)

DEFAULTS = {
    "g1_29dof": g1_29dof,
    "g1_29dof_fast_sac": g1_29dof_fast_sac,
    "g1_27dof": g1_27dof,
    "g1_27dof_fast_sac": g1_27dof_fast_sac,
    "t1_29dof": t1_29dof,
    "t1_29dof_fast_sac": t1_29dof_fast_sac,
    "g1_29dof_wbt": g1_29dof_wbt,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_w_object,
    "g1_29dof_wbt_w_object_actor": g1_29dof_wbt_w_object_actor,
    "g1_29dof_wbt_w_object_actor_dynaretarget": g1_29dof_wbt_w_object_actor_dynaretarget,
    "g1_29dof_wbt_w_object_actor_dynaretarget_dr": g1_29dof_wbt_w_object_actor_dynaretarget_dr,
    "g1_29dof_wbt_w_object_actor_dynaretarget_dr_sbto": g1_29dof_wbt_w_object_actor_dynaretarget_dr_sbto,
    "g1_29dof_wbt_w_object_actor_dynaretarget_dr_sbto_hdmi": g1_29dof_wbt_w_object_actor_dynaretarget_dr_sbto_hdmi,
    "g1_29dof_wbt_w_object_actor_grip_force": g1_29dof_wbt_w_object_actor_grip_force,
    "g1_29dof_wbt_w_object_actor_grip_force_dyn": g1_29dof_wbt_w_object_actor_grip_force_dyn,
    "g1_29dof_wbt_w_object_actor_grip_force_dyn_approach": g1_29dof_wbt_w_object_actor_grip_force_dyn_approach,
    "g1_29dof_wbt_w_object_actor_grip_force_dyn_stab": g1_29dof_wbt_w_object_actor_grip_force_dyn_stab,
    "g1_29dof_wbt_w_object_actor_grip_force_dyn_stab2": g1_29dof_wbt_w_object_actor_grip_force_dyn_stab2,
    "g1_29dof_wbt_w_object_actor_dyn_nogrip": g1_29dof_wbt_w_object_actor_dyn_nogrip,
    "g1_29dof_wbt_w_object_actor_dyn_nogrip_ff08": g1_29dof_wbt_w_object_actor_dyn_nogrip_ff08,
    "g1_29dof_wbt_w_object_actor_grip_force_dyn_qref": g1_29dof_wbt_w_object_actor_grip_force_dyn_qref,
    "g1_29dof_wbt_w_object_actor_grip_force_dyn_refobj": g1_29dof_wbt_w_object_actor_grip_force_dyn_refobj,
    "g1_29dof_wbt_w_object_actor_grip_force_dyn_both": g1_29dof_wbt_w_object_actor_grip_force_dyn_both,
    "g1_29dof_wbt_fast_sac": g1_29dof_wbt_fast_sac,
    "g1_29dof_wbt_fast_sac_w_object": g1_29dof_wbt_fast_sac_w_object,
    "g1_27dof_wbt": g1_27dof_wbt,
    "g1_27dof_wbt_w_object": g1_27dof_wbt_w_object,
    "g1_27dof_wbt_w_object_actor": g1_27dof_wbt_w_object_actor,
    "g1_27dof_wbt_fast_sac": g1_27dof_wbt_fast_sac,
    "g1_27dof_wbt_fast_sac_w_object": g1_27dof_wbt_fast_sac_w_object,
}

AnnotatedExperimentConfig = Annotated[
    ExperimentConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults(
            {f"exp:{k.replace('_', '-')}": v for k, v in DEFAULTS.items()}
        )
    ),
]
