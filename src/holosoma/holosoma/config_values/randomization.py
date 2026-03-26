"""Default randomization manager configurations."""

from holosoma.config_values.loco.g1.randomization import g1_29dof_randomization
from holosoma.config_values.loco.t1.randomization import t1_29dof_randomization
from holosoma.config_values.wbt.g1.randomization import g1_29dof_wbt_randomization_isaacsim, g1_29dof_wbt_randomization_isaacsim_w_object, g1_29dof_wbt_randomization_mjwarp, g1_29dof_wbt_randomization_mjwarp_w_object

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof": t1_29dof_randomization,
    "g1_29dof": g1_29dof_randomization,
    "g1_29dof_wbt_isaacsim": g1_29dof_wbt_randomization_isaacsim,
    "g1_29dof_wbt_isaacsim_w_object": g1_29dof_wbt_randomization_isaacsim_w_object,
    "g1_29dof_wbt_mjwarp": g1_29dof_wbt_randomization_mjwarp,
    "g1_29dof_wbt_mjwarp_w_object": g1_29dof_wbt_randomization_mjwarp_w_object,
}
