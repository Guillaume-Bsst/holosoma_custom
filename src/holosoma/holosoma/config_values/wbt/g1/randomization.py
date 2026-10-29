"""Whole Body Tracking randomization presets for the G1 robot."""

from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg

robot_state_dr_at_setup = {
    "randomize_robot_rigid_body_material_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_robot_rigid_body_material_startup",
        params={
            "static_friction_range": [0.3, 1.6],
            "dynamic_friction_range": [0.3, 1.2],
            "restitution_range": [0.0, 0.5],
        },
    ),
    "randomize_base_com_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_base_com_startup",
        params={
            "base_com_range": {"x": [-0.1, 0.1], "y": [-0.1, 0.1], "z": [-0.1, 0.1]},
            "enabled": True,
        },
    ),
    "setup_dof_pos_bias": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:setup_dof_pos_bias",
        params={
            "dof_pos_bias_range": [-0.025, 0.025],
            "enabled": True,
        },
    ),
}

object_state_dr_at_setup = {
    "randomize_object_rigid_body_material_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_object_rigid_body_material_startup",
        params={
            "static_friction_range": [0.1, 0.6],
            "dynamic_friction_range": [0.1, 0.6],
            "restitution_range": [0.0, 1.0],
        },
    ),
    "randomize_object_rigid_body_mass_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_object_rigid_body_mass_startup",
        params={
            # ADD to the 0.811 kg URDF base -> 0.81-1.21 kg (mean ~1.0), la vraie box36 pese ~1 kg.
            # Ancien [1.0, 4.0] => 1.8-4.8 kg : une queue lourde IRREELLE dont le poids residuel
            # (1-alpha)*m*g depassait la capacite de prise -> les boites lourdes derivaient dans la
            # bande morte [0.10, 0.25] m (survie OK, tracking KO), clouant obj_track_success a ~0.83
            # sous le gate 0.85 -> curriculum d'alpha gele. Cf. run vcbkx2mm.
            "mass_distribution_params": [0.0, 0.4],
        },
    ),
    "randomize_object_com_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_object_com_startup",
        params={
            # Une vraie caisse a un contenu : son COM n'est pas au centre geometrique. Le decalage
            # cree un couple permanent autour de la prise que la policy doit compenser -- effet du
            # premier ordre pour du portage. +-4 cm sur un cube de 0.36 m (demi-cote 0.18) = +-22%
            # du demi-cote : sensible mais pas absurde pour une caisse mal remplie. Pendant objet du
            # randomize_base_com_startup du robot.
            "com_range": {"x": [-0.04, 0.04], "y": [-0.04, 0.04], "z": [-0.04, 0.04]},
            "enabled": True,
        },
    ),
    "randomize_object_rigid_body_inertia_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_object_rigid_body_inertia_startup",
        params={
            "inertia_distribution_params_dict": {
                # In beyondmimic, only Ixx is randomized, which is probably a bug instead of a feature.
                # Here, we want to reproduce their work. User should feel free to randomize all terms.
                "Ixx": [0.5, 1.5],
                "Iyy": [1.0, 1.0],
                "Izz": [1.0, 1.0],
                "Ixy": [1.0, 1.0],
                "Iyz": [1.0, 1.0],
                "Ixz": [1.0, 1.0],
            }
        },
    ),
}

base_setup_terms = {
    "push_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState",
        params={
            "push_interval_s": [1.0, 3.0],
            "max_push_vel": [0.75, 0.75, 0.3, 0.52, 0.52, 0.78],
            "enabled": True,
        },
    ),
    "actuator_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState",
        params={
            "kp_range": [0.9, 1.1],
            "kd_range": [0.9, 1.1],
            "rfi_lim_range": [1.0, 1.0],
            "enable_pd_gain": True,
            "enable_rfi_lim": False,
        },
    ),
    "setup_action_delay_buffers": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:setup_action_delay_buffers",
        params={
            "ctrl_delay_step_range": [0, 1],
            "enabled": True,
        },
    ),
    **robot_state_dr_at_setup,
}

base_reset_terms = {
    "push_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"
    ),
    "randomize_push_schedule": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_push_schedule",
    ),
    "randomize_action_delay": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_action_delay",
    ),
    "actuator_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState"
    ),
    "randomize_dof_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_dof_state",
        params={
            "joint_pos_scale_range": [1.0, 1.0],
            "joint_vel_range": [0.0, 0.0],
            "joint_pos_bias_range": [-0.025, 0.025],
            "randomize_dof_pos_bias": True,
        },
    ),
}

base_step_terms = {
    "push_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"
    ),
    "apply_pushes": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:apply_pushes",
    ),
}

g1_29dof_wbt_randomization = RandomizationManagerCfg(
    setup_terms={**base_setup_terms},
    reset_terms={**base_reset_terms},
    step_terms={**base_step_terms},
)

g1_29dof_wbt_randomization_w_object = RandomizationManagerCfg(
    setup_terms={
        **base_setup_terms,
        **object_state_dr_at_setup,
    },
    reset_terms={
        **base_reset_terms,
    },
    step_terms={
        **base_step_terms,
    },
)

# ==================================================================================================
# Variante PUSHES ADOUCIS -- HYPOTHESE NON VALIDEE, aucune experience ne l'utilise par defaut.
#
# Disponible en CLI (--randomization:g1-29dof-wbt-w-object-soft-push) pour un A/B cible, PAS a
# activer par confort : moins de DR = moins de robustesse sim2real, et il faut une raison mesuree
# pour payer ca.
#
# CE QUI L'A MOTIVEE, ET POURQUOI CA N'A PAS SUFFI. Sur le run fjj74m7c, error_body_lin_vel plafonne
# a 0.63 m/s, et une bourrade tiree uniformement dans [-0.75, 0.75] par axe a une norme RMS de
# ... 0.63 m/s. La tentation etait de conclure que la DR sature la metrique. Le calcul dit non :
# _push_robots ajoute une impulsion INSTANTANEE a la vitesse de base (root_states[7:13] += delta),
# pas une perturbation continue. Avec un intervalle de 1 a 3 s et une recuperation de l'ordre de
# 0.3 s, les phases de recuperation occupent ~15 % des pas et ne peuvent contribuer qu'a ~0.05 m/s
# d'une moyenne TEMPORELLE sur l'episode. La coincidence numerique est une coincidence.
#
# QUAND LA RESSORTIR. Une seule situation la rend pertinente : feet_slip_on_ref_stance est passe a
# -2.0 dans le preset _stab, or sous bourrade le robot DOIT deplacer ses pieds pour se rattraper et
# paie une penalite inevitable. Si raw_rew_feet_slip_on_ref_stance refuse de descendre sous ~0.15
# malgre le poids x4, cette interaction devient l'explication candidate n°1 -- et c'est a ce
# moment-la, sur cette question precise, que l'A/B a un sens.
#
# Rien d'autre ne change ici : frottements, masse/COM/inertie de la caisse, gains PD, retard
# d'action, biais d'encodeur sont identiques au preset standard.
soft_push_setup_terms = {
    **base_setup_terms,
    "push_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState",
        params={
            # 2x moins souvent (2 a 5 bourrades par episode de 16 s au lieu de 5 a 16)...
            "push_interval_s": [3.0, 6.0],
            # ... et ~2x plus faible : 0.35 m/s reste une perturbation franche mais passe SOUS
            # l'amplitude propre du mouvement (0.54 m/s median) au lieu de la dominer.
            "max_push_vel": [0.35, 0.35, 0.15, 0.25, 0.25, 0.35],
            "enabled": True,
        },
    ),
}

g1_29dof_wbt_randomization_w_object_soft_push = RandomizationManagerCfg(
    setup_terms={
        **soft_push_setup_terms,
        **object_state_dr_at_setup,
    },
    reset_terms={
        **base_reset_terms,
    },
    step_terms={
        **base_step_terms,
    },
)

__all__ = [
    "g1_29dof_wbt_randomization",
    "g1_29dof_wbt_randomization_w_object",
    "g1_29dof_wbt_randomization_w_object_soft_push",
]


# =====================================================================================
# DynaRetarget (arXiv:2602.06827) section E, object domain randomization -- EXACTLY the
# four object mechanisms the paper names, and nothing else:
#
#   1. object pose randomized on resets in the initial frames
#   2. object velocities randomized on later resets
#   3. random external pushes to the object at random intervals
#   4. object friction randomized
#   5. object mass varied around the nominal value
#
# On restitution: DynaRetarget says only "friction parameters", but it also says its object DR
# is "apart from those used in [18]" (BeyondMimic), whose DR table DOES randomize restitution
# at U(0.0, 0.5) -- the same value this repo already applies to the robot in
# `robot_state_dr_at_setup`. So the object gets BeyondMimic's range rather than the repo's
# default [0.0, 1.0], which is unphysical (1.0 = perfectly elastic).
#
# Deliberately ABSENT: object COM and object inertia. BeyondMimic randomizes the robot's BASE
# inertial properties -- already covered here by `randomize_base_com_startup` in
# base_setup_terms -- not an object's, and DynaRetarget never mentions either. The repo's
# pre-existing `object_state_dr_at_setup` does carry them; they are left out on purpose so
# this preset stays exactly the union of the two papers.
#
# The paper gives NO numeric ranges for any of this. The values below are derived from the
# task itself, and each one is justified rather than typical:
#
#   mass [-0.2, +0.2] kg   additive (operation="add") on the box's MEASURED 0.811 kg
#                          (objects_box36.urdf), i.e. 0.61-1.01 kg, +/-25%. Symmetric
#                          because the paper says "around the nominal value"; note the
#                          repo's pre-existing object preset uses [0.0, 0.4], which can
#                          only make the box heavier.
#   friction [0.4, 1.0]    BRACKETS the mu=0.7 the SBTO retargeting actually ran at
#                          (cfg femto14_box36.yaml). The repo's existing object preset
#                          caps at 0.6, so a policy trained on it would never see the
#                          friction the trajectory was optimised for.
#   pushes                 |dv| <= 0.20 m/s on a 0.811 kg box = a 0.16 N.s impulse, every
#                          2-5 s. The clip's own object speed is 0.034 m/s median /
#                          0.937 m/s p90, so this is a real disturbance mid-carry without
#                          being a rip-out. Angular 0.5 rad/s against a 2.09 rad/s p90.
#   reset pose             +/-0.03 m, +/-0.08 rad. SBTO lands the box 0.017 m from the
#                          reference, and the palms sit 0.087 m inside the 0.18 m
#                          half-edge, so 3 cm perturbs without breaking the grasp.
#   reset velocity         +/-0.15 m/s, +/-0.4 rad/s ~ 15-20% of the clip's own p90.
#   initial_frames = 50    1.0 s at 50 Hz. The box leaves the floor around then, which is
#                          exactly the pose-regime / velocity-regime boundary the paper
#                          describes. The paper itself gives no number.
# =====================================================================================
object_dr_dynaretarget_setup = {
    "randomize_object_rigid_body_material_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_object_rigid_body_material_startup",
        params={
            "static_friction_range": [0.4, 1.0],
            "dynamic_friction_range": [0.4, 1.0],
            "restitution_range": [0.0, 0.5],
        },
    ),
    "randomize_object_rigid_body_mass_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_object_rigid_body_mass_startup",
        params={"mass_distribution_params": [-0.2, 0.2]},
    ),
    "object_push_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:ObjectPushRandomizerState",
        params={
            "push_interval_s": [2.0, 5.0],
            "max_push_lin_vel": 0.20,
            "max_push_ang_vel": 0.5,
            "enabled": True,
        },
    ),
}

g1_29dof_wbt_randomization_dynaretarget = RandomizationManagerCfg(
    setup_terms={**base_setup_terms, **object_dr_dynaretarget_setup},
    reset_terms={
        **base_reset_terms,
        "object_push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:ObjectPushRandomizerState"
        ),
    },
    step_terms={
        **base_step_terms,
        "object_push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:ObjectPushRandomizerState"
        ),
        "apply_object_pushes": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:apply_object_pushes",
        ),
    },
)
