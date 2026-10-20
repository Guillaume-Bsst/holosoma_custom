"""Whole Body Tracking observation presets for the G1 robot."""

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg

# NOTE: history_length=1 sur l'ACTEUR (et sur le critic). Impose par le WARM START : on repart du
# checkpoint full-loco 3ivghz1e (wandb guibsst-inria/WholeBodyTracking/3ivghz1e), dont l'acteur a
# 154 dims d'entree et le critic 286 -- des tenseurs empiles sur 3 frames ne s'y chargeraient pas
# (PPO.load fait load_state_dict SANS strict=False, cf. agents/ppo/ppo.py:654, donc toute difference
# de forme est fatale). L'empilement 3 frames reste souhaitable en soi (il donne la derive de la
# box, que l'obs instantanee ne porte pas) mais il est incompatible avec ce transfert : il faudra
# repartir de zero pour le retrouver.
actor_obs_shared = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
    terms={
        "motion_command": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_command",
            scale=1.0,
            noise=0.0,
        ),
        "motion_ref_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
            scale=1.0,
            noise=0.05,
        ),
        "base_ang_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:base_ang_vel",
            scale=1.0,
            noise=0.2,
        ),
        "dof_pos": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_pos",
            scale=1.0,
            noise=0.01,
        ),
        "dof_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_vel",
            scale=1.0,
            noise=0.5,
        ),
        "actions": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:actions",
            scale=1.0,
            noise=0.0,
        ),
    },
)

actor_obs_w_object = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,  # voir la note la-haut
    terms={
        **actor_obs_shared.terms,
        # Object pose is available at deployment via mocap/RGB-D perception; add measurement-level
        # noise so the policy is robust to that pipeline (~2 cm position, ~0.05 orientation).
        #
        # The position goes through PerceptionNoisyPosition, which adds the CORRELATED errors a real
        # estimator makes on top of that white noise: a per-episode constant bias (calibration /
        # mesh origin) and a per-episode latency (FoundationPose runs below the 50 Hz control rate).
        # White noise alone is exactly what the policy averages away -- and actor_obs now stacks 3
        # frames, which makes averaging easier -- while a constant bias never averages out.
        # NOTE: the term keeps the name "obj_pos_b" on purpose. Groups are concatenated in
        # ALPHABETICAL term order (ObservationManager.compute_group), so renaming it would reshuffle
        # the observation vector and desync every inference preset. Only `func` changes.
        "obj_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:PerceptionNoisyPosition",
            params={
                "source": "holosoma.managers.observation.terms.wbt:obj_pos_b",
                "bias_range": 0.03,
                "latency_step_range": (0, 3),
                # Occlusion dropout is OFF: freezing the box pose while the hands close on it
                # changes the task, not just the sensor. Worth its own A/B, not this run.
                "dropout_prob": 0.0,
            },
            scale=1.0,
            noise=0.02,
        ),
        "obj_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ori_b",
            scale=1.0,
            noise=0.05,
        ),
        # Table (support statique) : le robot la VOIT pour se placer correctement (dépôt) et ne pas
        # foncer dedans. Pose relative au torse, avec le même bruit de perception que la box.
        "support_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:support_pos_b",
            scale=1.0,
            noise=0.02,
        ),
        "support_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:support_ori_b",
            scale=1.0,
            noise=0.05,
        ),
    },
)

critic_obs_shared_terms = {
    "motion_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_command",
        scale=1.0,
        noise=0.0,
    ),
    "motion_ref_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_pos_b",
        scale=1.0,
        noise=0.25,
    ),
    "motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
        scale=1.0,
        noise=0.05,
    ),
    "robot_body_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_pos_b",
        scale=1.0,
        noise=0.0,
    ),
    "robot_body_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_ori_b",
        scale=1.0,
        noise=0.0,
    ),
    "base_lin_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_lin_vel",
        scale=1.0,
        noise=0.0,
    ),
    "base_ang_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_ang_vel",
        scale=1.0,
        noise=0.2,
    ),
    "dof_pos": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_pos",
        scale=1.0,
        noise=0.01,
    ),
    "dof_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_vel",
        scale=1.0,
        noise=0.5,
    ),
    "actions": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:actions",
        scale=1.0,
        noise=0.0,
    ),
}

critic_obs_w_object_terms = critic_obs_shared_terms.copy()
critic_obs_w_object_terms.update(
    {
        "obj_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ori_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_lin_vel_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_lin_vel_b",
            scale=1.0,
            noise=0.0,
        ),
        "support_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:support_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "support_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:support_ori_b",
            scale=1.0,
            noise=0.0,
        ),
    }
)

g1_29dof_wbt_observation = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
    },
)

g1_29dof_wbt_observation_w_object = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_terms,
        ),
    },
)

g1_29dof_wbt_observation_w_object_actor = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_w_object,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_terms,
        ),
    },
)

_OBS = "holosoma.managers.observation.terms.wbt:"

# Observations PRIVILEGIEES du critique (etage 05). Le critique n'est jamais deploye : il n'existe
# que pour estimer la valeur pendant l'entrainement, donc tout ce qui reduit la variance de cette
# estimation est gratuit vis-a-vis du robot reel. C'est la meme asymetrie qui lui donne deja
# base_lin_vel / obj_lin_vel_b / robot_body_pos_b que l'acteur n'a pas.
#
# AUCUN de ces termes ne doit rejoindre le groupe acteur : soit ils n'existent pas sur le robot
# (forces de contact mesurees -- le G1 n'a pas de capteur d'effort aux poignets), soit ils
# laisseraient la policy se caler sur la position dans le clip plutot que sur son etat (phase).
critic_obs_w_object_dyn_terms = {
    **critic_obs_w_object_terms,
    # Le trou principal : ni l'acteur ni le critique ne savaient ou ils en etaient dans le clip.
    # motion_command ne porte que la pose articulaire de reference. Avec le RSI qui demarre a une
    # phase uniforme et un timeout en fin de clip, le retour atteignable depend directement du temps
    # RESTANT -- que rien dans l'entree ne contenait.
    "motion_phase": ObsTermCfg(func=f"{_OBS}motion_phase", scale=1.0, noise=0.0),
    # Le plafond de reward varie avec la frame (3.0 des 11.0 positifs sont inaccessibles hors
    # contact). Ces flags SONT cette variation -- la meme information que reward/achievable.
    "ref_obj_contact_lr": ObsTermCfg(func=f"{_OBS}ref_obj_contact_lr", scale=1.0, noise=0.0),
    "ref_foot_contact_lr": ObsTermCfg(func=f"{_OBS}ref_foot_contact_lr", scale=1.0, noise=0.0),
    # "De combien il faut serrer" : profil de force mesure par le solve physique. scale=0.01 parce
    # que ces valeurs montent a ~190 N alors que toutes les autres observations sont en O(1).
    "ref_grip_force_lr": ObsTermCfg(func=f"{_OBS}ref_grip_force_lr", scale=0.01, noise=0.0),
    # Ce que le robot touche REELLEMENT, par opposition a ce que la reference prescrit. C'est la
    # variable d'etat a partir de laquelle les rewards de contact sont calculees. scale=0.01 :
    # meme raison, les GRF montent a ~2400 N.
    "measured_contact_forces": ObsTermCfg(func=f"{_OBS}measured_contact_forces", scale=0.01, noise=0.0),
    # Le critique avait obj_lin_vel_b mais pas l'angulaire. Or la mise en rotation de la caisse est
    # exactement ce que object_flat_contact_quality_exp cherche a empecher.
    "obj_ang_vel_b": ObsTermCfg(func=f"{_OBS}obj_ang_vel_b", scale=1.0, noise=0.0),
}

g1_29dof_wbt_observation_w_object_actor_dyn = ObservationManagerCfg(
    groups={
        # Acteur STRICTEMENT inchange -> deployabilite et dimensions preservees.
        "actor_obs": actor_obs_w_object,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_dyn_terms,
        ),
    },
)

# --- Experiment 4 (OTRL docs/agents/architecture.md): make the task observable for the ACTOR -----------------
# w_object_actor's actor sees the reference at the current frame only. This preset adds, with NEW term names
# (groups concatenate in alphabetical term order, so the existing terms keep their place relative to each other):
#   motion_future_joint_pos  ref joints at +0.1/+0.3/+0.6 s                         87
#   motion_future_ref_pos_b  ref anchor (torso) position at +0.1/+0.3/+0.6 s, body    9
#   motion_future_ref_ori_b  ref anchor orientation, same offsets, body frame          18
#   ref_obj_pos_b            REFERENCE box position at 0/+0.1/+0.3/+0.6 s, body       12
#   ref_obj_ori_b            reference box orientation, same offsets, body frame       24
#   motion_ref_pos_b         anchor position error vs the reference (existing term)     3
# = +153 actor dims (172 -> 325). Every future index is clamped to the end of the env's OWN clip
# (terms/wbt.py:future_frame_index). Fresh training only: no warm start from a 172-dim actor.
# Noise (actor only, uniform): the reference itself is exact (0, like motion_command); the body-frame poses
# depend on the robot's estimated pose, so they get the box's perception noise (0.02 m, 0.05).
# The CRITIC already had motion_ref_pos_b (not duplicated); it gets the five new terms noise-free (+150 dims,
# 307 -> 457) so that it sees at least what the actor sees.
FUTURE_OFFSETS_S = (0.1, 0.3, 0.6)
REF_OBJ_OFFSETS_S = (0.0, 0.1, 0.3, 0.6)


def _future_terms(noisy: bool) -> dict[str, ObsTermCfg]:
    pos_n, ori_n = (0.02, 0.05) if noisy else (0.0, 0.0)
    return {
        "motion_future_joint_pos": ObsTermCfg(
            func=f"{_OBS}motion_future_joint_pos", params={"offsets_s": FUTURE_OFFSETS_S}, scale=1.0, noise=0.0
        ),
        "motion_future_ref_pos_b": ObsTermCfg(
            func=f"{_OBS}motion_future_ref_pos_b", params={"offsets_s": FUTURE_OFFSETS_S}, scale=1.0, noise=pos_n
        ),
        "motion_future_ref_ori_b": ObsTermCfg(
            func=f"{_OBS}motion_future_ref_ori_b", params={"offsets_s": FUTURE_OFFSETS_S}, scale=1.0, noise=ori_n
        ),
        "ref_obj_pos_b": ObsTermCfg(
            func=f"{_OBS}ref_obj_pos_b", params={"offsets_s": REF_OBJ_OFFSETS_S}, scale=1.0, noise=pos_n
        ),
        "ref_obj_ori_b": ObsTermCfg(
            func=f"{_OBS}ref_obj_ori_b", params={"offsets_s": REF_OBJ_OFFSETS_S}, scale=1.0, noise=ori_n
        ),
    }


actor_obs_w_object_future = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
    terms={
        **actor_obs_w_object.terms,
        **_future_terms(noisy=True),
        "motion_ref_pos_b": ObsTermCfg(func=f"{_OBS}motion_ref_pos_b", scale=1.0, noise=0.02),
    },
)

critic_obs_w_object_future_terms = {**critic_obs_w_object_terms, **_future_terms(noisy=False)}

g1_29dof_wbt_observation_w_object_actor_future = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_w_object_future,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_future_terms,
        ),
    },
)

__all__ = [
    "g1_29dof_wbt_observation",
    "g1_29dof_wbt_observation_w_object",
    "g1_29dof_wbt_observation_w_object_actor",
    "g1_29dof_wbt_observation_w_object_actor_dyn",
    "g1_29dof_wbt_observation_w_object_actor_future",
]
