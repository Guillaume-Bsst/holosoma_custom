"""Whole Body Tracking action presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.action import (
    GripForceCfg,
    PositionReferenceCfg,
    TorqueFeedforwardCfg,
    TorqueReferenceNoiseCfg,
)
from holosoma.config_values.loco.g1.action import g1_29dof_joint_pos

# Object-carry training with real physical grip force instead of the box-physicality curriculum
# (see config_values/wbt/g1/command.py): the box is fully physical from step 0, and each hand
# presses it at grip_force.target_force_n whenever the command term's GT contact flag is active.
#
# 60 N (not 30): with object mass randomized abs [0.8, 1.2] kg and independently-randomized
# box/hand friction (box static/dynamic in [0.1, 0.6], hand in [0.3, 1.6]/[0.3, 1.2]), the worst-case
# combine="average" effective mu is ~0.20 -- 2 * 0.20 * 30N = 12.0N vs a max box weight of
# 1.2*9.81 = 11.77N is only a ~2% margin. 60N/hand gives 2 * 0.20 * 60N = 24N, a ~2x margin even in
# that worst-case draw (still within the wrist torque limits: ~3 N.m at 60N given the ~0.05m lever
# arm, under the 5 N.m pitch/yaw and 25 N.m roll effort limits).
grip_force_cfg = GripForceCfg(
    enable=True,
    target_force_n=60.0,
    force_command_max_n=90.0,
    # Recale sur la main HALF-SPHERE : origine de son hand_palm_joint dans le repere poignet.
    # Le defaut (0.0415, 0.003, 0) est celui de la main RUBBER -- c'est la valeur qu'utilisait la
    # run de reference c4k7xrin, qui tournait pourtant en half-sphere : elle appliquait donc sa
    # force 1.25 cm devant la paume reelle, soit ~25% d'erreur sur le bras de levier du couple
    # poignet. Corrige ici.
    hand_offset_local=(0.029, -0.003, 0.0),
)

g1_29dof_joint_pos_grip_force = replace(
    g1_29dof_joint_pos,
    terms={
        **g1_29dof_joint_pos.terms,
        "joint_control": replace(
            g1_29dof_joint_pos.terms["joint_control"],
            params={"grip_force": grip_force_cfg},
        ),
    },
)

# Variante etage-05 : la prise suit le PROFIL de force mesure par le solve physique (par main, par
# frame) au lieu du 60 N constant, et le couple de reference est injecte en feed-forward dans la loi
# PD (cf. TorqueFeedforwardCfg).
#
# Les deux retombent silencieusement sur le comportement ci-dessus si le clip charge ne porte pas
# les champs dyn_* (has_dyn_grip / has_dyn_tau) -- le feed-forward log un warning en se desactivant.
grip_force_cfg_profile = replace(grip_force_cfg, use_reference_profile=True)

# scale=0.5 : la moitie du couple de reference suffit a retirer l'essentiel de la charge de
# compensation de gravite tout en laissant a la boucle PD l'autorite de contredire le
# feed-forward -- le couple de reference est exact pour l'ETAT de reference, et la policy n'y est
# jamais exactement.
#
# Poignets EXCLUS (cf. TorqueFeedforwardCfg.exclude_joint_names) : sur ce clip, wrist_pitch et
# wrist_yaw sont colles a leur limite de 5 N.m sur 41-50 % des frames. La valeur enregistree y est
# donc la butee du cap de couple, pas la demande du mouvement -- la feed-forwarder reviendrait a
# commander un biais permanent a mi-butee que la policy devrait combattre. Les autres DOF saturent
# 0.9-16 % du temps, ce qui reste un signal exploitable.
torque_ff_cfg = TorqueFeedforwardCfg(
    enable=True,
    scale=0.5,
    exclude_joint_names=("wrist_pitch", "wrist_yaw"),
)

# Bruit de couple proportionnel au couple DEMANDE plutot qu'a la butee de l'actionneur (cf.
# TorqueReferenceNoiseCfg). Le RFI existant (actuator_randomizer_state) est desactive
# (enable_rfi_lim: false) et perturbe de toute facon une fraction fixe de la LIMITE de chaque joint,
# identique a chaque frame -- ce qui donne un bruit negligeable sous charge et dominant a vide.
#
# ref_scale=0.15 : +/-15 % du couple demande. floor_scale=0.01 : plancher a 1 % de la butee, pour
# que l'exploration ne disparaisse pas la ou tau_ref ~ 0 (jambe en vol), ou un bruit purement
# proportionnel s'annulerait exactement la ou la policy a le plus de latitude.
#
# Poignets exclus pour la meme raison que le feed-forward, PLUS une propre au bruit : colles a leur
# butee ~50 % du clip, le clip_torques final rogne toute excursion positive, donc le "bruit" y
# devient un biais unilateral vers le bas -- pire que pas de randomisation du tout.
torque_noise_cfg = TorqueReferenceNoiseCfg(
    enable=True,
    ref_scale=0.15,
    floor_scale=0.01,
    exclude_joint_names=("wrist_pitch", "wrist_yaw"),
)

g1_29dof_joint_pos_grip_force_dyn = replace(
    g1_29dof_joint_pos,
    terms={
        **g1_29dof_joint_pos.terms,
        "joint_control": replace(
            g1_29dof_joint_pos.terms["joint_control"],
            params={
                "grip_force": grip_force_cfg_profile,
                "torque_feedforward": torque_ff_cfg,
                "torque_reference_noise": torque_noise_cfg,
            },
        ),
    },
)

# Ancre PD rebasee sur la pose de reference (cf. PositionReferenceCfg). scale=1.0 : action nulle ==
# "suis exactement la reference", la policy n'apprend plus que la CORRECTION.
#
# Mesure qui le motive, sur ce clip : l'action necessaire pour atteindre q_ref depuis la pose par
# defaut vaut (q_ref - q_default) / action_scale, soit 1.87 en mediane, 8.78 au maximum, et 8 DOF
# sur 29 au-dessus de 3.0 -- shoulder_pitch a 5.68, les poignets entre 6.8 et 8.8. Avec
# init_noise_std=1.0 ces cibles sont a 5-9 sigma de l'initialisation. Le run tn5t8rjj a atteint
# action_mean_abs=0.89 en 950 iterations, mpjpe fige a ~0.21 rad, et les mains ne sont jamais
# descendues sous 0.55 m du centre de la caisse : la policy n'avait pas fini de traverser cet
# ecart. Rebaser l'ancre reduit ce trajet a une correction de ~0.2-0.5, dans la gaussienne initiale.
#
# Ce n'est pas de la triche : la trajectoire de reference a ete resolue dans MuJoCo AVEC contacts
# par l'etage 05, donc action=0 est une trajectoire physiquement admissible. C'est le pendant
# cinematique du feed-forward de couple, qui donne deja le couple de reference -- le preset offrait
# le prior dynamique en retenant le prior cinematique, bien moins cher.
#
# Poignets NON exclus, contrairement au couple : q_ref est une grandeur cinematique, jamais rognee
# par une butee d'actionneur, donc la reference y est exploitable meme la ou dyn_tau ne l'est pas.
position_ref_cfg = PositionReferenceCfg(enable=True, scale=1.0)

g1_29dof_joint_pos_grip_force_dyn_qref = replace(
    g1_29dof_joint_pos,
    terms={
        **g1_29dof_joint_pos.terms,
        "joint_control": replace(
            g1_29dof_joint_pos.terms["joint_control"],
            params={
                "grip_force": grip_force_cfg_profile,
                "torque_feedforward": torque_ff_cfg,
                "torque_reference_noise": torque_noise_cfg,
                "position_reference": position_ref_cfg,
            },
        ),
    },
)

# SANS PRISE SCRIPTEE. Identique a g1_29dof_joint_pos_grip_force_dyn, grip_force desactive.
#
# POURQUOI. grip_force commande un serrage de target_force_n (60 N) sur les 3 DDL de poignet des que
# le flag de contact GT du clip s'active, via tau = J^T F. Ce flag est SCRIPTE depuis la reference :
# il ne vient d'aucun capteur, il ne depend pas de ce que la policy fait. C'est donc une prothese qui
# impose la prise au lieu de la laisser emerger de la dynamique resolue par l'etage 05 -- exactement
# le contraire de ce que ce clip enrichi est cense enseigner. Meme famille que torque_feedforward, et
# meme dette au deploiement (il faut rejouer le profil indexe sur la phase, sinon la policy serre a
# contretemps).
#
# CE QUE LA MESURE DIT DU RISQUE (checkpoint 32000, run 0ftoz226). La caisse est portee entre les
# AVANT-BRAS -- wrist_pitch encaisse jusqu'a 292 N, wrist_roll 194 N, elbow 163 N, tandis que
# wrist_yaw (le corps d'ancrage, ou vivent les demi-spheres de prise) lit exactement 0.000 N. La
# force normale reelle vaut ~16 N par avant-bras pour une caisse de 8 a 12 N :
#
#   mu effectif    capacite de frottement    marge / poids
#   0.20                    6.4 N                0.54 x     -> tombe
#   0.35                   11.2 N                0.94 x     -> tombe
#   0.50                   16.0 N                1.35 x
#   0.70                   22.4 N                1.89 x
#
# avec un frottement caisse randomise sur [0.1, 0.6]. La prise est donc marginale AVEC les 60 N
# commandes ; sans eux, la source de cette force normale disparait et la caisse tombera sur une
# bonne partie de la plage. C'est un risque assume et documente, pas un oubli : l'alternative
# (annealing progressif de target_force_n) garde la prothese dans la boucle plus longtemps.
#
# CE QUI NE CHANGE PAS, et pourquoi la reprise a chaud reste valide : la dimension d'action reste
# 29, l'observation ne bouge pas (ref_grip_force_lr reste dans le critique -- il lit le clip, pas la
# force appliquee), et grip_active / grip_active_lr continuent d'etre calcules par le command term
# (wbt.py:1397), donc les metriques motion/grip_active survivent. Seul le biais de couple disparait.
g1_29dof_joint_pos_dyn_nogrip = replace(
    g1_29dof_joint_pos_grip_force_dyn,
    terms={
        **g1_29dof_joint_pos_grip_force_dyn.terms,
        "joint_control": replace(
            g1_29dof_joint_pos_grip_force_dyn.terms["joint_control"],
            params={
                "grip_force": replace(grip_force_cfg_profile, enable=False),
                "torque_feedforward": torque_ff_cfg,
                "torque_reference_noise": torque_noise_cfg,
            },
        ),
    },
)

# SONDE FEEDFORWARD 0.8 -- identique a g1_29dof_joint_pos_dyn_nogrip, scale 0.5 -> 0.8.
#
# CE QUI LA MOTIVE. Un actionneur en position produit tau = kp x erreur. SPIDER resout a kp=500 :
# 4 deg d'erreur suffisent a sortir les 35 N.m de la cheville. L'entrainement tourne a kp=28.5 : il
# en faudrait 70 deg, sur un joint dont la course totale fait 80 deg. La conversion exacte d'un
# solve kp=500 vers ces gains, q_target = q_ref + dyn_tau/kp, sort de la course articulaire sur
# waist_pitch (70 deg demandes pour 60 de course), waist_roll et ankle_roll. Elle est donc
# mathematiquement bien definie et physiquement irrealisable.
#
# Le feed-forward EST cette conversion, appliquee partiellement : le PD n'a plus a fournir que
# (1 - scale) x dyn_tau, donc le decalage de cible necessaire est divise d'autant.
#
#   scale     decalage cheville     decalage waist_pitch
#   0.5           35 deg / 80            35 deg / 60      <- actuel, tout juste
#   0.8           14 deg / 80            14 deg / 60      <- confortable partout
#
# CE QUE LA MESURE DIT DU BESOIN. Sur 15 000 iterations et trois presets de reward, error_joint_vel
# est passee de 12.58 a 11.74 (-6.7 %) et asymptote. Le dernier test etait explicitement falsifiable
# (cf. entete de _dyn_stab2) : porter torque_envelope a 5.4 % du budget positif. Le terme a repondu
# -- sa valeur a chute de 20 % en 1000 iterations -- et error_joint_vel n'a pas suivi (-3 % puis
# plat). Le jitter n'est donc PAS reductible par le reward : ce n'est pas un exces d'effort que la
# policy pourrait economiser, c'est une autorite qui manque.
#
# L'AVERTISSEMENT DU CODE NE MORD PAS ICI. _compute_torque_feedforward loggue torque_ff/limit_frac
# avec le commentaire "si ca approche 1, la policy n'a plus d'autorite pour corriger". A scale=0.8 ce
# diagnostic vaudra 0.8 par construction. Mais la marge de correction reelle vaut
# limite - |dyn_tau|, INDEPENDANTE de scale : ce que le feed-forward ne fournit pas, le PD doit le
# fournir, la somme est la meme. scale ne change pas le couple total, il change l'ERREUR DE POSITION
# necessaire pour l'obtenir -- et c'est precisement cette erreur qui degrade le tracking aujourd'hui.
#
# COUT ASSUME : la policy devient plus dependante du rejeu de dyn_tau a l'inference (le plumbing
# existe et a ete valide -- --task.torque-feedforward-scale doit alors valoir 0.8 aussi, sinon le
# sim2sim commande un couple faux). C'est une SONDE : si le plongeon recule nettement, le mecanisme
# est confirme et on tranchera ensuite entre ce levier et une remontee des gains d'entrainement, qui
# obtient le meme effet sans dette de deploiement.
g1_29dof_joint_pos_dyn_nogrip_ff08 = replace(
    g1_29dof_joint_pos_dyn_nogrip,
    terms={
        **g1_29dof_joint_pos_dyn_nogrip.terms,
        "joint_control": replace(
            g1_29dof_joint_pos_dyn_nogrip.terms["joint_control"],
            params={
                **g1_29dof_joint_pos_dyn_nogrip.terms["joint_control"].params,
                "torque_feedforward": replace(torque_ff_cfg, scale=0.8),
            },
        ),
    },
)

__all__ = [
    "g1_29dof_joint_pos_grip_force",
    "g1_29dof_joint_pos_grip_force_dyn",
    "g1_29dof_joint_pos_grip_force_dyn_qref",
    "g1_29dof_joint_pos_dyn_nogrip",
    "g1_29dof_joint_pos_dyn_nogrip_ff08",
]
