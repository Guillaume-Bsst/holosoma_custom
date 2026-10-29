#!/usr/bin/env bash
# A/B de l'etage 05 : deux leviers isoles issus du diagnostic du run tn5t8rjj.
#
# Usage :  bash run_train_dynamics_ablation.sh {base|qref|refobj|both} [args tyro...]
#
# CE QUE LE RUN DE REFERENCE A MONTRE (950 iterations, clip femto14_box36 halfsphere_torquecap) :
#   corps OK      error_body_pos 0.080 m en baisse, ref_pos jamais cause de mort, episode x7
#   PRISE KO      object_hand_dist plafonne a 0.55 m du centre (surface a 0.18)
#                 les 6 capteurs de contact caisse lisent 0.0000 a TOUTES les iterations
#                 morts par object_pos : 0.5 % -> 49 %
#   qualite plate achieved_frac fige a 0.40 pendant que la longueur d'episode fait x3.5
#                 mpjpe fige a ~0.21 rad (12 deg/joint)
#
# LES QUATRE VARIANTES (meme clip, meme reward, meme URDF caisse+table -- seul le levier change) :
#
#   approach echelle 0.3 ajoutee aux noyaux de contact -- LEVIER PRINCIPAL   acteur 172 critic 322
#   base    reference, a relancer si on veut la courbe de comparaison propre     acteur 172 critic 322
#   qref    ancre PD sur q_ref : action=0 == "suis la reference"                 acteur 172 critic 322
#   refobj  la CIBLE de la caisse entre dans l'observation (+9 dims)             acteur 181 critic 331
#   both    les deux -- a ne lancer qu'apres les deux A/B isoles                 acteur 181 critic 331
#
# QUOI REGARDER, dans cet ordre :
#   1. Env/motion/object_hand_dist   doit descendre sous ~0.25 m (surface de la caisse a 0.18)
#   2. Env/motion/box_force_hands    doit devenir NON NUL -- s'il reste a 0.0000 il n'y a
#                                    toujours aucun contact physique main<->caisse
#   3. Env/reward/achieved_frac      doit MONTER, pas seulement Mean episode length
#   4. Env/motion/mpjpe              doit descendre sous 0.21 rad
#
# ATTENTION DEPLOIEMENT : qref et both rendent la policy dependante d'un rejeu de q_ref indexe sur
# la phase a l'inference, au meme titre que le feed-forward de couple l'est deja pour tau_ref.
# C'est plus simple a cabler (le low-level Unitree prend q nativement) mais ca doit exister.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."   # racine du repo wbt_rl

VARIANT="${1:-approach}"; shift || true
CKPT=""
case "$VARIANT" in
  approach) EXP="g1-29dof-wbt-w-object-actor-grip-force-dyn-approach" ;;
  base)   EXP="g1-29dof-wbt-w-object-actor-grip-force-dyn" ;;
  qref)   EXP="g1-29dof-wbt-w-object-actor-grip-force-dyn-qref" ;;
  refobj) EXP="g1-29dof-wbt-w-object-actor-grip-force-dyn-refobj" ;;
  both)   EXP="g1-29dof-wbt-w-object-actor-grip-force-dyn-both" ;;
  # SEULE variante qui REPREND A CHAUD, et c'est deliberé : elle ne teste pas un levier
  # d'apprentissage de la tache (la tache est acquise -- succes 65 % au checkpoint 20000) mais la
  # QUALITE du mouvement. Repartir de zero re-paierait 21 h pour revenir a un point deja atteint,
  # et les nouveaux poids (feet_slip -2.0 notamment) sont calibres sur une policy qui tient debout :
  # sur une init aleatoire ils creent un attracteur "ne bouge pas". Cf. l'entete du preset
  # g1_29dof_wbt_reward_w_object_dyn_stab dans config_values/wbt/g1/reward.py.
  stab)
    EXP="g1-29dof-wbt-w-object-actor-grip-force-dyn-stab"
    : "${CKPT:=logs/WholeBodyTracking/20260810_181959-g1_29dof_wbt_w_object_actor_grip_force_dyn_approach-locomotion/model_20000.pt}"
    ;;
  # Suite de 'stab', reprise de SON checkpoint 32000. Meme raisonnement que ci-dessus pour la
  # reprise a chaud : la tache est acquise, on ne re-paie pas 20 h pour revenir au meme point.
  stab2)
    EXP="g1-29dof-wbt-w-object-actor-grip-force-dyn-stab2"
    : "${CKPT:=logs/WholeBodyTracking/20260811_175223-g1_29dof_wbt_w_object_actor_grip_force_dyn_stab-locomotion/model_32000.pt}"
    ;;
  # Retrait de la prise scriptee (grip_force). Reprise du MEME checkpoint 32000 que stab2, pour que
  # la seule difference entre les deux runs soit la prothese.
  nogrip)
    EXP="g1-29dof-wbt-w-object-actor-dyn-nogrip"
    : "${CKPT:=logs/WholeBodyTracking/20260811_175223-g1_29dof_wbt_w_object_actor_grip_force_dyn_stab-locomotion/model_32000.pt}"
    ;;
  # Sonde d'autorite actionneur : nogrip + torque_feedforward.scale 0.5 -> 0.8. MEME checkpoint de
  # depart que nogrip, pour que celui-ci serve de controle exact (un seul nombre les separe).
  ff08)
    EXP="g1-29dof-wbt-w-object-actor-dyn-nogrip-ff08"
    : "${CKPT:=logs/WholeBodyTracking/20260811_175223-g1_29dof_wbt_w_object_actor_grip_force_dyn_stab-locomotion/model_32000.pt}"
    ;;
  *) echo "variante inconnue: $VARIANT (attendu: approach|base|qref|refobj|both|stab|stab2|nogrip|ff08)" >&2; exit 2 ;;
esac

: "${NUM_ENVS:=4096}"   # NUM_ENVS=2048 pour faire tourner deux variantes en parallele sur 16 Go
# num-learning-iterations est un NOMBRE D'ITERATIONS EN PLUS, pas une cible absolue : la boucle fait
# range(current_learning_iteration, current + N) et current vient du checkpoint (ppo.py:361-362).
# Sur une reprise a 20000, 20000 ici veut donc dire "aller jusqu'a 40000".
: "${ITERS:=30000}"

if [[ -n "$CKPT" && ! -f "$CKPT" ]]; then
  echo "checkpoint introuvable: $CKPT" >&2; exit 2
fi

source modules/third_party/holosoma_custom/scripts/source_isaacsim_setup.sh

echo "=== ablation etage-05 : variante '$VARIANT' -> exp:$EXP (num_envs=$NUM_ENVS, iters=+$ITERS)"
[[ -n "$CKPT" ]] && echo "=== reprise a chaud depuis $CKPT"

# Pas de --training.checkpoint sur les variantes d'ablation : init aleatoire, comme le run de
# reference, sinon l'A/B ne compare plus les leviers mais les points de depart. 'stab' fait
# exception (voir ci-dessus).
python modules/02_training/holosoma_custom/train_agent.py "exp:$EXP" \
    logger:wandb \
    --training.num-envs "$NUM_ENVS" \
    --algo.config.num-learning-iterations "$ITERS" \
    ${CKPT:+--training.checkpoint "$CKPT"} \
    "$@"
