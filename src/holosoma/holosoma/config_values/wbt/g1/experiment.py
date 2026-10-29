from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values.wbt.g1.action import (
    g1_29dof_joint_pos_dyn_nogrip,
    g1_29dof_joint_pos_dyn_nogrip_ff08,
    g1_29dof_joint_pos_grip_force,
    g1_29dof_joint_pos_grip_force_dyn,
    g1_29dof_joint_pos_grip_force_dyn_qref,
)
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

g1_29dof_wbt = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=30000,
            num_learning_epochs=5,
            save_interval=4000,
            entropy_coef=0.005,
            init_noise_std=1.0,
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            init_at_random_ep_len=True,
            empirical_normalization=True,
            use_symmetry=False,
            actor_optimizer=replace(algo.ppo.config.actor_optimizer, weight_decay=0.000),
            critic_optimizer=replace(algo.ppo.config.critic_optimizer, weight_decay=0.000),
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    ),
    robot=replace(
        robot.g1_29dof,
        control=replace(
            robot.g1_29dof.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_reward,
    nightly=NightlyConfig(
        iterations=8000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.3, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.4, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [0.85, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.7, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.60, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.45, "inf"],
        },
    ),
)

g1_29dof_wbt_fast_sac = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_fast_sac_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.fast_sac,
        config=replace(
            algo.fast_sac.config,
            num_learning_iterations=400000,
            # v_max 20 -> 30 : le critic distributionnel a un SUPPORT BORNE, les atomes du haut
            # saturent si le retour depasse v_max. Somme des poids positifs = 11.5 (corps 5.0 +
            # objet 6.5) -> 0.23/step a dt=0.02 -> retour actualise max 22.9 a gamma=0.99 sur des
            # episodes de 10 s. C'est le max theorique (tous les termes a 1.0 en meme temps, ce qui
            # n'arrive pas), mais la marge etait deja nulle avant (20.9) et on perdrait la
            # resolution du critic exactement dans le regime haute performance. 501 atomes sur
            # +-30 laissent 0.12 de resolution.
            v_max=30.0,
            v_min=-20.0,
            gamma=0.99,  # For motion tracking, high gamma + high num_steps is better
            num_steps=1,
            num_updates=4,
            num_atoms=501,
            policy_frequency=2,
            target_entropy_ratio=0.5,
            tau=0.05,
            use_symmetry=False,
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    ),
    robot=replace(
        robot.g1_29dof,
        control=replace(
            robot.g1_29dof.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_fast_sac_reward,
    nightly=NightlyConfig(
        iterations=200000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.40, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.25, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [1.1, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.35, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.45, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.15, "inf"],
        },
    ),
)

g1_29dof_wbt_w_object = replace(
    g1_29dof_wbt,
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(
            robot.g1_29dof_w_object.asset,
            enable_self_collisions=True,
        ),
        # box36 (cube 0.36 m, demi-tailles 0.18) et NON largebox.obj. Les rewards de contact
        # calculent leur SDF contre grasp_settle.box_half_extents = (0.18, 0.18, 0.18) : le mesh
        # spawne doit correspondre. largebox.obj mesure en realite 0.471 x 0.459 x 0.408
        # (demi-tailles 0.236 / 0.229 / 0.204), soit ~5 cm de plus par cote. Avec ce mesh, une
        # prise parfaitement plate lit une distance signee de ~+0.05 m au lieu de 0, donc
        # object_flat_contact_quality_exp plafonne a exp(-0.05^2/0.03^2) = 0.06 -- le terme est
        # mort sans aucune faute du robot, et object_surface_contact_error_exp prend le meme biais
        # sur sa composante profondeur. Les deux URDF ont la meme masse (0.811 kg), le changement
        # est purement geometrique.
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_box36.urdf",
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

g1_29dof_wbt_fast_sac_w_object = replace(
    g1_29dof_wbt_fast_sac,
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(robot.g1_29dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_box36.urdf",
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

g1_27dof_wbt = replace(
    g1_29dof_wbt,
    training=replace(g1_29dof_wbt.training, name="g1_27dof_wbt_manager"),
    robot=replace(
        robot.g1_27dof,
        control=replace(robot.g1_27dof.control, action_scales_by_effort_limit_over_p_gain=True, action_scale=0.25),
        asset=replace(robot.g1_27dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_27dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    command=command.g1_27dof_wbt_command,
    termination=termination.g1_27dof_wbt_termination,
)

g1_27dof_wbt_w_object = replace(
    g1_27dof_wbt,
    command=command.g1_27dof_wbt_command_w_object,
    robot=replace(
        robot.g1_27dof_w_object,
        asset=replace(robot.g1_27dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.g1_27dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_box36.urdf",
        ),
        init_state=replace(robot.g1_27dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

g1_27dof_wbt_fast_sac = replace(
    g1_29dof_wbt_fast_sac,
    training=replace(g1_29dof_wbt_fast_sac.training, name="g1_27dof_wbt_fast_sac_manager"),
    robot=replace(
        robot.g1_27dof,
        control=replace(robot.g1_27dof.control, action_scales_by_effort_limit_over_p_gain=True, action_scale=0.25),
        asset=replace(robot.g1_27dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_27dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    command=command.g1_27dof_wbt_command,
    termination=termination.g1_27dof_wbt_termination,
)

g1_29dof_wbt_w_object_actor = replace(
    g1_29dof_wbt_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object_actor,
)

# Portage caisse + table avec FORCE DE PRISE REELLE (60 N par main) au lieu du curriculum de
# physicalite. La caisse est physique des le step 0 ; chaque main la presse a
# grip_force.target_force_n des que le flag de contact GT du command term est actif (cf.
# GripForceCfg et JointPositionActionTerm._configure_grip_force).
#
# A NE PAS COMBINER avec kinematic-object-during-contact / physicality-curriculum : les deux
# mecanismes repondent au meme probleme (tenir la caisse) et se neutralisent -- une caisse pilotee
# cinematiquement n'a rien a se faire serrer. Les defauts de GraspSettleConfig laissent bien les
# deux a False, donc ce preset se lance SANS aucun flag de curriculum.
#
# L'observation est la variante ACTEUR : la policy voit la pose de la caisse ET de la table
# (obj_pos_b/obj_ori_b + support_pos_b/support_ori_b), en history_length=1 -- impose par le warm
# start depuis le checkpoint full-loco 3ivghz1e (voir la note en tete de observation.py).
g1_29dof_wbt_w_object_actor_grip_force = replace(
    g1_29dof_wbt_w_object_actor,
    training=replace(
        g1_29dof_wbt_w_object_actor.training,
        name="g1_29dof_wbt_w_object_actor_grip_force",
    ),
    action=g1_29dof_joint_pos_grip_force,
    # Clip CABLE lui aussi : le defaut herite est sub3_largebox_003_mj_w_obj.npz, qui n'a ni table
    # ni contact GT. Celui-ci porte object_* ET support_* complets (support_ref_contact,
    # support_half_extents), donc support_surface_contact_error_exp est reellement actif.
    command=command.g1_29dof_wbt_command_w_object_femto14_box36,
    robot=replace(
        g1_29dof_wbt_w_object_actor.robot,
        object=replace(
            g1_29dof_wbt_w_object_actor.robot.object,
            # La table est CABLEE ici, pas laissee a un flag CLI. Sans support_urdf_path,
            # isaacsim.py:453 ne spawne aucun acteur "support" : le clip porte bien la pose de la
            # table, la policy la voit dans son obs, le reward support_surface_contact la note --
            # mais elle n'existe pas physiquement et la caisse traverserait au depot.
            support_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/support_table.urdf",
        ),
    ),
)

# Variante ETAGE 05 du preset ci-dessus : meme tache, meme robot, meme observation, meme warm start
# -- seule la source d'information change. Le clip est la version enrichie (dyn_* fusionnes depuis
# le run SPIDER femto14_box36_halfsphere_torquecap, couples dans les limites de l'URDF), et trois
# choses s'en servent :
#   1. reward : feet_contact_schedule + feet_slip_on_ref_stance + torque_envelope_penalty, et les
#      deux rewards de contact main<->caisse deviennent bimanuels (cf. reward.py).
#   2. action : la force de prise suit le profil mesure par main au lieu du 60 N constant.
#   3. action : le couple de reference est injecte a 50 % en feed-forward dans la loi PD.
#
# L'espace d'observation et le nombre de parametres sont INCHANGES, donc le warm start depuis
# 3ivghz1e reste utilisable tel quel.
#
# Point de vigilance deploiement : (3) suppose que holosoma_inference sache rejouer le meme couple
# de feed-forward, indexe sur la phase du clip (le low-level Unitree a bien un champ tau). Une
# policy entrainee avec et deployee sans commandera un couple faux. Tant que ce plumbing n'est pas
# fait, ce preset est un preset d'ENTRAINEMENT/etude, pas un candidat au deploiement.
g1_29dof_wbt_w_object_actor_grip_force_dyn = replace(
    g1_29dof_wbt_w_object_actor_grip_force,
    training=replace(
        g1_29dof_wbt_w_object_actor_grip_force.training,
        name="g1_29dof_wbt_w_object_actor_grip_force_dyn",
    ),
    command=command.g1_29dof_wbt_command_w_object_femto14_box36_dyn,
    action=g1_29dof_joint_pos_grip_force_dyn,
    reward=reward.g1_29dof_wbt_reward_w_object_dyn,
    # Seuil objet 0.15 -> 0.45 m : mesure a l'appui dans termination.py. A 350 iterations, 2 morts
    # sur 3 sont "la caisse a bouge", et l'episode meurt au lieu de laisser la policy rattraper.
    termination=termination.g1_29dof_wbt_termination_dyn,
    # Critique enrichi (+17 dims), ACTEUR INCHANGE. Le critique n'etant jamais deploye, tout ce qui
    # reduit la variance de son estimation de valeur est gratuit cote robot. Detail dans
    # observation.py -- le gain principal est motion_phase : avec le RSI, le retour atteignable
    # depend du temps restant dans le clip, que rien ne disait au critique.
    observation=observation.g1_29dof_wbt_observation_w_object_actor_dyn,
)

# ==================================================================================================
# Variantes issues du diagnostic du run tn5t8rjj (950 iterations sur le clip torquecap).
#
# CE QUE LE RUN A MONTRE. Le corps n'est pas le probleme : error_body_pos 0.080 m et en baisse,
# ref_pos jamais cause de mort (le robot ne tombe pas), longueur d'episode x7 (33 -> 236 pas). La
# PRISE est le probleme : object_hand_dist plafonne a 0.55 m du centre de la caisse (dont la surface
# est a 0.18), les six capteurs de contact filtres sur la caisse lisent 0.0000 a TOUTES les
# iterations, et la part des morts par object_pos passe de 0.5 % a 49 %. Les trois rewards de
# contact montent (x7 en 600 iterations) mais depuis un niveau quasi nul, et surtout achieved_frac
# reste PLAT a 0.40 pendant que la longueur d'episode fait x3.5 : la policy survit plus longtemps
# sans mieux faire. mpjpe fige a ~0.21 rad (12 deg par joint) sur toute la fenetre.
#
# DEUX LEVIERS INDEPENDANTS, testables separement. Chacun retire une chose que la policy devait
# reconstruire sans que rien ne la lui donne, et chacun se justifie par le precedent qu'a deja cree
# le feed-forward de couple (donner un prior issu du clip, au prix d'un replay a l'inference).
# ==================================================================================================

# LEVIER PRINCIPAL -- echelle d'APPROCHE sur les noyaux de contact.
#
# C'est celui que le diagnostic soutient le plus directement, et le seul qui explique le fait brut :
# les six capteurs de contact caisse ont lu 0.0000 pendant 950 iterations. Les trois rewards de
# contact main<->caisse sont numeriquement NULS au-dela de 0.25 m (voir le tableau dans reward.py)
# et le robot vit autour de 0.55 m : il est dans une zone ou le reward de prise est plat, donc rien
# ne lui demande de s'approcher. Il VOIT la caisse -- obj_pos_b est dans son observation -- mais
# voir ne cree pas de gradient.
#
# Aucun poids ne change, aucun terme n'est ajoute : seuls les noyaux gagnent une echelle a 0.3.
# C'est le changement le plus conservateur des trois et il ne touche ni l'observation (172/322
# inchangees, donc warm start toujours valide) ni la loi de commande.
g1_29dof_wbt_w_object_actor_grip_force_dyn_approach = replace(
    g1_29dof_wbt_w_object_actor_grip_force_dyn,
    training=replace(
        g1_29dof_wbt_w_object_actor_grip_force_dyn.training,
        name="g1_29dof_wbt_w_object_actor_grip_force_dyn_approach",
    ),
    reward=reward.g1_29dof_wbt_reward_w_object_dyn_approach,
)

# LEVIER (3) -- ancre PD sur q_ref. Observation INCHANGEE (172 dims), seule la definition de
# q_target change. C'est le levier que le diagnostic soutient le plus directement : il faut une
# action de 1.87 en mediane (8.78 au pire, 5.68 sur shoulder_pitch) pour atteindre q_ref depuis la
# pose par defaut, et la policy n'etait qu'a action_mean_abs=0.89 apres 950 iterations. Les bras
# sont precisement les DOF qui demandent les plus grandes actions -- et ce sont eux qui doivent
# amener les mains sur la caisse.
g1_29dof_wbt_w_object_actor_grip_force_dyn_qref = replace(
    g1_29dof_wbt_w_object_actor_grip_force_dyn,
    training=replace(
        g1_29dof_wbt_w_object_actor_grip_force_dyn.training,
        name="g1_29dof_wbt_w_object_actor_grip_force_dyn_qref",
    ),
    action=g1_29dof_joint_pos_grip_force_dyn_qref,
)

# LEVIER (2) -- la CIBLE de la caisse entre dans l'observation (+9 dims acteur et critique). Action
# INCHANGEE. Attaque l'attribution de credit plutot que l'amplitude : 3.5 des 11.0 poids positifs
# notent l'ecart entre la caisse et sa cible, et la cible n'etait un input d'aucun des deux reseaux.
# A la difference des cibles de corps (cinematique directe de q_ref, deja disponible), elle n'est
# pas derivable -- seulement memorisable comme un lookup de 527 frames.
g1_29dof_wbt_w_object_actor_grip_force_dyn_refobj = replace(
    g1_29dof_wbt_w_object_actor_grip_force_dyn,
    training=replace(
        g1_29dof_wbt_w_object_actor_grip_force_dyn.training,
        name="g1_29dof_wbt_w_object_actor_grip_force_dyn_refobj",
    ),
    observation=observation.g1_29dof_wbt_observation_w_object_actor_dyn_refobj,
)

# ==================================================================================================
# STABILITE -- reprise a chaud du checkpoint 20000 de la variante APPROCHE (run fjj74m7c).
#
# Ce preset n'attaque plus la PRISE (elle est acquise : succes 65 %, object_pos_error 0.074 m) mais
# la QUALITE du mouvement, qui est la seule chose que le run n'a pas apprise : les positions sont
# suivies (mpkpe 0.061 m) et les VITESSES ne le sont pas (error_joint_vel 12.58 rad/s contre
# 3.92 rad/s d'amplitude de reference, et -9 % seulement en 20k iterations). Le detail chiffre du
# diagnostic et de chaque poids est dans reward.py, entete de g1_29dof_wbt_reward_w_object_dyn_stab.
#
# UN SEUL AXE DE CHANGEMENT : le reward. Noyaux de vitesse repondères, feet_slip x4,
# torque_envelope x3 (avec marge resserree), feet_contact_schedule et feet_load_share degonfles.
#
# LA RANDOMISATION N'EST PAS TOUCHEE, apres avoir failli l'etre. L'hypothese etait que les bourrades
# (0.75 m/s crete, toutes les 1 a 3 s) depassent l'amplitude propre du mouvement (0.54 m/s median) et
# plafonnent error_body_lin_vel, dont la valeur mesuree (0.63 m/s) tombe pile sur la RMS d'une
# bourrade (uniforme [-0.75, 0.75] par axe -> 0.63 m/s en norme). La coincidence est jolie et elle ne
# prouve rien : _push_robots ajoute une impulsion INSTANTANEE a la vitesse de base, donc avec une
# recuperation de l'ordre de 0.3 s toutes les ~2 s elle ne peut contribuer qu'a ~0.05 m/s d'une
# moyenne TEMPORELLE sur l'episode, pas a 0.63. Baisser la DR sur cette base aurait sacrifie de la
# robustesse sim2real contre une correlation.
#
# Le preset g1_29dof_wbt_randomization_w_object_soft_push existe et reste selectionnable en CLI si on
# veut l'A/B un jour -- notamment si raw_rew_feet_slip_on_ref_stance refuse de descendre malgre le
# poids x4, auquel cas l'interaction "sous bourrade le robot DOIT bouger ses pieds et paie quand
# meme" redeviendra une explication candidate. Ce n'est pas le point de depart.
#
# CONCU POUR LA REPRISE A CHAUD, ce qui impose une contrainte dure : ni l'ACTEUR ni le CRITIQUE ne
# changent de dimension d'observation (172 / 322), sinon PPO.load fait un load_state_dict strict et
# le resume explose sur un mismatch de shape. Aucun terme d'observation n'est donc ajoute, y compris
# ceux qui seraient utiles -- voir la limite connue ci-dessous.
#
# LIMITE CONNUE (assumee). torque_envelope_penalty voit son poids x3, or dyn_tau -- la reference de
# couple contre laquelle il note -- n'est PAS dans l'observation du critique. Le critique ne peut le
# reconstruire que par memorisation depuis motion_phase (qui, lui, est bien present), soit un lookup
# de 327 frames x 29 ddl. C'est faisable pour un MLP 512-256-128 mais ca ajoute de la variance a
# l'estimation de valeur. Le corriger proprement demanderait +29 dims cote critique -- donc un
# redemarrage a froid. C'est precisement l'arbitrage qui justifie de garder le poids modere a -0.15
# plutot que -0.2 : on ne veut pas d'un terme lourd que le critique ne sait pas anticiper.
#
# ATTENDU AU REDEMARRAGE : le budget net passe de +4.51 a ~+3.26 par pas, donc la courbe de reward
# PLONGE d'environ 25 % a l'iteration 20001 et le critique met quelques centaines d'iterations a se
# recaler. C'est normal et attendu, ce n'est pas un echec du run.
g1_29dof_wbt_w_object_actor_grip_force_dyn_stab = replace(
    g1_29dof_wbt_w_object_actor_grip_force_dyn_approach,
    training=replace(
        g1_29dof_wbt_w_object_actor_grip_force_dyn_approach.training,
        name="g1_29dof_wbt_w_object_actor_grip_force_dyn_stab",
    ),
    reward=reward.g1_29dof_wbt_reward_w_object_dyn_stab,
)

# STAB-2 -- suite directe de _stab, reprise a chaud de SON checkpoint 32000 (run 0ftoz226).
#
# Un seul changement par rapport a _stab, et il ne fait que deplacer du budget deja depense :
# action_rate_l2 -0.1 -> -0.01 (58 % du budget negatif rendu, pour un terme mesure comme etant
# integralement du bruit d'exploration) et torque_envelope -0.15 -> -0.5 avec la marge a 1.10.
# Justification chiffree complete dans reward.py, entete de g1_29dof_wbt_reward_w_object_dyn_stab2.
#
# Observation, action, commande, terminaison, randomisation : INCHANGEES depuis _stab, donc la
# reprise a chaud reste valide (le critique se charge en load_state_dict strict).
g1_29dof_wbt_w_object_actor_grip_force_dyn_stab2 = replace(
    g1_29dof_wbt_w_object_actor_grip_force_dyn_stab,
    training=replace(
        g1_29dof_wbt_w_object_actor_grip_force_dyn_stab.training,
        name="g1_29dof_wbt_w_object_actor_grip_force_dyn_stab2",
    ),
    reward=reward.g1_29dof_wbt_reward_w_object_dyn_stab2,
)

# SANS PRISE SCRIPTEE -- reward de _stab2, action sans grip_force. Reprise du checkpoint 32000.
#
# INTENTION : que la prise emerge de la dynamique resolue par l'etage 05 au lieu d'etre imposee par
# un serrage de 60 N scripte sur le flag de contact GT du clip. La prothese court-circuitait
# precisement ce que le clip enrichi doit enseigner, et elle porte une dette de deploiement (il faut
# rejouer le profil indexe sur la phase). Detail du mecanisme et de la mesure de risque dans
# action.py, entete de g1_29dof_joint_pos_dyn_nogrip.
#
# RISQUE ASSUME, CHIFFRE : au checkpoint 32000 la caisse est tenue entre les avant-bras avec ~16 N
# de force normale par cote pour un poids de 8-12 N, soit une marge de frottement de ~1 sur une
# bonne partie de la plage randomisee ([0.1, 0.6] cote caisse). Retirer les 60 N commandes retire la
# source de cette force. La caisse peut tomber -- c'est le resultat a mesurer, pas un accident.
#
# A SURVEILLER EN PRIORITE, dans cet ordre :
#   Env/motion/box_force_forearms      s'effondre-t-il vers 0 ? (etait ~33 N)
#   Env/termination_cause/object_pos   explose-t-il ? (etait 0.29)
#   Env/motion/object_pos_error        depasse-t-il durablement 0.10 m ? (etait 0.074)
# Si les trois virent au rouge et n'ont pas commence a revenir a ~2000 iterations, la prise ne
# reemerge pas seule et il faudra un annealing de target_force_n plutot qu'un retrait sec.
#
# La reprise a chaud reste valide : action 29 DDL, observations inchangees (172 / 322).
g1_29dof_wbt_w_object_actor_dyn_nogrip = replace(
    g1_29dof_wbt_w_object_actor_grip_force_dyn_stab2,
    training=replace(
        g1_29dof_wbt_w_object_actor_grip_force_dyn_stab2.training,
        name="g1_29dof_wbt_w_object_actor_dyn_nogrip",
    ),
    action=g1_29dof_joint_pos_dyn_nogrip,
)

# SONDE FEEDFORWARD 0.8. Strictement _dyn_nogrip avec torque_feedforward.scale 0.5 -> 0.8, repris du
# MEME checkpoint 32000 : nogrip devient le controle exact, un seul nombre les separe.
#
# C'est une SONDE, pas une reparation. Elle repond a une question binaire : le plongeon en avant a la
# prise vient-il d'un manque d'autorite a la cheville ? Justification chiffree complete dans
# action.py, entete de g1_29dof_joint_pos_dyn_nogrip_ff08.
#
# LECTURE. Si le plongeon recule nettement en sim2sim et que error_joint_vel decroche enfin de son
# asymptote (11.74 sur les 15 000 dernieres iterations), le mecanisme est confirme et il faudra
# trancher entre ce levier et une remontee des gains d'entrainement -- meme effet, mais sans dette de
# deploiement. Si rien ne bouge, l'autorite n'etait pas le facteur limitant et il faut regarder la
# trajectoire elle-meme (vitesse de la prise, bras de levier de la caisse).
#
# ATTENTION SIM2SIM : tout test d'un checkpoint de ce run doit passer
# --task.torque-feedforward-scale 0.8, sinon l'inference commande un couple different de celui de
# l'entrainement et la mesure ne veut rien dire.
g1_29dof_wbt_w_object_actor_dyn_nogrip_ff08 = replace(
    g1_29dof_wbt_w_object_actor_dyn_nogrip,
    training=replace(
        g1_29dof_wbt_w_object_actor_dyn_nogrip.training,
        name="g1_29dof_wbt_w_object_actor_dyn_nogrip_ff08",
    ),
    action=g1_29dof_joint_pos_dyn_nogrip_ff08,
)

# LES DEUX. A ne lancer qu'apres les deux A/B isoles, sinon un gain ne sera pas attribuable.
g1_29dof_wbt_w_object_actor_grip_force_dyn_both = replace(
    g1_29dof_wbt_w_object_actor_grip_force_dyn,
    training=replace(
        g1_29dof_wbt_w_object_actor_grip_force_dyn.training,
        name="g1_29dof_wbt_w_object_actor_grip_force_dyn_both",
    ),
    action=g1_29dof_joint_pos_grip_force_dyn_qref,
    observation=observation.g1_29dof_wbt_observation_w_object_actor_dyn_refobj,
)

g1_27dof_wbt_w_object_actor = replace(
    g1_27dof_wbt_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object_actor,
)

g1_27dof_wbt_fast_sac_w_object = replace(
    g1_27dof_wbt_fast_sac,
    command=command.g1_27dof_wbt_command_w_object,
    robot=replace(
        robot.g1_27dof_w_object,
        asset=replace(robot.g1_27dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.g1_27dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_box36.urdf",
        ),
        init_state=replace(robot.g1_27dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

__all__ = [
    "g1_29dof_wbt",
    "g1_29dof_wbt_fast_sac",
    "g1_29dof_wbt_fast_sac_w_object",
    "g1_29dof_wbt_w_object",
    "g1_29dof_wbt_w_object_actor",
    "g1_29dof_wbt_w_object_actor_grip_force",
    "g1_29dof_wbt_w_object_actor_grip_force_dyn",
    "g1_29dof_wbt_w_object_actor_grip_force_dyn_approach",
    "g1_29dof_wbt_w_object_actor_grip_force_dyn_stab",
    "g1_29dof_wbt_w_object_actor_grip_force_dyn_stab2",
    "g1_29dof_wbt_w_object_actor_dyn_nogrip",
    "g1_29dof_wbt_w_object_actor_dyn_nogrip_ff08",
    "g1_29dof_wbt_w_object_actor_grip_force_dyn_qref",
    "g1_29dof_wbt_w_object_actor_grip_force_dyn_refobj",
    "g1_29dof_wbt_w_object_actor_grip_force_dyn_both",
    "g1_27dof_wbt",
    "g1_27dof_wbt_w_object",
    "g1_27dof_wbt_w_object_actor",
    "g1_27dof_wbt_fast_sac",
    "g1_27dof_wbt_fast_sac_w_object",
]

"""
Example 1: Robot only:
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt

Example 2: Robot+Object:
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object

Example 3: Robot+Terrain:
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt \
  terrain:terrain-load-obj \
  --terrain.terrain-term.obj-file-path="holosoma/data/motions/g1_29dof/whole_body_tracking/terrain_slope.obj" \
  --command.setup_terms.motion_command.params.motion_config.motion_file\
="holosoma/data/motions/g1_29dof/whole_body_tracking/motion_crawl_slope.npz" \
  --simulator.config.scene.env_spacing=0.0
"""


# =====================================================================================
# DynaRetarget (arXiv:2602.06827) reward table, on the femto14 box-carry task.
#
# Base is g1_29dof_wbt_w_object_actor -- the preset behind wandb run vjyjtipa
# (guibsst-inria/WholeBodyTracking, 2026-07-22, commit 75146d08) -- with the reward set
# swapped for the paper's Table IV and NOTHING else changed, so an A/B against that run
# attributes any difference to the reward design alone.
#
# The exact source tree of vjyjtipa is NOT recoverable (`git fetch 75146d08` ->
# "upload-pack: not our ref"), so the baseline was reconstructed from the run's own
# resolved W&B config. One term of that run no longer exists in this tree:
# motion_relative_hand_object_position_error_exp (w=0.3). It is not in Table IV either,
# so it is simply absent here rather than replaced.
#
# Already satisfied by the base preset, and left untouched on purpose:
#   * residual action space -- JointPositionActionTerm anchors on the reference
#     ("action of 0 means track the reference", joint_control.py:359) at scale=1.
#   * no object-tracking termination -- the baseline terminates on timeout and
#     bad_tracking only, which is what the paper explicitly asks for.
#   * adaptive sampling -- use_adaptive_timesteps_sampler is already on in the
#     wbt command preset.
#
# NOT yet wired here, and the reason each is a separate decision -- see the branch notes:
#   * object domain randomization (mass / friction / pushes / reset-pose spread). The
#     startup terms exist (randomize_object_rigid_body_{material,mass}_startup, ...) but
#     the baseline ran with randomization.terms EMPTY, so switching them on is a second
#     change on top of the reward one and would confound the A/B.
#   * the paper's 8192 envs x 10000 iterations. The baseline ran 4096 envs; env count
#     changes the effective batch, so it is left at the baseline value.
# =====================================================================================
g1_29dof_wbt_w_object_actor_dynaretarget = replace(
    g1_29dof_wbt_w_object_actor,
    training=replace(
        g1_29dof_wbt_w_object_actor.training,
        name="g1_29dof_wbt_w_object_actor_dynaretarget",
    ),
    reward=reward.g1_29dof_wbt_reward_dynaretarget,
    robot=replace(
        g1_29dof_wbt_w_object_actor.robot,
        object=replace(
            g1_29dof_wbt_w_object_actor.robot.object,
            # The table is WIRED here, not left to a CLI flag. Without support_urdf_path
            # isaacsim.py spawns no "support" actor: the clip carries the table pose, the
            # policy sees it in its observation -- but it does not exist physically and the
            # box falls through at deposit. The baseline run vjyjtipa passed it on the command
            # line; since then the tree moved the support POSE from CLI args to the clip's own
            # support_pos_w / support_quat_w, so both halves have to line up.
            support_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/support_table.urdf",
        ),
    ),
    # Object-tracking terminations disabled per the paper; robot fall detection kept.
    # See the block above g1_29dof_wbt_termination_dynaretarget for why this is not
    # simply "drop the bad_tracking term".
    termination=termination.g1_29dof_wbt_termination_dynaretarget,
)


# Same preset PLUS the paper's object domain randomization (section E): friction, mass,
# random pushes on the object, and the phase-conditional pose/velocity reset. Kept separate
# from the preset above so the reward table can be A/B'd on its own first -- the baseline
# ran with randomization disabled entirely, so folding DR in would change two things at once.
g1_29dof_wbt_w_object_actor_dynaretarget_dr = replace(
    g1_29dof_wbt_w_object_actor_dynaretarget,
    training=replace(
        g1_29dof_wbt_w_object_actor_dynaretarget.training,
        name="g1_29dof_wbt_w_object_actor_dynaretarget_dr",
    ),
    randomization=randomization.g1_29dof_wbt_randomization_dynaretarget,
    command=command.g1_29dof_wbt_command_w_object_femto14_box36_dynaretarget,
)

# Same preset, reference motion swapped for SBTO's own dynamically-feasible solve instead of
# the kinematic-only clip (see command.g1_29dof_wbt_command_w_object_femto14_box36_dynaretarget_sbto
# for what that changes). Reward/observation/randomization/termination are untouched -- verified
# against this exact reward table (g1_29dof_wbt_reward_dynaretarget only reads joint_pos/joint_vel/
# body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w/object_*_w/dyn_obj_contact_lr, and the
# observation set only additionally reads dyn_foot_contact_lr -- none of support_ref_*/object_ref_*/
# support_pointcloud/dyn_tau/dyn_grip_force_lr/dyn_foot_grf_lr, which this file does not carry).
g1_29dof_wbt_w_object_actor_dynaretarget_dr_sbto = replace(
    g1_29dof_wbt_w_object_actor_dynaretarget_dr,
    training=replace(
        g1_29dof_wbt_w_object_actor_dynaretarget_dr.training,
        name="g1_29dof_wbt_w_object_actor_dynaretarget_dr_sbto",
    ),
    command=command.g1_29dof_wbt_command_w_object_femto14_box36_dynaretarget_sbto,
)

# Same, reward table swapped for the HDMI-augmented one (see reward.g1_29dof_wbt_reward_dynaretarget_hdmi
# for why: object_hand_dist regressed and stalled ~0.6-1.0m over the first 4500 iterations of the plain
# run, the paper's own contact term (object_contact_match) is a zero-gradient gate below real contact).
g1_29dof_wbt_w_object_actor_dynaretarget_dr_sbto_hdmi = replace(
    g1_29dof_wbt_w_object_actor_dynaretarget_dr_sbto,
    training=replace(
        g1_29dof_wbt_w_object_actor_dynaretarget_dr_sbto.training,
        name="g1_29dof_wbt_w_object_actor_dynaretarget_dr_sbto_hdmi",
    ),
    reward=reward.g1_29dof_wbt_reward_dynaretarget_hdmi,
)
