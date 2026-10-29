"""Whole Body Tracking reward presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

g1_29dof_wbt_reward = RewardManagerCfg(
    terms={
        # Motion tracking rewards - global reference frame
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=0.5,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        # Motion tracking rewards - relative body frame
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
        # Motion tracking rewards - body velocities
        "motion_global_body_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={"sigma": 1.0},
            weight=1.0,
        ),
        "motion_global_body_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={"sigma": 3.14},
            weight=1.0,
        ),
        # Regularization rewards
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-0.1,
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9},
            weight=-10.0,
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": (
                    "^(?!left_foot_contact_point$)(?!right_foot_contact_point$)"
                    "(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                    "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"
                ),
                # Reference-masked (see the w_object preset below for the original rationale):
                # exempts a body the REFERENCE itself puts on the ground at that frame (e.g. a knee
                # or the torso during a grounding/floor-contact motion), while still penalising a
                # body touching something the reference never puts there. Needed for CMU clips like
                # 133_01 that spend a lot of the motion on the ground -- without this, the plain
                # regex penalty fights the tracking reward on every frame the reference is grounded.
                "use_reference_mask": True,
                "ref_contact_margin": 0.15,
                "ground_margin": 0.10,
            },
            weight=-0.1,
        ),
    }
)

g1_29dof_wbt_fast_sac_reward = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-1.0,
        ),
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        # Motion tracking rewards - relative body frame
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=2.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
    }
)

g1_29dof_wbt_reward_w_object = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        # Motion tracking rewards - global reference frame
        "object_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        # Meme fonction que le terme "coarse" ci-dessus, sigma resserre : reward MULTI-ECHELLE sur
        # la position de la box. Le terme large (sigma=0.3) est plat la ou ca compte -- la
        # termination bad_object_pos coupe a 0.15 m, et exp(-0.15^2/0.3^2)=0.78 : sur TOUTE la bande
        # ou l'episode est encore vivant le coarse ne varie que de 1.00 -> 0.78, soit ~0.02 par cm.
        # Resultat observe : la box stagne quelques cm sous la hauteur de portage, le gradient ne
        # paie pas l'effort des derniers cm. sigma=0.12 redonne de la dynamique sur cette bande
        # (1.00 -> 0.21 entre 0 et 15 cm, ~3.5x le gradient du coarse a 10 cm). Le coarse reste pour
        # le champ lointain (reapproche apres une derive / phase assistee a alpha eleve).
        "object_global_ref_position_error_fine_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_position_error_exp",
            params={"sigma": 0.12},
            weight=1.0,
        ),
        "object_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
        # Dense grasp signal: hand<->object relative position on contact frames (neutral on free
        # frames). Complements the global object terms, which say nothing about WHERE in the hand
        # frame the object should be while carried.
        # Poids 0.5 (et non 1.0) : trois termes notent la geometrie main<->box (celui-ci,
        # object_surface_contact et object_flat_contact_quality). Celui-ci est le moins informatif
        # des trois -- une pose relative RIGIDE ne distingue pas "bon endroit sur la box" de
        # "mauvais endroit au bon offset", ce que surface_contact tranche via le witness. Il reste
        # utile comme signal dense/lisse, mais il ne merite pas le meme poids que les deux autres.
        "object_grasp_relative_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_grasp_relative_error_exp",
            params={"sigma": 0.1},
            weight=0.5,
        ),
        # WHERE on the box surface + how deep the current contact is, vs the retargeting reference
        # (HoloV2's own witness/distance point-cloud contact fields, baked per-frame -- see
        # gvhmr-fp-pipeline/contact_from_retarget.py). Neutral automatically if the loaded motion
        # doesn't carry a reference witness (older/synthetic clips).
        # sigma_dist 0.05 -> 0.08 : ce terme est un PRODUIT de deux exp (geodesique x profondeur),
        # donc les deux pentes se cumulent. Et d_ref vient du witness FoundationPose du retargeting,
        # lui-meme bruite : a 0.05 la tolerance en profondeur etait plus serree que le bruit de sa
        # propre reference, ce qui note du bruit plutot que la policy.
        "object_surface_contact_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_surface_contact_error_exp",
            params={"sigma_geodesic": 0.1, "sigma_dist": 0.08},
            weight=1.0,
        ),
        # Contact QUALITY (option 2): reward the hand's flat-face keypoints to be flush against the
        # box (patch contact that resists rotation). Independent of the reference witness -- teaches
        # HOW to grip, pairs with the physicality curriculum. sigma=0.03 m (flush within a few cm).
        "object_flat_contact_quality_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_flat_contact_quality_exp",
            params={"sigma": 0.03},
            weight=1.0,
        ),
        # Compagnon LARGE du terme ci-dessus (meme func, sigma 0.10, poids moitie). sigma=0.03 seul
        # est inapprenable : une policy qui n'a pas encore trouve le patch plat est a >=5 cm de rms,
        # ou exp(-0.05^2/0.03^2)=0.06, et a 8 cm 0.0008 -- numeriquement nul, donc aucun gradient
        # pour DECOUVRIR le patch (et le terme vaut deja 0 hors contact, il n'y a pas d'autre
        # source de signal). Le sigma=0.10 porte de 8 cm a 2 cm (0.53 -> 0.96), le sigma=0.03 prend
        # le relais sur les derniers cm. Meme construction multi-echelle que
        # object_global_ref_position_error_fine_exp.
        "object_flat_contact_quality_coarse_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_flat_contact_quality_exp",
            params={"sigma": 0.10},
            weight=0.5,
        ),
        # robot<->TABLE : même reward de contact de surface (witness + SDF) que pour la box, mais
        # sur l'objet statique table. Apprend au robot à placer sa main au bon endroit près de la
        # table (approche/dépôt) et à ne pas foncer dedans. Neutre si le clip ne porte pas de table.
        # Poids plus faible que la box (signal secondaire de placement, pas la tâche principale).
        "support_surface_contact_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:support_surface_contact_error_exp",
            params={"sigma_geodesic": 0.1, "sigma_dist": 0.05},
            weight=0.5,
        ),
        # Override du terme herite : un contact est "indesirable" s'il n'est PAS dans le
        # retargeting a cette frame, pas s'il est dans une liste de noms de corps. Le regex seul
        # n'excluait que pieds/chevilles/wrist_yaw ; or en portant un cube de 0.36 m la REFERENCE
        # elle-meme colle l'avant-bras (elbow_link) et le poignet (wrist_roll/pitch) contre la box,
        # et le terme renvoie un COMPTE de corps en contact -> ~-0.4/s en permanence sur exactement
        # le comportement qu'on entraine. Le mask (cf. UndesiredContacts) exempte par frame tout
        # corps que la reference met au contact du sol / de la box / de la table, et continue de
        # punir un corps loin de tout dans la reference qui touche quand meme en sim (genou au sol,
        # torse dans la table) -- quel que soit son nom. Le regex reste comme plancher inconditionnel.
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                **g1_29dof_wbt_reward.terms["undesired_contacts"].params,
                "use_reference_mask": True,
                # Marges generreuses : body_pos_w donne l'ORIGINE du link, pas sa surface de
                # collision, donc la marge doit absorber le rayon du link. L'asymetrie justifie de
                # viser large -- une exemption en trop ne fait que ne pas punir un contact, une
                # punition en trop combat activement la tache.
                "ref_contact_margin": 0.15,
                "ground_margin": 0.10,
            },
            weight=-0.1,
        ),
    }
)

# Portage caisse + table, avec les signaux de l'etage 05 (SPIDER) fusionnes dans le clip par
# wbt_rl/scripts/merge_dynamics.py (cf. motion_config_w_object_femto14_box36_dyn).
#
# Les termes herites du preset w_object ne changent PAS de poids : object_grasp_relative_error_exp
# et object_flat_contact_quality_exp deviennent simplement bimanuels d'eux-memes des que le clip
# porte dyn_obj_contact_lr (cf. _bimanual_available), en gardant leur echelle 0..1 -- la moyenne sur
# les mains en contact, pas la somme. Un clip sans sidecar retombe automatiquement sur l'ancre
# unique. Seuls les trois termes ci-dessous sont nouveaux, et chacun renvoie 0 sans sidecar, donc ce
# preset est utilisable tel quel sur un clip non enrichi (il vaut alors exactement w_object).
_GATE = "holosoma.managers.reward.terms.wbt:"

# FUSION DES PAIRES COARSE/FINE. Le preset w_object configure DEUX fois la meme fonction a deux
# sigmas et laisse la somme ponderee faire le melange multi-echelle. Ca marche, mais ca depense deux
# poids pleins pour UNE grandeur physique : 2.0 pour la position de la caisse (autant que tout le
# tracking relatif du corps) et 1.5 pour la qualite de contact plat. Cumule sur 21 termes, ca a fait
# passer le budget "caisse" (6.5) devant le budget "corps" (5.0) sans que ce soit un choix.
#
# _multiscale_exp fait la moyenne des noyaux A L'INTERIEUR d'un terme : profil de gradient
# identique, borne a 0..1, un seul poids. sigma_weights conserve l'emphase que la version eclatee
# encodait dans ses deux poids (2:1 pour fin:large sur le contact plat, ou 1.0 et 0.5).
# Economie : 2.0 -> 1.0 et 1.5 -> 1.0, soit 1.5 de budget rendu, zero gradient perdu.
_merged = {
    k: v
    for k, v in g1_29dof_wbt_reward_w_object.terms.items()
    if k not in ("object_global_ref_position_error_fine_exp", "object_flat_contact_quality_coarse_exp")
}

g1_29dof_wbt_reward_w_object_dyn = RewardManagerCfg(
    terms={
        **_merged,
        # Ex object_global_ref_position_error_exp (sigma 0.3, w 1.0) + _fine_exp (sigma 0.12, w 1.0).
        "object_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_position_error_exp",
            params={"sigmas": (0.3, 0.12)},
            weight=1.0,
        ),
        # Ex object_flat_contact_quality_exp (sigma 0.03, w 1.0) + _coarse_exp (sigma 0.10, w 0.5).
        # Le 2:1 reproduit exactement le rapport des deux anciens poids : le sigma large sert a
        # DECOUVRIR le patch (a 8 cm le sigma=0.03 vaut 0.0008, numeriquement nul), le sigma fin
        # prend le relais sur les derniers centimetres.
        "object_flat_contact_quality_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_flat_contact_quality_exp",
            params={"sigmas": (0.03, 0.10), "sigma_weights": (2.0, 1.0)},
            weight=1.0,
            achievable_gate=f"{_GATE}gate_object_contact",
        ),
        # limits_dof_pos : -10.0 representait 93 % du budget negatif, deux ordres de grandeur
        # au-dessus de tout le reste, sur une somme brute de RADIANS non normalisee (0.1 rad de
        # depassement ne veut pas dire la meme chose sur un poignet a 0.5 rad de course que sur une
        # epaule a 5 rad). normalize=True divise par la marge butee souple -> butee dure : 1.0 = "ce
        # joint a consomme toute sa marge", identiquement pour tous. Le terme se lit alors comme un
        # NOMBRE de DOF satures, et -1.0 est interpretable seul.
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9, "normalize": True},
            weight=-1.0,
        ),
        # Gates de reward atteignable (diagnostic pur, cf. RewardTermCfg.achievable_gate) : ces
        # termes valent 0 quand la reference n'a pas de contact a noter, donc le budget disponible
        # varie le long du clip et le scalaire de reward n'est pas comparable d'une phase a l'autre.
        "object_grasp_relative_error_exp": replace(
            _merged["object_grasp_relative_error_exp"], achievable_gate=f"{_GATE}gate_object_contact"
        ),
        "object_surface_contact_error_exp": replace(
            _merged["object_surface_contact_error_exp"], achievable_gate=f"{_GATE}gate_object_contact"
        ),
        "support_surface_contact_error_exp": replace(
            _merged["support_surface_contact_error_exp"], achievable_gate=f"{_GATE}gate_support_contact"
        ),
        # Calendrier d'appui par pied. Rien dans les termes de tracking ne dit QUEL pied doit porter
        # a QUEL instant -- ils contraignent ou sont les liens, ce qui laisse la policy libre
        # d'inventer sa propre sequence d'appuis (trainage, double appui permanent) tant que les
        # positions y sont approximativement. Poids 0.5 : c'est un terme dense, borne a 1, qui doit
        # informer la demarche sans concurrencer le tracking de corps (poids 1.0 chacun).
        "feet_contact_schedule": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:FeetContactSchedule",
            params={"threshold": 1.0},
            weight=0.5,
            achievable_gate="holosoma.managers.reward.terms.wbt:gate_dyn_sidecar",
        ),
        # Repartition de charge gauche/droite en double appui. Complement direct du terme ci-dessus,
        # qui ne lit que le booleen : sur femto14_box36 la reference est en double appui sur 257 des
        # 327 frames (79 %), donc 4 frames sur 5 le calendrier vaut 1.0 quoi que fasse la policy de
        # son poids. C'est la GRF (dyn_foot_grf_lr) qui informe ces frames-la, sous forme de RATIO
        # et non de force absolue -- la reference pique a 2427 N sur un pied (~7x le poids) aux
        # transitions de contact, meme artefact kp=500 que dyn_tau, et le ratio l'annule.
        # Poids 0.5, aligne sur feet_contact_schedule : meme famille, meme echelle 0..1, et ne doit
        # pas concurrencer le tracking de corps (1.0).
        # sigma=0.15 mesure sur le clip : la reference balaie une part gauche de 0.02 a 1.00, et le
        # calibrage se fait contre la policy PARESSEUSE (poids fige a 50/50, l'optimum trivial de ce
        # terme). Elle marque 0.79 a sigma=0.25 -- seulement 0.21 d'ecart avec la policy parfaite,
        # trop peu pour payer un transfert de poids actif. sigma=0.15 la ramene a 0.63 (0.37
        # d'ecart) tout en laissant 0.64 a 10 % d'erreur de repartition, donc le quasi-juste reste
        # paye. sigma=0.10 serait le piege inverse : 0.002 a 25 % d'erreur, numeriquement nul, donc
        # aucun gradient pour DECOUVRIR la bonne repartition depuis un depart quelconque -- exactement
        # l'ecueil documente sur object_flat_contact_quality_exp.
        "feet_load_share": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:FeetLoadShare",
            params={"sigma": 0.15},
            weight=0.5,
            achievable_gate="holosoma.managers.reward.terms.wbt:gate_feet_double_support",
        ),
        # Anti-glissement, actif uniquement quand la REFERENCE a le pied plante (une jambe en vol
        # doit aller vite). Le clip femto14_box36 glisse a 0.020 m/s median en appui -- soit un cout
        # de 0.5 * 0.0004 = 2e-4/pas au niveau observe, negligeable, et qui ne mord que si la policy
        # se met a patiner franchement (0.2 m/s -> 0.02). C'est l'intention : une barriere, pas une
        # taxe permanente.
        "feet_slip_on_ref_stance": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:FeetSlipOnRefStance",
            weight=-0.5,
        ),
        # Enveloppe de couple UNILATERALE contre le couple que le mouvement demande reellement
        # (dyn_tau). Ne punit jamais le sous-couple -- cf. le docstring de torque_envelope_penalty :
        # dyn_tau vient d'un solve a kp=500 qui peut suivre des transitions de contact plus
        # sechement que les vrais gains, donc ses pics ne sont pas atteignables et un terme de
        # tracking punirait la policy de ne pas faire l'impossible.
        # margin=1.5 : 50 % de marge au-dessus de la reference avant toute penalite. Poids -0.05
        # sur une somme normalisee de 29 relu^2 -- volontairement petit, c'est un regularisateur
        # (remplacer un proxy d'action-rate par la vraie grandeur physique), pas un objectif.
        "torque_envelope_penalty": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:torque_envelope_penalty",
            params={"margin": 1.5},
            weight=-0.05,
        ),
    }
)

# ==================================================================================================
# Variante APPROCHE : meme reward que ci-dessus, avec une echelle LARGE ajoutee aux trois noyaux de
# contact main<->caisse. Aucun poids ne change, aucun terme n'est ajoute ni retire -- seule la forme
# des noyaux change, donc le budget (11.0 positifs) et l'equilibre corps/objet sont identiques.
#
# MESURE QUI LA MOTIVE (run tn5t8rjj, 950 iterations, physique complete). Valeur des rewards de
# contact en fonction de l'erreur main<->cible :
#
#   erreur (m)                                 0.55    0.45    0.35    0.25    0.20    0.15    0.10
#   object_grasp_relative      (sigma 0.10)  0.0000  0.0000  0.0000  0.0019  0.0183  0.1054  0.3679
#   object_surface_contact geo (sigma 0.10)  0.0000  0.0000  0.0000  0.0019  0.0183  0.1054  0.3679
#   object_flat_contact  (sigma 0.03+0.10)   0.0000  0.0000  0.0000  0.0006  0.0061  0.0351  0.1226
#
# Les mains sont restees autour de 0.55 m du centre de la caisse et les six capteurs de contact ont
# lu 0.0000 pendant tout le run : la policy a vecu entierement DANS la zone ou ces trois termes sont
# plats. Elle voit la caisse (obj_pos_b est dans l'observation acteur) mais aucun gradient ne lui
# demandait de s'en rapprocher -- voir ne suffit pas, il faut que le reward varie. Le seul signal
# qui tirait le poignet vers sa cible etait motion_relative_body_position_error_exp, moyenne sur 14
# corps : ~0.09 de gain pour diviser par deux l'erreur d'un poignet, contre ~0.67 une fois au
# contact, avec 30 cm de plateau entre les deux.
#
# sigma=0.30 (0.035 / 0.26 / 0.50 a 0.55 / 0.35 / 0.25 m) redonne une pente continue sur toute la
# zone morte, les noyaux fins gardant la precision finale. Les termes restent bornes 0..1 et valent
# toujours exactement 0 hors frames de contact de reference : l'attracteur "mourir avant le contact"
# n'est pas reintroduit.
#
# C'est le meme raisonnement que le compagnon large de object_flat_contact_quality_exp, mais
# calibre sur la distance REELLE de depart (0.55 m) et non sur 8 cm.
g1_29dof_wbt_reward_w_object_dyn_approach = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward_w_object_dyn.terms,
        "object_grasp_relative_error_exp": replace(
            g1_29dof_wbt_reward_w_object_dyn.terms["object_grasp_relative_error_exp"],
            params={"sigmas": (0.1, 0.3)},
        ),
        # 3 echelles : 0.03 precision du patch, 0.10 derniers centimetres, 0.30 approche. Le 2:1
        # entre les deux premieres est conserve (c'etait le rapport des deux anciens poids) et
        # l'echelle d'approche prend 1.0 -- assez pour creer la pente, pas assez pour payer une
        # policy qui resterait a 30 cm.
        "object_flat_contact_quality_exp": replace(
            g1_29dof_wbt_reward_w_object_dyn.terms["object_flat_contact_quality_exp"],
            params={"sigmas": (0.03, 0.10, 0.30), "sigma_weights": (2.0, 1.0, 1.0)},
        ),
        "object_surface_contact_error_exp": replace(
            g1_29dof_wbt_reward_w_object_dyn.terms["object_surface_contact_error_exp"],
            params={"sigmas_geodesic": (0.1, 0.3), "sigma_dist": 0.08},
        ),
    }
)

# ==================================================================================================
# Variante STABILITE -- reprise a chaud du checkpoint 20000 du run fjj74m7c (20 500 iterations,
# 1.9 G samples, 21 h sur 4096 envs).
#
# CE QUE LE RUN A MONTRE. La tache est apprise : succes 65 %, mpkpe 0.061 m, mpjpe 0.179 rad,
# object_pos_error 0.074 m, et ca progressait encore (succes 0.593 -> 0.653 sur les 4 dernieres k
# iterations). Ce qui ne va pas est ailleurs, et c'est mesurable -- les VITESSES ne sont pas
# suivies, alors que les positions le sont :
#
#   grandeur                     sim (erreur)   reference (mediane)   rapport
#   error_joint_vel              12.58 rad/s    3.92 rad/s            3.2x
#   error_body_ang_vel            3.27 rad/s    1.26 rad/s            2.6x
#   error_body_lin_vel            0.63 m/s      0.54 m/s              1.2x
#
# Bonnes poses, mauvaises vitesses = mouvement saccade. Et ca ne bougeait PAS : error_joint_vel
# 13.86 -> 12.58 sur 20k iterations (-9 %) pendant que le tracking de position relatif passait de
# 0.86 a 0.94. Continuer la meme run n'y changera rien : le probleme est dans le budget de reward,
# pas dans l'optimisation.
#
# POURQUOI RIEN NE S'Y OPPOSAIT. Budget mesure au checkpoint 20000, en valeur PAR PAS (la valeur
# brute non ponderee du terme, corrigee de la dilution par max_episode_length_s) :
#
#   positifs 6.15   negatifs 1.64
#   dont action_rate_l2 : 14.12 de valeur brute -> -1.41, soit 86 % de TOUT le budget negatif.
#
# Sauf que ce -1.41 ne penalise rien. action_rate_l2 mesure sum_29 (a_t - a_{t-1})^2 sur l'action
# ECHANTILLONNEE, bruit d'exploration compris. Avec mean_noise_std = 0.5045 et un bruit i.i.d.
# entre pas, le bruit SEUL predit 2 * 29 * 0.5045^2 = 14.76, contre 14.12 mesure : la variation de
# l'action MOYENNE -- celle qui part dans l'ONNX -- est sous le plancher de bruit. Ce terme est donc
# une taxe quasi constante sur sigma, pas sur la regularite de la policy deployee ; le monter
# ecraserait l'exploration sans rien lisser. Il reste a -0.1, inchange, et ne doit pas etre lu comme
# un regularisateur.
#
# La regularisation REELLE valait donc -0.22/pas sur 6.15, soit 3.6 % du budget. Ce preset la porte
# a un niveau ou elle mord, sans toucher ni a l'observation ni a la loi de commande (contrainte
# imposee par la reprise a chaud : le critique est charge par load_state_dict strict, toute
# dimension d'obs qui bouge fait exploser le resume).
#
# BUDGET ATTENDU APRES MODIFS : positifs ~6.46, negatifs ~2.24, net ~+4.22 (contre +4.51).
# Le net reste franchement positif, donc aucun attracteur "mourir tot rapporte plus que vivre" --
# verification obligatoire des lors qu'on augmente des penalites avec only_positive_rewards=False.
# Et l'echelle globale bouge peu (-6 %), ce qui limite le recalage que le critique doit encaisser
# a la reprise : c'est un effet secondaire heureux du choix de corriger les POIDS des termes de
# vitesse plutot que la forme de leurs noyaux (voir (1) ci-dessous).
g1_29dof_wbt_reward_w_object_dyn_stab = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward_w_object_dyn_approach.terms,
        # --- (1) les vitesses : c'est le POIDS qui manque, pas le gradient ------------------------
        # Le reflexe naturel serait de resserrer sigma (3.14 est pi, une constante mathematique,
        # alors que la vitesse angulaire de reference vaut 1.26 rad/s en mediane). MESURE FAITE, ET
        # ELLE DIT NON. A l'erreur actuelle (3.42 rad/s) le noyau large n'est PAS plat, il est en
        # pleine pente -- il vaut 0.305, et faire descendre l'erreur a 2.0 rad/s rapporte +0.362.
        # Un multi-echelle (3.14, 1.5) ferait tomber ce meme gain a +0.263 : la moyenne des noyaux
        # ecrase la pente du large la ou il est justement bon, parce que le noyau fin vaut 0.005 a
        # 3.42 rad/s et ne se reveille pas avant ~2 rad/s. Le multi-echelle sert a REPARER un noyau
        # plat (cf. object_flat_contact_quality_exp, dont le sigma=0.03 valait 0.0008 a 8 cm) ; ici
        # il n'y a rien a reparer, et l'appliquer serait une regression.
        #
        # Le vrai diagnostic : le gradient existe, il ne PAIE simplement pas assez. Les deux termes
        # de vitesse pesaient ensemble 0.95 sur 6.15 de positifs, soit 15 %, contre 39 % pour les
        # quatre termes de position/orientation du corps et de la caisse. La policy a rationnellement
        # achete de la position avec de la vitesse. On corrige le prix, pas la forme du noyau.
        #
        # ang_vel x2 (c'est l'erreur la plus criante : 2.6x l'amplitude de reference) et lin_vel x1.5
        # (1.2x seulement, donc moins urgent) portent la vitesse a 23 % du budget positif.
        # SI l'erreur descend un jour sous ~1.5 rad/s, le noyau large deviendra plat a son tour et il
        # faudra alors ajouter une echelle fine : motion_global_body_ang_vel accepte desormais
        # sigmas/sigma_weights pour ca. Ce n'est pas le regime actuel.
        "motion_global_body_ang_vel": replace(
            g1_29dof_wbt_reward_w_object_dyn_approach.terms["motion_global_body_ang_vel"],
            weight=2.0,
        ),
        "motion_global_body_lin_vel": replace(
            g1_29dof_wbt_reward_w_object_dyn_approach.terms["motion_global_body_lin_vel"],
            weight=1.5,
        ),
        # --- (2) le glissement des pieds, la mesure la plus directe de l'instabilite --------------
        # Le docstring d'origine dimensionnait ce terme sur le glissement de la REFERENCE (0.020 m/s
        # en appui, cout 2e-4/pas, "une barriere, pas une taxe"). La policy mesuree est a
        # raw = 0.261 = somme sur les pieds en appui de ||v_xy||^2, soit ~0.4 m/s RMS par pied
        # plante avec 1.69 pied en appui en moyenne : VINGT FOIS la reference. La barriere n'en
        # etait pas une, elle pesait 8 % d'un budget negatif deja minuscule (2 % du total).
        #
        # Le terme FONCTIONNE (0.466 -> 0.261 sur le run, il apprend) -- il est juste sous-pondere.
        # -2.0 le porte a ~-0.52/pas, 8.5 % des positifs, au niveau ou il peut reellement arbitrer
        # contre le tracking. Et c'est le terme qui vise pile le symptome observe en sim2sim : le
        # robot qui part en avant sur un pied d'appui qui patine.
        #
        # INTERACTION CONNUE avec les bourrades : sous push le robot DOIT deplacer ses pieds pour se
        # rattraper, et paie alors une penalite qu'il ne peut pas eviter. La DR n'est PAS baissee
        # pour autant (voir l'entete de l'experience _stab : l'hypothese "les pushes plafonnent le
        # tracking de vitesse" ne resiste pas au calcul, et sacrifier de la robustesse sim2real sur
        # une correlation serait un mauvais echange). C'est donc un cout assume, borne : les phases
        # de recuperation representent de l'ordre de 15 % des pas.
        # SI raw_rew_feet_slip_on_ref_stance refuse de descendre sous ~0.15 malgre le poids x4, cette
        # interaction redevient l'explication candidate n°1 et c'est LA qu'il faudra A/B le preset
        # g1_29dof_wbt_randomization_w_object_soft_push -- pas avant.
        "feet_slip_on_ref_stance": replace(
            g1_29dof_wbt_reward_w_object_dyn_approach.terms["feet_slip_on_ref_stance"],
            weight=-2.0,
        ),
        # --- (3) l'enveloppe de couple, qui devait remplacer le proxy d'action-rate ---------------
        # C'etait deja l'intention du terme ("remplacer un proxy d'action-rate par la vraie grandeur
        # physique"), mais a margin=1.5 et poids -0.05 il rendait -0.031/pas, 1.9 % des negatifs :
        # il ne mordait pas. Maintenant que l'on sait que action_rate_l2 ne regularise rien du tout,
        # c'est ce terme-ci qui doit porter la regularisation -- il mesure le couple REELLEMENT
        # commande contre celui que le mouvement demande, ce qui est exactement "la policy se bat
        # contre elle-meme / tremble", et il survit au bruit d'exploration puisqu'il lit l'etat
        # physique et non l'action.
        #
        # margin 1.5 -> 1.25 et poids -0.05 -> -0.15, VOLONTAIREMENT modere. La reponse de la valeur
        # brute au resserrement de la marge n'est pas mesuree (elle depend du couple commande en
        # sim, que le run ne loggue pas), donc l'estimation -0.24/pas est une extrapolation.
        # A SURVEILLER dans les 200 premieres iterations : RawEpisode/raw_rew_torque_envelope_penalty
        # partait de 0.629. S'il depasse ~3.0, la contribution passe -0.45 et il faut redescendre le
        # poids a -0.10 ; s'il reste sous ~1.0, on peut serrer la marge a 1.15.
        "torque_envelope_penalty": replace(
            g1_29dof_wbt_reward_w_object_dyn_approach.terms["torque_envelope_penalty"],
            params={"margin": 1.25},
            weight=-0.15,
        ),
        # --- (4) elagage : deux termes qui occupent du budget sans discriminer --------------------
        # feet_contact_schedule vaut 0.8405 DES L'ITERATION 0 et 0.8798 a 20k : il est sature au
        # depart. Le clip est en double appui sur 69 % des frames (mesure sur dyn_foot_contact_lr),
        # donc le terme est vrai gratuitement et se comporte comme un bonus de survie deguise, pour
        # 7 % du budget positif. Son docstring redoutait exactement ca. Poids 0.5 -> 0.2 : on garde
        # l'information residuelle sur les transitions d'appui, on rend 0.26 de budget.
        "feet_contact_schedule": replace(
            g1_29dof_wbt_reward_w_object_dyn_approach.terms["feet_contact_schedule"],
            weight=0.2,
        ),
        # feet_load_share REGRESSE : 0.144 -> 0.110 sur le run, pendant que tout le reste progresse,
        # pour 0.9 % du budget. Soit le gate `valid` ne s'ouvre quasiment jamais, soit sigma=0.15
        # est trop serre pour la repartition que la policy atteint.
        #
        # ATTENTION -- poids 0.0 == SUPPRESSION PURE ET SIMPLE, pas "desactive mais observe".
        # RewardManager fait `if term_cfg.weight == 0.0: continue` a la construction (manager.py:66),
        # donc le terme n'est jamais instancie, jamais calcule, et RawEpisode/raw_rew_feet_load_share
        # N'EXISTE PAS dans les logs -- verifie sur le run 0ftoz226, le tag est absent. L'intention
        # initiale ("le garder a 0 pour continuer a voir sa courbe") ne marche pas avec cette
        # implementation. Pour observer un terme sans qu'il pese, il faut un poids petit mais NON
        # NUL (c'est ce que fait action_rate_l2 a -0.01 dans _stab2, et son tag est bien loggue).
        "feet_load_share": replace(
            g1_29dof_wbt_reward_w_object_dyn_approach.terms["feet_load_share"],
            weight=0.0,
        ),
        # NON MODIFIE, et c'est un choix : action_rate_l2 reste a -0.1. Voir l'entete de ce preset --
        # ce terme mesure le bruit d'exploration, pas la regularite de la policy deployee.
    }
)

# ==================================================================================================
# STAB-2 -- reallocation du budget negatif, calibree sur ce que _stab a REELLEMENT produit.
#
# CE QUE _stab A MONTRE (run 0ftoz226, reprise a 20000, mesure a 34386 -- 14 400 iterations, 15 h).
# Deux termes ont recu le meme traitement (poids x3-4) et un seul a repondu :
#
#   terme                poids        val/pas 20k -> 34k   contribution/pas   resultat
#   feet_slip           -0.5 -> -2.0     0.260 -> 0.149         -0.298          -43 %   OK
#   torque_envelope    -0.05 -> -0.15    0.626 -> 0.646         -0.097           +3 %   RIEN
#
# La difference n'est pas la nature du terme, c'est la CONTRIBUTION ABSOLUE. A -0.298/pas sur un
# budget positif de 6.4 (4.7 %), la policy negocie. A -0.097 (1.5 %), elle ignore. C'est la seule
# lecon exploitable du run, et elle est directement actionnable.
#
# LE JITTER BAISSE, MAIS TROP LENTEMENT POUR QUE CA COMPTE. Rollout deterministe (eval_dump_qpos,
# la seule mesure non contaminee par le bruit d'exploration) :
#
#                          reference    ck 20000    ck 32000     delta
#   |q.| rms  (rad/s)         0.98        1.59        1.44       -9.0 %
#   |q..| rms (rad/s^2)      18.8        68.4        59.1       -13.6 %
#   energie de q. > 8 Hz      4.1 %      26.4 %      23.1 %     -12.4 %
#
# -13.6 % par 12 000 iterations : atteindre le niveau de la reference demanderait ~94 000 iterations
# de plus (~4 jours). Le rapport p99/rms de q.. vaut 3.9 pour la policy contre 4.2 pour la reference
# -- meme FORME de distribution, amplitude 3x. Ce ne sont donc pas des pics d'impact de contact
# (qui donneraient une distribution plus pointue) mais un exces de mouvement a large bande, que le
# couple commande peut effectivement attraper -- a condition de peser.
#
# D'OU VIENT LE BUDGET. action_rate_l2 coute -1.46/pas, soit 58 % de TOUT le budget negatif, pour un
# effet nul : sa valeur brute est passee de 14.10 a 14.57 pendant que le poids restait le meme, et
# les deux mesures de l'entete de _stab ont etabli qu'elle est integralement du bruit d'exploration
# (2 x 29 x sigma^2 = 14.8 predit, 14.1 mesure). On dilapidait la majorite du budget de penalite sur
# un terme mort et 4 % sur celui qui doit faire le travail.
#
# BUDGET PROJETE : positifs ~6.4, negatifs ~-1.0, net ~+5.4 (contre +3.95 sur _stab). L'echelle
# REMONTE de 37 %, donc nouveau transitoire sur le critique a la reprise -- meme classe de choc que
# 20000 -> _stab, qui s'etait resorbe en quelques centaines d'iterations.
#
# LE PARI EST EXPLICITE ET FALSIFIABLE : feet_slip a prouve qu'un terme a ~5 % du budget positif
# deplace sa grandeur de 40 %. On donne ce meme niveau a celui qui vise le jitter. Si q.. ne bouge
# toujours pas a -0.5/pas, le jitter n'est pas reductible par le reward et il faut aller voir les
# gains PD (kp genou 99, epaule 14.25 -- wn de 1.7 a 3.8 Hz) ou la frequence de commande.
g1_29dof_wbt_reward_w_object_dyn_stab2 = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward_w_object_dyn_stab.terms,
        # action_rate_l2 RESTE A -0.1. Tentative faite, MESUREE, et annulee -- voici pourquoi, parce
        # que le raisonnement qui menait a le baisser est seduisant et faux.
        #
        # L'analyse etait : ce terme coute -1.46/pas (58 % du budget negatif) et sa valeur brute est
        # integralement du bruit d'exploration (2 x 29 x sigma^2 = 14.8 predit, 14.1 mesure), donc il
        # ne regularise PAS la policy moyenne et on peut recuperer son budget. Les deux premisses
        # sont vraies. La conclusion ne l'est pas.
        #
        # Ce que le terme fait AUSSI, et que personne n'avait attribue : il est le SEUL regulateur de
        # sigma du setup. entropy_coef=0.005 pousse sigma vers le haut, aucun plafond n'existe
        # (min_noise_std=None, et il n'y a pas de max), et le cout -w x 2 x 29 x sigma^2 est ce qui
        # equilibrait. Passe a -0.01, mesure sur 660 iterations depuis le checkpoint 32000 :
        #
        #   iteration    sigma    ep_len    error_joint_vel    reward
        #   32060        0.553      814          12.62          90.7
        #   32660        0.848      687          17.10          59.2
        #
        # sigma +54 %, longueur d'episode -16 %, reward -35 %, et error_joint_vel -- la grandeur
        # meme qu'on cherche a reduire -- +35 %. Monotone sur les quatre. Le bruit d'action injecte
        # du jitter physique reel, donc baisser cette taxe DEGRADE directement l'objectif.
        #
        # A l'equilibre sigma ~ 1/sqrt(poids) tant que ce terme domine : diviser le poids par 10
        # multiplie sigma par ~3.2 (0.50 -> 1.6), ce qui colle a la trajectoire observee. Il n'y a
        # donc pas de "poids intermediaire" interessant : -0.03 donnerait deja sigma ~0.9.
        #
        # LEÇON GENERALE : un terme peut ne rien regulariser au niveau de la POLICY et rester
        # indispensable au niveau de la DISTRIBUTION. "Il ne penalise que le bruit" ne veut pas dire
        # "il est gratuit a enlever". Le budget de torque_envelope ci-dessous ne vient donc PAS d'ici
        # -- on assume simplement un budget negatif total plus eleve.
        # NON MODIFIE (herite de _stab, valeur d'origine).
        # Le terme qui doit desormais porter la regularisation, amene au niveau ou feet_slip a
        # fonctionne. Deux leviers ensemble parce qu'ils agissent sur des choses differentes :
        #   - poids -0.15 -> -0.5 : multiplie la contribution par 3.3 a valeur brute constante ;
        #   - marge 1.25 -> 1.10 : fait MONTER la valeur brute en abaissant le seuil du relu.
        # Le passage 1.5 -> 1.25 n'avait pas fait monter la valeur brute (0.626 -> 0.646, dans le
        # bruit), ce qui veut dire que l'essentiel du couple commande est encore SOUS 1.25 x tau_ref.
        # A 1.10 on entre dans la masse de la distribution. Cible : ~-0.5/pas, soit 8 % des positifs.
        # A SURVEILLER a ~200 iterations, et cette fois avec la BONNE conversion (celle que j'avais
        # donnee de travers) : val/pas = raw x 0.02 / (mean_episode_length / 1000). C'est cette
        # valeur-la qui se compare a 0.646, pas la somme d'episode brute qui varie avec la duree.
        # Si val/pas x 0.5 depasse -0.9 (14 % des positifs), redescendre le poids a -0.35.
        "torque_envelope_penalty": replace(
            g1_29dof_wbt_reward_w_object_dyn_stab.terms["torque_envelope_penalty"],
            params={"margin": 1.10},
            weight=-0.5,
        ),
        # NON MODIFIE : feet_slip reste a -2.0. Il a fait son travail (-43 %) et sa valeur continue de
        # descendre (0.182 -> 0.163 -> 0.158 -> 0.149 sur les quatre fenetres du run). On ne touche
        # pas a un terme qui converge.
    }
)

__all__ = [
    "g1_29dof_wbt_fast_sac_reward",
    "g1_29dof_wbt_reward",
    "g1_29dof_wbt_reward_w_object",
    "g1_29dof_wbt_reward_w_object_dyn",
    "g1_29dof_wbt_reward_w_object_dyn_approach",
    "g1_29dof_wbt_reward_w_object_dyn_stab",
    "g1_29dof_wbt_reward_w_object_dyn_stab2",
]


# =====================================================================================
# DynaRetarget (arXiv:2602.06827) Table IV, verbatim.
#
# This is NOT the WBT reward set with extra terms bolted on: it is the paper's table as
# published, so an A/B against the femto14 baseline (wandb vjyjtipa) attributes any
# difference to the reward design rather than to a mixture of both.
#
# SIGMA CONVERSION. The paper writes exp(-k*||.||^2); every kernel in this repo is
# exp(-||.||^2/sigma^2). They agree at sigma = 1/sqrt(k), which is what the comments
# below record per term. Pasting the paper's k straight in as a sigma would invert and
# square every width -- e.g. Body Ang Vel k=0.05 is a VERY WIDE kernel (sigma=4.47), and
# as a sigma it would be the narrowest term in the table.
#
# Dropped on purpose vs the vjyjtipa baseline: motion_relative_hand_object_position_error_exp
# (w=0.3). The table's only hand<->object signal is Contact Match; keeping an extra dense
# grasp term would be a third reward design, neither ours nor theirs.
# =====================================================================================

g1_29dof_wbt_reward_dynaretarget = RewardManagerCfg(
    terms={
        # --- Motion tracking -------------------------------------------------------
        # Root Position, k=5 -> sigma=0.4472
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.4472136},
            weight=0.5,
        ),
        # Root Orientation, k=3 -> sigma=0.5774
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.5773503},
            weight=0.5,
        ),
        # Body Position, k=5 -> sigma=0.4472
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.4472136},
            weight=1.0,
        ),
        # Body Orientation, k=3 -> sigma=0.5774
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.5773503},
            weight=1.0,
        ),
        # Body Linear Velocity, k=0.5 -> sigma=1.4142
        "motion_global_body_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={"sigma": 1.4142136},
            weight=1.0,
        ),
        # Body Angular Velocity, k=0.05 -> sigma=4.4721
        "motion_global_body_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={"sigma": 4.4721360},
            weight=1.0,
        ),
        # Joint Pos Tracking, k=5 -> sigma=0.4472. Highest motion weight in the table.
        "motion_joint_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_joint_position_error_exp",
            params={"sigma": 0.4472136},
            weight=2.0,
        ),
        # --- Object tracking -------------------------------------------------------
        # Contact Match: I(c = c_ref) * exp(-0.1*(||F|| - 10)), penalising only above 10 N.
        "object_contact_match": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:ContactMatch",
            params={"force_target": 10.0, "force_decay": 0.1, "contact_threshold": 1.0},
            weight=1.25,
        ),
        # Object Position, k=8 -> sigma=0.3536 (the tightest kernel in the table)
        "object_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_position_error_exp",
            params={"sigma": 0.3535534},
            weight=1.0,
        ),
        # Object Orientation, k=5 -> sigma=0.4472
        "object_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_orientation_error_exp",
            params={"sigma": 0.4472136},
            weight=1.0,
        ),
        # Object Linear Velocity, k=2 -> sigma=0.7071
        "object_global_ref_lin_vel_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_lin_vel_error_exp",
            params={"sigma": 0.7071068},
            weight=1.0,
        ),
        # Object Angular Velocity, k=0.2 -> sigma=2.2361
        "object_global_ref_ang_vel_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_ang_vel_error_exp",
            params={"sigma": 2.2360680},
            weight=1.0,
        ),
        # --- Regularization --------------------------------------------------------
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-0.1,
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9},
            weight=-10.0,
        ),
        # Table IV's "Self-collisions" counts colliding pairs at -1.0. The nearest thing this
        # codebase has is UndesiredContacts, which thresholds contact FORCE on every body that
        # is not a foot or a hand -- a superset of self-collision (it also fires on a knee
        # touching the box). The vjyjtipa baseline ran it at -0.1; the table's -1.0 is carried
        # over here, and the regex is the baseline's verbatim.
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": (
                    "^(?!left_foot_contact_point$)(?!right_foot_contact_point$)"
                    "(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                    "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"
                ),
            },
            weight=-1.0,
        ),
    },
)

# g1_29dof_wbt_reward_dynaretarget + HDMI's measured-force contact term (object_contact_force_match_exp,
# arXiv:2509.16757 Table I), ported from feat/object-interaction-rewards (commit 309c2a3^, since dropped
# on that branch). Not part of the DynaRetarget paper's own Table IV -- added on top because the paper
# table's only contact signal is object_contact_match, a binary gate on a MATCHED boolean contact state
# that pays exactly 0 everywhere the policy has not yet found real contact, and motion_relative_body_
# position_error_exp (the only other pull toward the hand's target) measured at ~0.09 reward per halving
# of a wrist's error against ~0.67 once in contact -- too weak a gradient across the ~0.6-1.0m starting
# distance observed here (dyn_retarget_dr_sbto run, object_hand_dist stuck/regressing over 4500 iters).
# This term is DENSE (exp(-distance/sigma_pos), no flat zone) and additionally rewards real measured
# force once contact is found, so it can tell a grip that bears load from a hand merely resting nearby.
g1_29dof_wbt_reward_dynaretarget_hdmi = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward_dynaretarget.terms,
        "object_contact_force_match_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_contact_force_match_exp",
            params={
                "sigma_pos": 0.1,
                "sigma_force": 20.0,
                "force_threshold": 10.0,
                "max_force_bonus": 2.0,
            },
            weight=1.0,
        ),
    },
)
