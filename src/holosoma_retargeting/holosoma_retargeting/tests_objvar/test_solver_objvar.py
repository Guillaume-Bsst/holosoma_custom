import numpy as np


def _run(n, overrides, tmp_path, name):
    from tests_objvar.largebox_fixture import build_largebox_inputs
    r, mk = build_largebox_inputs(n, overrides)
    out = str(tmp_path / f"{name}.npz")
    r.retarget_motion(dest_res_path=out, **mk)
    return np.load(out)["qpos"], mk


def test_flag_off_bit_identical(tmp_path):
    qpos, _ = _run(15, {}, tmp_path, "off")
    base = np.load("tests_objvar/data/baseline_largebox_15f.npz")["qpos"]
    assert np.array_equal(qpos, base), np.abs(qpos - base).max()


def test_flag_off_with_ground_bit_identical(tmp_path):
    """Meme garde-fou que ci-dessus, mais sol ON -- la configuration des six pivots DEJA
    PUBLIES de la campagne sol-seul (2026-08-03), que `test_flag_off_bit_identical` ne
    couvre pas : sa fixture passe `ground_points_world=None`, donc elle ne traverse ni les
    deux blocs sol de `solve_single_iteration` ni `_calc_ground_vertex_jacobian`.

    Ces blocs sont gardes par `self.object_variable`, ce qui doit rendre le chemin flag-off
    bit-identique a la revision d'AVANT la tache 1 meme quand la grille de sol est presente.
    `data/baseline_largebox_ground_15f.npz` a ete produit sur cette revision de base
    (`8f42d33`, ou le `NotImplementedError` objvar+sol etait encore en place), par ce meme
    chemin, et verifie deterministe (deux runs bit-identiques) avant d'etre committe.

    Non-vacuite : la seconde assertion exige que la grille change effectivement la
    trajectoire. Sans elle, une regression qui IGNORERAIT `ground_points_world` passerait
    silencieusement (le baseline l'ignorerait tout autant), et le test ne couvrirait rien.
    """
    from tests_objvar.largebox_fixture import build_largebox_inputs
    from holosoma_retargeting.config_types.task import TaskConfig
    from holosoma_retargeting.examples.robot_retarget import create_ground_points

    r, mk = build_largebox_inputs(15, {})
    assert not r.object_variable, "ce test porte sur le chemin FLAG OFF"
    cfg = TaskConfig()
    mk["ground_points_world"] = create_ground_points(
        cfg.climbing_ground_range, cfg.climbing_ground_range, cfg.climbing_ground_size)
    out = str(tmp_path / "off_ground.npz")
    r.retarget_motion(dest_res_path=out, **mk)
    qpos = np.load(out)["qpos"]

    base = np.load("tests_objvar/data/baseline_largebox_ground_15f.npz")["qpos"]
    assert np.array_equal(qpos, base), np.abs(qpos - base).max()

    no_ground = np.load("tests_objvar/data/baseline_largebox_15f.npz")["qpos"]
    assert np.abs(qpos - no_ground).max() > 1e-6, \
        "la grille de sol n'a eu aucun effet -- ce test ne couvre alors plus le chemin sol"


def test_strong_anchor_recovers_fixed_object(tmp_path):
    qpos, mk = _run(15, {"object_variable": True, "w_object_tracking": 1e6}, tmp_path, "strong")
    ref = mk["object_poses_augmented"][:15]        # ordre mujoco [pos(3), quat wxyz(4)]
    dpos = np.abs(qpos[:, -7:-4] - ref[:, :3]).max()
    # Seuil 2.5 mm (pas 1 mm) : la reference largebox_003 frotte le sol/le genou a
    # moins de penetration_tolerance (1e-3) pres -- masks_ok laisse desormais entrer
    # objet<->sol (et objet<->corps) dans le QP quand object_variable est actif, donc
    # ces contraintes DURES bornent l'ancre (verifie : phi objet-sol/objet-genou pin
    # exactement a -penetration_tolerance, et le residu est invariant a w_object_tracking
    # 1e6->1e9 -- ce n'est pas un sous-poids de l'ancre, c'est une butee de collision).
    assert dpos < 2.5e-3, dpos
    for t in range(15):                            # angle quat < 0.2 deg
        qa = qpos[t, -4:] / np.linalg.norm(qpos[t, -4:])
        qb = ref[t, 3:7] / np.linalg.norm(ref[t, 3:7])
        ang = 2 * np.degrees(np.arccos(np.clip(abs(qa @ qb), -1, 1)))
        assert ang < 0.2, (t, ang)


def test_free_object_actually_moves(tmp_path):
    qpos, mk = _run(15, {"object_variable": True, "w_object_tracking": 0.0}, tmp_path, "free")
    ref = mk["object_poses_augmented"][:15]
    dpos = np.abs(qpos[:, -7:-4] - ref[:, :3]).max()
    assert dpos > 1e-4, "l'objet n'a pas bouge -- la variable n'est pas cablee"


def test_anchor_quat_hemisphere_insensitive(tmp_path):
    # Sur les 15 premieres frames de largebox_003 (fixture par defaut des autres
    # tests), l'objet ne tourne quasi pas (vitesse angulaire ~0.04 deg/frame) : un
    # flip de signe y pousse l'ancre buguee vers une cible purement RADIALE dans
    # l'espace quat (target = -2*courant), que la renormalisation post-pas absorbe
    # sans le moindre effet rotationnel (verifie : dot(ref,cur) reste exactement a
    # -1.0000 iteration apres iteration, cout inchange -> arret par convergence des
    # la 2e iteration). Le bug ne bascule visiblement l'objet que la ou une VRAIE
    # rotation frame-a-frame existe deja (la composante non-radiale du residu buggue
    # y devient significative) : dans largebox_003 c'est a partir de ~frame 25 (jusqu'a
    # ~2 deg/frame). On flip donc a partir de la frame 25 sur une fenetre de 50 frames --
    # sans le fix (verifie manuellement) l'objet fait une bascule transitoire de pres de
    # 160 deg avant de se re-verrouiller (meme rotation physique, hemisphere oppose) ;
    # avec le fix l'angle pic reste sous 0.1 deg.
    from tests_objvar.largebox_fixture import build_largebox_inputs
    r, mk = build_largebox_inputs(50, {"object_variable": True, "w_object_tracking": 1e6})
    ref = mk["object_poses_augmented"]
    flipped = ref.copy()
    flipped[25:, 3:7] *= -1.0            # meme rotation, representation antipodale
    mk["object_poses_augmented"] = flipped
    out = str(tmp_path / "hemi.npz")
    r.retarget_motion(dest_res_path=out, **mk)
    qpos = np.load(out)["qpos"]
    for t in range(50):                  # l'objet suit la ROTATION de la reference
        qa = qpos[t, -4:] / np.linalg.norm(qpos[t, -4:])
        qb = ref[t, 3:7] / np.linalg.norm(ref[t, 3:7])
        ang = 2 * np.degrees(np.arccos(np.clip(abs(qa @ qb), -1, 1)))
        assert ang < 0.5, (t, ang)
