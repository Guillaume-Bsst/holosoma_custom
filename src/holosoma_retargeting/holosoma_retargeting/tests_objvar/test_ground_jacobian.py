"""Jacobienne des sommets SOL en repere objet, validee par differences finies.

Le sol est FIXE dans le monde : en repere objet ses coordonnees bougent quand la pose de
l'objet bouge, donc ses lignes de J_V ne peuvent pas rester nulles quand object_variable
est actif (contrairement aux lignes de l'objet, constantes dans son propre repere).

Meme gabarit que test_relative_jacobian.py : meme fixture, eps=1e-6 centre, atol=5e-5.
La verite terrain est plus simple ici -- R_o^T (g - t_o) se calcule directement, sans FK.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def _obj_frame_pos_of_world_point(q, g_world):
    """Verite terrain : position d'un point MONDE fixe, exprimee dans le repere objet."""
    quat = q[-4:]
    R_o = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    return R_o.T @ (g_world - q[-7:-4])


def test_ground_jacobian_finite_differences():
    from tests_objvar.largebox_fixture import build_largebox_inputs
    r, mk = build_largebox_inputs(2, {"object_variable": True})
    q = np.load("tests_objvar/data/baseline_largebox_15f.npz")["qpos"][5].copy()
    # pose objet non triviale (rotation + translation) pour exciter tous les termes
    q[-7:-4] += np.array([0.11, -0.07, 0.05])
    q[-4:] = Rotation.from_euler("xyz", [0.3, -0.2, 0.4]).as_quat()[[3, 0, 1, 2]]

    import mujoco
    r.robot_data.qpos[:] = q
    mujoco.mj_forward(r.robot_model, r.robot_data)

    # un point de sol quelconque, hors axes pour exciter tous les termes de rotation
    g_world = np.array([0.8, -1.3, 0.0])

    J = r._calc_ground_vertex_jacobian(g_world)          # (3, n_opt)
    assert J.shape == (3, r.n_opt), J.shape

    eps = 1e-6
    for col, qi in enumerate(r.q_opt_indices):
        qp, qm = q.copy(), q.copy()
        qp[qi] += eps
        qm[qi] -= eps
        for qq in (qp, qm):               # renorm des quats perturbes (base + objet)
            qq[3:7] /= np.linalg.norm(qq[3:7])
            qq[-4:] /= np.linalg.norm(qq[-4:])
        fd = (_obj_frame_pos_of_world_point(qp, g_world)
              - _obj_frame_pos_of_world_point(qm, g_world)) / (2 * eps)
        np.testing.assert_allclose(J[:, col], fd, atol=5e-5,
                                   err_msg=f"colonne {col} (qpos {qi})")


def test_ground_jacobian_is_insensitive_to_robot_joints():
    """Le sol ne bouge pas avec le robot : les colonnes des dof ROBOT doivent etre nulles.

    C'est ce qui distingue le cas sol du cas manipulateur, et c'est aussi le garde-fou
    contre une erreur de signe partielle : une jacobienne qui melangerait les deux blocs
    aurait des colonnes robot non nulles.
    """
    from tests_objvar.largebox_fixture import build_largebox_inputs
    r, mk = build_largebox_inputs(2, {"object_variable": True})
    q = np.load("tests_objvar/data/baseline_largebox_15f.npz")["qpos"][5].copy()
    import mujoco
    r.robot_data.qpos[:] = q
    mujoco.mj_forward(r.robot_model, r.robot_data)

    J = r._calc_ground_vertex_jacobian(np.array([0.8, -1.3, 0.0]))
    n_obj_cols = 7                                        # pose objet = 7 coords qpos
    np.testing.assert_allclose(J[:, :-n_obj_cols], 0.0, atol=1e-12)
    assert np.abs(J[:, -n_obj_cols:]).max() > 1e-6, "les colonnes objet ne doivent PAS etre nulles"
