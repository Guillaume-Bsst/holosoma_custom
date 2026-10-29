"""Portage a l'inference du feed-forward de couple de reference (etage 05 / SPIDER).

Pourquoi ce fichier existe
--------------------------
A l'entrainement, ``dyn_tau`` (couple articulaire de reference produit en rejouant le clip dans un
vrai solveur de contact MuJoCo, cf. wbt_rl/scripts/merge_dynamics.py) est injecte directement dans
la loi PD par l'environnement (``JointPositionActionTerm._compute_torque_feedforward``, voir
``holosoma/config_types/action.py:TorqueFeedforwardCfg``) ::

    torques = kp*(q_des - q) - kd*qd + grip_bias + scale * dyn_tau[reference_frame]

Ce couple ne sort JAMAIS du reseau -- la policy apprend a corriger AUTOUR de lui, pas a le produire
-- donc l'ONNX exporte n'en contient rien. Sans ce module, un checkpoint entraine avec
``torque_feedforward.enable=True`` (preset ``g1_29dof_joint_pos_grip_force_dyn``) tourne en
sim2sim/reel amputee de la moitie de son couple de gravite/inertie appris, exactement comme la
caisse tombait avant ``grip_force.py``.

Volontairement plus simple que ``grip_force.py`` : pas de FK, pas de pinocchio, pas de geometrie --
juste une lecture indexee par le temps dans le tableau ``dyn_tau`` du clip, reindexee de l'ordre
DOF du clip vers l'ordre DOF du robot. C'est la meme donnee que celle utilisee cote entrainement par
``MotionCommand.dyn_tau`` (``managers/command/terms/wbt.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger


@dataclass(frozen=True)
class TorqueFeedforwardParams:
    """Miroir de ``holosoma.config_types.action.TorqueFeedforwardCfg``.

    Toute divergence avec le run d'entrainement (scale, exclude_joint_names) se traduit par un
    couple different de celui vu a l'entrainement, donc un ecart sim2real silencieux.
    """

    scale: float = 0.5
    """Fraction du couple de reference injectee. DOIT egaler
    ``action.terms.joint_control.params.torque_feedforward.scale`` du run d'entrainement."""

    joint_names: tuple[str, ...] | None = None
    """Restreint le feed-forward a ces DOF (sous-chaine, meme convention que l'entrainement).
    ``None`` = tous les DOF."""

    exclude_joint_names: tuple[str, ...] = field(default_factory=tuple)
    """DOF a EXCLURE (sous-chaine), applique apres ``joint_names``. DOIT reprendre
    ``exclude_joint_names`` du run d'entrainement : ce sont les articulations dont ``dyn_tau`` est
    en butee de couple (valeur = plafond de l'actionneur cote solve physique, pas la demande reelle
    du mouvement), et que le feed-forward ne doit donc pas rejouer."""

    max_abs_nm: float | None = None
    """Plafond optionnel, applique APRES ``scale``, meme borne pour tous les DOF actifs. A
    l'entrainement le clip final porte sur le couple TOTAL (PD + biais) contre les limites reelles
    de l'actionneur ; a l'inference le PD est calcule dans le driver moteur et reste invisible ici,
    donc on ne peut borner QUE le feed-forward. ``None`` = pas de plafond (le ``scale`` et
    ``exclude_joint_names`` de l'entrainement sont censes suffire, cf. leur docstring)."""


class TorqueFeedforwardBias:
    """Reconstruit ``_compute_torque_feedforward`` a partir du seul clip (pas de proprioception)."""

    def __init__(self, dof_names: tuple[str, ...], params: TorqueFeedforwardParams) -> None:
        self.dof_names = tuple(dof_names)
        self.params = params

        include = params.joint_names
        exclude = params.exclude_joint_names
        keep = [
            (include is None or any(sel in dof for sel in include)) and not any(sel in dof for sel in exclude)
            for dof in self.dof_names
        ]
        if not any(keep):
            raise ValueError(
                f"torque_feedforward joint_names={include} / exclude_joint_names={exclude} "
                f"ne laisse aucun DOF actif parmi {self.dof_names}"
            )
        self._mask = np.asarray(keep, dtype=np.float64)
        dropped = [d for d, k in zip(self.dof_names, keep) if not k]
        if dropped:
            logger.info(f"torque_feedforward: actif sur {int(self._mask.sum())} DOF, exclus {dropped}")

        self._dyn_tau: np.ndarray | None = None  # (T, num_dofs) in dof_names order, post-scale/mask

    def load(self, npz_path: str, prepend: int = 0) -> None:
        """Charge ``dyn_tau`` du clip et le reindexe dans l'ordre ``dof_names`` du robot.

        La fenetre de prepend (pose par defaut) est completee a couple NUL -- pas par repetition de
        la frame 0 -- exactement comme cote entrainement (``managers/command/terms/wbt.py``: "dyn_tau
        0 N.m -> no feed-forward during the transition, the PD loop alone drives").
        """
        data = np.load(npz_path, allow_pickle=True)
        if "dyn_tau" not in data.files:
            raise ValueError(
                f"{npz_path} ne porte pas dyn_tau : ce clip n'a pas ete enrichi par l'etage 05. "
                f"Executer wbt_rl/scripts/merge_dynamics.py dessus d'abord, ou desactiver "
                f"--task.torque-feedforward-enable."
            )
        if "joint_names" not in data.files:
            raise ValueError(f"{npz_path} porte dyn_tau mais pas joint_names : reindexation impossible.")

        clip_joint_names = [str(n) for n in data["joint_names"]]
        missing = [n for n in self.dof_names if n not in clip_joint_names]
        if missing:
            raise ValueError(
                f"{npz_path}: joint_names du clip ne couvre pas {missing} attendus par le robot."
            )
        col_idx = np.array([clip_joint_names.index(n) for n in self.dof_names], dtype=int)

        dyn_tau = np.asarray(data["dyn_tau"], dtype=np.float64)[:, col_idx]  # (T, num_dofs), robot order
        if prepend:
            pad = np.zeros((prepend, dyn_tau.shape[1]), dtype=np.float64)
            dyn_tau = np.concatenate([pad, dyn_tau], axis=0)

        self._dyn_tau = dyn_tau * self.params.scale * self._mask[None, :]
        logger.info(
            f"TorqueFeedforwardBias pret : {dyn_tau.shape[0]} frames depuis {npz_path.split('/')[-1]} "
            f"(prepend={prepend}), scale={self.params.scale}"
        )

    def compute(self, motion_timestep: float) -> np.ndarray:
        """Couple feedforward (num_dofs,) a additionner a ``cmd_tau`` pour cette frame."""
        if self._dyn_tau is None:
            return np.zeros(len(self.dof_names), dtype=np.float64)
        idx = int(np.clip(int(round(motion_timestep)), 0, self._dyn_tau.shape[0] - 1))
        tau = self._dyn_tau[idx]
        if self.params.max_abs_nm is not None:
            tau = np.clip(tau, -self.params.max_abs_nm, self.params.max_abs_nm)
        return tau
