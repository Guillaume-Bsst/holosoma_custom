"""The "Both" mode grid must use the multi-entity (climbing) values, not robot_only's.

robot_only's grid assumes the ground is the sole entity and the motion is rebased to the
origin. Our object_interaction scenes travel well outside (-1, 1)^2 -- smallbox047 spends
68 % of its frames outside it -- so a grid built from those values would measure "a ground
patch present a third of the time", not "ground in the graph".
"""
import sys
from pathlib import Path

import numpy as np
import pytest

FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))

from examples.robot_retarget import create_ground_points          # noqa: E402
from config_types.task import TaskConfig                          # noqa: E402


def test_climbing_values_are_the_multi_entity_ones():
    """Les valeurs climbing sont bien plus larges et plus grossières que robot_only."""
    cfg = TaskConfig()
    assert cfg.climbing_ground_range == (-2.0, 2.0)
    assert cfg.climbing_ground_size == 8
    assert cfg.ground_range == (-1.0, 1.0)          # robot_only, inchangé
    assert cfg.ground_size == 15


def test_grid_extent_and_count():
    cfg = TaskConfig()
    pts = create_ground_points(cfg.climbing_ground_range, cfg.climbing_ground_range,
                               cfg.climbing_ground_size)
    assert pts.shape == (64, 3), pts.shape
    assert pts[:, 0].min() == pytest.approx(-2.0)
    assert pts[:, 0].max() == pytest.approx(2.0)
    assert pts[:, 1].min() == pytest.approx(-2.0)
    assert pts[:, 1].max() == pytest.approx(2.0)
    # subordonnée à l'objet (100 points échantillonnés), comme en climbing
    assert pts.shape[0] < 100


def test_covers_the_manipulation_scenes():
    """Emprise xy des trois scènes de manipulation, mesurée sur les pivots committés.

    Valeurs relevées le 2026-08-03 (racine robot, pivot holov2) -- si une scène change,
    ce test doit tomber avant que la campagne ne produise des chiffres trompeurs.
    """
    extents = {                       # (|x|max, |y|max)
        "smallbox047": (0.65, 1.58),
        "plasticbox043": (1.04, 0.94),
        "largebox003": (0.93, 1.24),
    }
    cfg = TaskConfig()
    half = cfg.climbing_ground_range[1]
    for scene, (ax, ay) in extents.items():
        assert max(ax, ay) < half, f"{scene}: la grille ne couvre pas le mouvement"


def test_both_mode_uses_the_climbing_grid():
    """LE test du changement : il échoue tant que le site d'appel prend robot_only.

    Les trois tests ci-dessus passent avant comme après la bascule (ils appellent
    create_ground_points directement). Celui-ci passe par le vrai chemin du mode Both.
    """
    from examples.robot_retarget import both_mode_ground_points
    cfg = TaskConfig(with_ground=True)
    pts = both_mode_ground_points(cfg)
    assert pts is not None
    assert pts.shape == (64, 3), f"{pts.shape} -- 225 signifie la grille robot_only"
    assert pts[:, 0].max() == pytest.approx(2.0)


def test_both_mode_off_returns_none():
    cfg = TaskConfig(with_ground=False)
    from examples.robot_retarget import both_mode_ground_points
    assert both_mode_ground_points(cfg) is None
