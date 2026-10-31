"""MotionCommand.box_half_extents_view: the box half-extents read by the contact rewards, through one method so that
a subclass can give one box per env (OTRNet). By default, the very tensor the rewards built from
grasp_settle_cfg.box_half_extents at 6fc32ec.

    cd src/holosoma && $PYS -m pytest tests/managers/command/test_wbt_box_half_extents.py -q
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import torch

import holosoma.managers.reward.terms.wbt as rewards
from holosoma.managers.command.terms.wbt import MotionCommand


def test_by_default_the_tensor_of_the_config():
    cmd = object.__new__(MotionCommand)
    cmd.grasp_settle_cfg = SimpleNamespace(box_half_extents=(0.18, 0.18, 0.18))
    for ndim, dtype in ((2, torch.float32), (3, torch.float64)):
        h = cmd.box_half_extents_view(ndim, "cpu", dtype)
        assert torch.equal(h, torch.tensor((0.18, 0.18, 0.18), device="cpu", dtype=dtype)) and h.dtype == dtype


def test_the_contact_rewards_read_the_box_through_it():
    tree = ast.parse(inspect.getsource(rewards))
    direct = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "box_half_extents"]
    views = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "box_half_extents_view"]
    assert direct == [] and len(views) == 3
