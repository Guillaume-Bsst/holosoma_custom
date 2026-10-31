"""box_geometry with one box per query row (OTRNet, one object per env): half_extents of shape (3,) gives the very
numbers of 6fc32ec; half_extents broadcastable to the queries (..., 3) gives, row by row, the numbers of the (3,)
call with that row's box.

    cd src/holosoma && $PYS -m pytest tests/utils/test_box_geometry_per_env.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from holosoma.utils import box_geometry as new

_spec = importlib.util.spec_from_file_location("box_geometry_6fc32ec", Path(__file__).with_name("box_geometry_6fc32ec.py"))
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)

HALF = torch.tensor([0.235, 0.229, 0.204])


def _points(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(*shape, 3, generator=g) - 0.5) * 0.8  # inside and outside the box


def test_a_single_box_gives_the_numbers_of_6fc32ec():
    for shape in ((500,), (64, 5)):
        p = _points(*shape)
        d_new, n_new = new.box_nearest_and_signed_distance(p, HALF)
        d_old, n_old = old.box_nearest_and_signed_distance(p, HALF)
        assert torch.equal(d_new, d_old) and torch.equal(n_new, n_old)
        q = _points(*shape, seed=1)
        _, w1 = old.box_nearest_and_signed_distance(p, HALF)
        _, w2 = old.box_nearest_and_signed_distance(q, HALF)
        assert torch.equal(new.box_surface_geodesic_distance(w1, w2, HALF),
                           old.box_surface_geodesic_distance(w1, w2, HALF))


def test_one_box_per_row_is_the_single_box_call_row_by_row():
    n, k = 40, 5
    g = torch.Generator().manual_seed(2)
    half = 0.15 + 0.1 * torch.rand(n, 3, generator=g)
    p = _points(n, k, seed=3)
    q = _points(n, k, seed=4)
    d, w = new.box_nearest_and_signed_distance(p, half[:, None, :])
    _, w2 = new.box_nearest_and_signed_distance(q, half[:, None, :])
    geo = new.box_surface_geodesic_distance(w, w2, half[:, None, :])
    for i in range(n):
        d_i, w_i = old.box_nearest_and_signed_distance(p[i], half[i])
        _, w2_i = old.box_nearest_and_signed_distance(q[i], half[i])
        assert torch.equal(d[i], d_i) and torch.equal(w[i], w_i)
        assert torch.equal(geo[i], old.box_surface_geodesic_distance(w_i, w2_i, half[i]))
    d2, _ = new.box_nearest_and_signed_distance(p[:, 0], half)  # (N,3) queries, (N,3) boxes
    assert torch.equal(d2, d[:, 0])
