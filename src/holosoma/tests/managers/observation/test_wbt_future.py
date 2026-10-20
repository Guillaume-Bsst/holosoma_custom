"""Future-reference observation terms (OTRL architecture.md, experiment 4), without IsaacSim.

The trap is the clip boundary: in a multi-clip run every clip lives in ONE concatenated timeline, so t + k past
the end of an env's clip would read the next clip. ``future_frame_index`` must clamp to the env's own clip.
The terms are checked on a fake MotionCommand (``object.__new__``, no setup) against the EXISTING terms at
offset 0 (motion_ref_pos_b, motion_ref_ori_b, obj_pos_b, obj_ori_b).

    cd src/holosoma && $PYS -m pytest tests/managers/observation/test_wbt_future.py -q
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.observation.terms import wbt as T


# ------------------------------------------------------------------------------------------------ pure index
def test_offsets_seconds_to_steps_rounds():
    # int() would give 14 and 29: 0.3 / 0.02 = 14.999999999999998
    assert T.future_offsets_steps((0.1, 0.3, 0.6), 0.02) == [5, 15, 30]
    assert T.future_offsets_steps([0.0, 0.1], 0.02) == [0, 5]
    with pytest.raises(ValueError):
        T.future_offsets_steps((-0.1,), 0.02)


def _layout(lengths):
    ends = torch.tensor(lengths).cumsum(0)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), ends[:-1]])
    return starts, ends


def test_future_index_multiclip_hand_cases():
    starts, ends = _layout([10, 4, 7])  # clips [0,10) [10,14) [14,21)
    mids = torch.tensor([0, 0, 1, 1, 2, 2])
    t = torch.tensor([0, 8, 10, 13, 14, 20])
    idx = T.future_frame_index(t, mids, ends, torch.tensor([5, 15, 30]))
    assert idx.tolist() == [
        [5, 9, 9],
        [9, 9, 9],
        [13, 13, 13],  # NOT 15 / 25 / 40: those are clip 2 or past the array
        [13, 13, 13],
        [19, 20, 20],
        [20, 20, 20],
    ]


def test_future_index_offset_zero_is_identity():
    starts, ends = _layout([10, 4, 7])
    mids = torch.tensor([0, 1, 2])
    t = torch.tensor([3, 12, 20])
    assert T.future_frame_index(t, mids, ends, torch.tensor([0]))[:, 0].tolist() == t.tolist()


def test_future_index_single_clip_loader():
    # MotionLoader: motion_end_idx = [T], motion_ids all 0
    ends = torch.tensor([50])
    t = torch.tensor([0, 30, 49])
    idx = T.future_frame_index(t, torch.zeros(3, dtype=torch.long), ends, torch.tensor([5, 30]))
    assert idx.tolist() == [[5, 30], [35, 49], [49, 49]]


@pytest.mark.parametrize("seed", range(5))
def test_future_index_never_leaves_own_clip_random(seed):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(2, 60, (int(torch.randint(1, 12, (1,), generator=g)),), generator=g).tolist()
    starts, ends = _layout(lengths)
    n = 4000
    mids = torch.randint(0, len(lengths), (n,), generator=g)
    t = starts[mids] + (torch.rand(n, generator=g) * (ends[mids] - starts[mids]).float()).long()
    offsets = torch.tensor([0, 1, 5, 15, 30, 100])
    idx = T.future_frame_index(t, mids, ends, offsets)
    clip_of = torch.searchsorted(ends, idx, right=True)  # clip owning each global frame
    assert torch.equal(clip_of, mids[:, None].expand_as(idx))
    assert (idx >= t[:, None]).all() and (idx <= ends[mids, None] - 1).all()
    inside = t[:, None] + offsets[None] <= ends[mids, None] - 1
    assert torch.equal(idx[inside], (t[:, None] + offsets[None]).expand_as(idx)[inside])


# ------------------------------------------------------------------------------------------------ fake env
def _rand_quat(n, g):
    q = torch.randn(*n, 4, generator=g)
    return q / q.norm(dim=-1, keepdim=True)


def _fake_env(lengths, mids, t, seed=0, n_bodies=3, ref_body=1, n_joints=29):
    """A MotionCommand (no setup) over a concatenated timeline; clip c's box sits at x = 100 c (unmissable)."""
    g = torch.Generator().manual_seed(seed)
    starts, ends = _layout(lengths)
    total = int(ends[-1])
    clip_of_frame = torch.searchsorted(ends, torch.arange(total), right=True)
    joint_pos = torch.arange(total, dtype=torch.float32)[:, None].repeat(1, n_joints)  # value = global frame
    body_pos = torch.randn(total, n_bodies, 3, generator=g)
    body_pos[:, ref_body, 0] += 100.0 * clip_of_frame
    obj_pos = torch.randn(total, 3, generator=g)
    obj_pos[:, 0] += 100.0 * clip_of_frame
    motion = SimpleNamespace(
        joint_pos=joint_pos,
        body_pos_w=body_pos,
        body_quat_w=_rand_quat((total, n_bodies), g),
        object_pos_w=obj_pos,
        object_quat_w=_rand_quat((total,), g),
        motion_start_idx=starts,
        motion_end_idx=ends,
        has_object=True,
    )
    n = len(t)
    sim = SimpleNamespace(
        _rigid_body_pos=torch.randn(n, n_bodies, 3, generator=g),
        _rigid_body_rot=_rand_quat((n, n_bodies), g),
        scene=SimpleNamespace(env_origins=torch.randn(n, 3, generator=g) * 5.0),
        all_root_states=torch.zeros(n, 13),
    )
    env = SimpleNamespace(num_envs=n, device="cpu", dt=0.02, simulator=sim)
    mc = object.__new__(MotionCommand)
    mc._env = env
    mc.motion = motion
    mc.time_steps = torch.as_tensor(t, dtype=torch.long)
    mc.motion_ids = torch.as_tensor(mids, dtype=torch.long)
    mc.ref_body_index = ref_body
    mc.object_indices_in_simulator = torch.arange(n)
    env.command_manager = SimpleNamespace(get_state=lambda name: mc)
    return env, mc


LENGTHS = [40, 12, 35]  # ends 40, 52, 87


def test_future_joint_pos_stays_in_clip():
    env, _ = _fake_env(LENGTHS, mids=[0, 1, 2], t=[38, 45, 80])
    out = T.motion_future_joint_pos(env, offsets_s=(0.1, 0.3, 0.6)).view(3, 3, 29)[..., 0]
    assert out.tolist() == [[39, 39, 39], [50, 51, 51], [85, 86, 86]]
    assert out.shape == (3, 3)


def test_ref_obj_future_never_reads_next_clip():
    env, mc = _fake_env(LENGTHS, mids=[0, 1], t=[39, 51])
    # the robot anchor at the env origin, identity: body frame = world frame minus origin
    mc._env.simulator._rigid_body_pos[:, 1] = env.simulator.scene.env_origins
    mc._env.simulator._rigid_body_rot[:, 1] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    p = T.ref_obj_pos_b(env, offsets_s=(0.0, 0.1, 0.3, 0.6)).view(2, 4, 3)
    assert (p[0, :, 0].abs() < 10).all(), "clip 0 env read a clip-1 box (x ~ 100)"
    assert ((p[1, :, 0] - 100).abs() < 10).all(), "clip 1 env read a clip-2 box (x ~ 200)"
    last = mc.motion.object_pos_w[[39, 51]]
    assert torch.allclose(p[:, -1], last, atol=1e-5)


def test_offset_zero_matches_existing_terms():
    env, mc = _fake_env(LENGTHS, mids=[0, 0, 1, 2, 2], t=[0, 20, 47, 52, 86], seed=3)
    zero = (0.0,)
    assert torch.allclose(T.motion_future_ref_pos_b(env, offsets_s=zero), T.motion_ref_pos_b(env), atol=1e-5)
    assert torch.allclose(T.motion_future_ref_ori_b(env, offsets_s=zero), T.motion_ref_ori_b(env), atol=1e-5)
    # simulated box = reference box  ->  ref_obj_* at offset 0 = obj_*
    rs = env.simulator.all_root_states
    rs[:, :3] = mc.object_pos_w
    rs[:, 3:7] = mc.object_quat_w
    assert torch.allclose(T.ref_obj_pos_b(env, offsets_s=zero), T.obj_pos_b(env), atol=1e-5)
    assert torch.allclose(T.ref_obj_ori_b(env, offsets_s=zero), T.obj_ori_b(env), atol=1e-5)


def test_body_frame_rotation():
    # robot anchor yawed +90 deg at the origin: a box 1 m along world +x is at body -y
    env, mc = _fake_env([10], mids=[0], t=[0])
    env.simulator.scene.env_origins.zero_()
    mc._env.simulator._rigid_body_pos[:, 1] = 0.0
    s = math.sin(math.pi / 4)
    mc._env.simulator._rigid_body_rot[:, 1] = torch.tensor([0.0, 0.0, s, s])
    mc.motion.object_pos_w[:] = torch.tensor([1.0, 0.0, 0.0])
    p = T.ref_obj_pos_b(env, offsets_s=(0.0, 0.1)).view(2, 3)
    assert torch.allclose(p, torch.tensor([[0.0, -1.0, 0.0]] * 2), atol=1e-5)


def test_dims():
    env, _ = _fake_env(LENGTHS, mids=[0, 1], t=[0, 41])
    f, r = (0.1, 0.3, 0.6), (0.0, 0.1, 0.3, 0.6)
    dims = {
        "motion_future_joint_pos": T.motion_future_joint_pos(env, offsets_s=f).shape[1],
        "motion_future_ref_pos_b": T.motion_future_ref_pos_b(env, offsets_s=f).shape[1],
        "motion_future_ref_ori_b": T.motion_future_ref_ori_b(env, offsets_s=f).shape[1],
        "ref_obj_pos_b": T.ref_obj_pos_b(env, offsets_s=r).shape[1],
        "ref_obj_ori_b": T.ref_obj_ori_b(env, offsets_s=r).shape[1],
        "motion_ref_pos_b": T.motion_ref_pos_b(env).shape[1],
    }
    assert dims == {
        "motion_future_joint_pos": 87,
        "motion_future_ref_pos_b": 9,
        "motion_future_ref_ori_b": 18,
        "ref_obj_pos_b": 12,
        "ref_obj_ori_b": 24,
        "motion_ref_pos_b": 3,
    }
    assert sum(dims.values()) == 153


def test_no_object_gives_zeros():
    env, mc = _fake_env(LENGTHS, mids=[0], t=[0])
    mc.motion.has_object = False
    assert torch.equal(T.ref_obj_pos_b(env), torch.zeros(1, 12))
    assert torch.equal(T.ref_obj_ori_b(env), torch.zeros(1, 24))


# ------------------------------------------------------------------------------------------------ presets
def test_presets_additive_only():
    from holosoma.config_values import experiment as E
    from holosoma.config_values import observation as OD
    from holosoma.config_values.wbt.g1 import observation as O

    new, old = O.g1_29dof_wbt_observation_w_object_actor_future, O.g1_29dof_wbt_observation_w_object_actor
    added = {"motion_future_joint_pos", "motion_future_ref_pos_b", "motion_future_ref_ori_b", "ref_obj_pos_b",
             "ref_obj_ori_b"}
    a_new, a_old = new.groups["actor_obs"].terms, old.groups["actor_obs"].terms
    assert set(a_new) == set(a_old) | added | {"motion_ref_pos_b"}
    assert all(a_new[k] == a_old[k] for k in a_old)
    c_new, c_old = new.groups["critic_obs"].terms, old.groups["critic_obs"].terms
    assert set(c_new) == set(c_old) | added  # motion_ref_pos_b already in the critic: not duplicated
    assert all(c_new[k] == c_old[k] for k in c_old)
    assert all(c_new[k].noise == 0.0 for k in added)
    # the existing preset is untouched
    assert old.groups["actor_obs"] is O.actor_obs_w_object
    assert not (set(a_old) & (added | {"motion_ref_pos_b"}))
    assert E.DEFAULTS["g1_29dof_wbt_w_object_actor"].observation is old
    fut = E.DEFAULTS["g1_29dof_wbt_w_object_actor_future"]
    assert fut.observation is new
    base = E.DEFAULTS["g1_29dof_wbt_w_object_actor"]
    for f in ("reward", "termination", "command", "robot", "algo", "randomization", "simulator", "action"):
        assert getattr(fut, f) == getattr(base, f), f
    assert OD.DEFAULTS["g1_29dof_wbt_w_object_actor_future"] is new
