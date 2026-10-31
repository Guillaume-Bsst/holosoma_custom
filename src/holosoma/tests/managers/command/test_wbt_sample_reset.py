"""MotionCommand._sample_reset_steps: the draw of each reset (motion of each env, start time step) moved out of
reset() unchanged, so that a subclass can draw otherwise (OTRNet, a sampler with bins local to each clip). Checked on
a fake MotionCommand (``object.__new__``, no setup) against the inline code of reset() at 6fc32ec, copied below.

    cd src/holosoma && $PYS -m pytest tests/managers/command/test_wbt_sample_reset.py -q
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.managers.command.terms.wbt import AdaptiveTimestepsSampler, MotionCommand

LENGTHS = [120, 300, 45, 210]


def _fake(adaptive: bool, zero_prob: float, evaluating: bool = False, forced=None, num_envs: int = 64):
    ends = torch.tensor(LENGTHS).cumsum(0)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), ends[:-1]])
    cmd = object.__new__(MotionCommand)
    cmd.device = "cpu"
    cmd.num_envs = num_envs
    cmd.motion_cfg = SimpleNamespace(use_adaptive_timesteps_sampler=adaptive, start_at_timestep_zero_prob=zero_prob)
    cmd.motion = SimpleNamespace(motion_start_idx=starts, motion_end_idx=ends, num_motions=len(LENGTHS),
                                 time_step_total=int(ends[-1]))
    if adaptive:
        cmd.adaptive_timesteps_sampler = AdaptiveTimestepsSampler(int(ends[-1]), "cpu", 50)
    g = torch.Generator().manual_seed(7)
    terminated = torch.rand(num_envs, generator=g) < 0.4
    cmd._env = SimpleNamespace(termination_manager=SimpleNamespace(terminated=terminated), is_evaluating=evaluating)
    cmd.motion_ids = torch.randint(0, len(LENGTHS), (num_envs,), generator=g)
    cmd.time_steps = cmd.motion.motion_start_idx[cmd.motion_ids] + 3
    cmd._force_start_timesteps = forced
    return cmd


def _inline_6fc32ec(self, env_ids):
    """reset() of 6fc32ec, from "# 0. Sample the time steps" to the last-timestep clamp, word for word."""
    if self.motion_cfg.use_adaptive_timesteps_sampler:
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            failed_at_time_step = self.time_steps[env_ids][episode_failed]
            self.adaptive_timesteps_sampler.update_current_bin_failed_count(failed_at_time_step)
        phase = self.adaptive_timesteps_sampler.sample(env_ids.numel())
    else:
        phase = torch.rand(env_ids.numel(), device=self.device)
    if self._env.is_evaluating:
        phase = torch.zeros_like(phase)
    n = env_ids.numel()
    num_motions = self.motion.num_motions
    self.motion_ids[env_ids] = torch.randint(0, num_motions, (n,), device=self.device)
    start_idx = self.motion.motion_start_idx[self.motion_ids[env_ids]]
    end_idx = self.motion.motion_end_idx[self.motion_ids[env_ids]]
    motion_len = end_idx - start_idx
    self.time_steps[env_ids] = start_idx + (phase * (motion_len - 1).float()).long()
    prob = self.motion_cfg.start_at_timestep_zero_prob
    if prob >= 1.0:
        self.time_steps[env_ids] = start_idx
    elif prob > 0.0:
        subset = self.time_steps[env_ids]
        rand_vals = torch.rand_like(subset, dtype=torch.float32)
        subset = torch.where(rand_vals < prob, start_idx, subset)
        self.time_steps[env_ids] = subset
    if self._force_start_timesteps is not None:
        forced = self._force_start_timesteps.to(device=self.device, dtype=torch.long)
        end_clamp = self.motion.motion_end_idx[self.motion_ids[env_ids]] - 2
        self.time_steps[env_ids] = torch.minimum(forced[env_ids].clamp(min=0), end_clamp)
    already_last_timestep_mask = self.time_steps[env_ids] >= end_idx - 1
    self.time_steps[env_ids] = torch.where(already_last_timestep_mask, end_idx - 2, self.time_steps[env_ids])


@pytest.mark.parametrize(
    "adaptive,zero_prob,evaluating,forced",
    [(True, 0.2, False, None), (False, 0.0, False, None), (True, 1.0, False, None), (True, 0.2, True, None),
     (True, 0.2, False, torch.arange(64) * 11)],
)
def test_the_extracted_draw_is_the_inline_draw(adaptive, zero_prob, evaluating, forced):
    env_ids = torch.tensor([0, 3, 5, 6, 10, 17, 31, 32, 40, 63])
    a, b = _fake(adaptive, zero_prob, evaluating, forced), _fake(adaptive, zero_prob, evaluating, forced)
    torch.manual_seed(123)
    _inline_6fc32ec(a, env_ids)
    torch.manual_seed(123)
    MotionCommand._sample_reset_steps(b, env_ids)
    assert torch.equal(a.motion_ids, b.motion_ids) and torch.equal(a.time_steps, b.time_steps)
    if adaptive:
        assert torch.equal(a.adaptive_timesteps_sampler.current_bin_failed_count,
                           b.adaptive_timesteps_sampler.current_bin_failed_count)


def test_reset_draws_through_the_method():
    """reset() calls _sample_reset_steps first (a subclass's draw replaces the inline one)."""

    class Drawn(Exception):
        pass

    cmd = _fake(True, 0.2)

    def draw(env_ids):
        raise Drawn(env_ids)

    cmd._sample_reset_steps = draw
    with pytest.raises(Drawn):
        MotionCommand.reset(cmd, torch.tensor([1, 2]))


def test_reset_reads_no_local_of_the_moved_block():
    """Every name reset() reads is assigned in it, an argument, a module global of wbt.py or a builtin: none was left
    to the moved block (``n`` was, until it was assigned again in reset)."""
    import ast
    import builtins
    import inspect

    import holosoma.managers.command.terms.wbt as wbt

    tree = ast.parse(inspect.getsource(wbt))
    cls = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "MotionCommand")
    fn = next(f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "reset")
    stored = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    read = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    args = {a.arg for a in fn.args.args}
    free = read - stored - args - set(vars(wbt)) - set(dir(builtins))
    assert not free, free
