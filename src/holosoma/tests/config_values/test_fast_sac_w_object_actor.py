"""g1_29dof_wbt_fast_sac_w_object_actor: FastSAC with the observations of w_object_actor (the PPO preset of the box
runs), so that an A/B of the two algorithms differs by the algorithm alone (OTRNet, E2).

    cd src/holosoma && $PYS -m pytest tests/config_values/test_fast_sac_w_object_actor.py -q
"""

from __future__ import annotations

import dataclasses

from holosoma.config_values.experiment import DEFAULTS
from holosoma.config_values.wbt.g1 import experiment as E


def test_registered():
    assert DEFAULTS["g1_29dof_wbt_fast_sac_w_object_actor"] is E.g1_29dof_wbt_fast_sac_w_object_actor


def test_it_is_fast_sac_w_object_with_the_actor_observations():
    new, base = E.g1_29dof_wbt_fast_sac_w_object_actor, E.g1_29dof_wbt_fast_sac_w_object
    assert new.observation == E.g1_29dof_wbt_w_object_actor.observation
    assert new.training.name == "g1_29dof_wbt_fast_sac_w_object_actor"
    assert dataclasses.replace(new, observation=base.observation, training=base.training) == base


def test_against_the_ppo_preset_only_the_algorithm_and_the_run_name_differ():
    new, ppo = E.g1_29dof_wbt_fast_sac_w_object_actor, E.g1_29dof_wbt_w_object_actor
    same = [f.name for f in dataclasses.fields(new) if f.name not in ("algo", "training", "nightly")]
    assert [k for k in same if getattr(new, k) != getattr(ppo, k)] == []
    assert new.algo != ppo.algo and new.training.num_envs == ppo.training.num_envs
