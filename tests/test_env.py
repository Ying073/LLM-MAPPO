import numpy as np
from llm_mappo.env.env import SearchEnv


def test_reset_step_shape():
    env = SearchEnv()
    obs, _ = env.reset(seed=0)
    assert len(obs) == env.cfg.n_uav
    actions = np.zeros(env.cfg.n_uav, dtype=int)
    obs, rew, term, trunc, info = env.step(actions)
    assert rew.shape == (env.cfg.n_uav,)
    assert isinstance(term, bool) and isinstance(trunc, bool)


def test_action_masking_blocks_collision():
    env = SearchEnv()
    env.reset(seed=0)
    masks = env.action_masks()
    assert masks.shape == (env.cfg.n_uav, 6)
    assert (masks.sum(axis=1) > 0).all()


def test_obs_dim_consistent():
    env = SearchEnv()
    obs, _ = env.reset(seed=0)
    dims = [o.shape[0] for o in obs]
    assert len(set(dims)) == 1          # 所有 UAV 观测维度一致
