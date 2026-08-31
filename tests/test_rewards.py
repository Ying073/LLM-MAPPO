import numpy as np
from llm_mappo.env.env import SearchEnv
from llm_mappo.utils.config import Config
from llm_mappo.baselines.rewards import (
    sparse_reward, handcraft_reward, mdps_reward, mdps_improved_reward,
)


def test_all_rewards_shape_and_finite():
    cfg = Config()
    env = SearchEnv(cfg)
    env.reset(seed=0)
    env.step(np.zeros(cfg.env.n_uav, dtype=int))
    for fn in [sparse_reward, handcraft_reward, mdps_reward, mdps_improved_reward]:
        r = fn(env)
        assert r.shape == (cfg.env.n_uav,)
        assert np.isfinite(r).all()
