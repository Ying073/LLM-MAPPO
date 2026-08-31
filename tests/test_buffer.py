import numpy as np
from llm_mappo.mappo.buffer import RolloutBuffer


def test_gae_single_step_terminal():
    buf = RolloutBuffer(n_agents=1, obs_dim=2, n_actions=2)
    buf.add(obs=np.zeros((1, 2)), actions=np.zeros(1, dtype=int),
            logits=np.zeros((1, 2)), rewards=np.array([5.0]), dones=np.array([1.0]))
    buf.compute_gae(values=np.array([[2.0]]), gamma=0.95, lam=0.95)
    # 终止步：advantage = r - V，return = r
    assert np.allclose(buf.advantages[0], [3.0])
    assert np.allclose(buf.returns[0], [5.0])


def test_gae_shapes():
    buf = RolloutBuffer(n_agents=2, obs_dim=4, n_actions=3)
    for _ in range(3):
        buf.add(obs=np.zeros((2, 4)), actions=np.zeros(2, dtype=int),
                logits=np.zeros((2, 3)), rewards=np.ones(2), dones=np.zeros(2))
    values = np.ones((3, 2))
    buf.compute_gae(values, gamma=0.95, lam=0.95)
    assert buf.advantages.shape == (3, 2)
    assert buf.returns.shape == (3, 2)
