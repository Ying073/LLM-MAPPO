import numpy as np
from llm_mappo.env.maps import bayes_update_prob, entropy, fuse_global


def test_bayes_detection_increases_prob():
    p_new = bayes_update_prob(0.5, detected=True, p_d=0.9, p_f=0.1)
    assert p_new > 0.5


def test_bayes_no_detection_decreases_prob():
    p_new = bayes_update_prob(0.5, detected=False, p_d=0.9, p_f=0.1)
    assert p_new < 0.5


def test_entropy_max_at_half():
    e = entropy(0.5)
    assert abs(e - 1.0) < 1e-6


def test_entropy_zero_at_extremes():
    assert entropy(0.0) == 0.0
    assert entropy(1.0) == 0.0


def test_fuse_global_min_uncertainty():
    # 3 架 UAV 对同一网格的不确定度取最小
    local_unc = np.array([[0.8], [0.3], [0.6]])      # (n_uav, n_grid)
    local_prob = np.array([[0.5], [0.9], [0.7]])
    g_unc, g_prob = fuse_global(local_unc, local_prob)
    assert g_unc[0] == 0.3
    assert g_prob[0] == 0.9   # 唯一最小不确定度的 UAV 的概率
