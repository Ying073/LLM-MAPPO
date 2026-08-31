import numpy as np
from llm_mappo.env.maps import fuse_global


def sparse_reward(env) -> np.ndarray:
    """原始稀疏目标：找到目标给 1，否则 0（MAPPO 基线）。"""
    r = np.zeros(env.cfg.n_uav)
    if env._searched_targets() > 0:
        r += 1.0
    return r


def handcraft_reward(env) -> np.ndarray:
    """人工稠密奖励（Handcraft 基线）：目标搜索 + 不确定度下降 + 分散。"""
    n = env.cfg.n_uav
    r = np.zeros(n)
    r += env._searched_targets()            # 目标搜索
    r += (1.0 - env._avg_uncertainty())     # 低不确定度
    # 分散项：UAV 间最小距离的奖励
    pos = env.uav_pos[:, :2].astype(float)
    if n > 1:
        diff = pos[:, None, :] - pos[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        d[d == 0] = np.inf
        r += d.min(axis=1) * 0.001
    return r


def mdps_reward(env) -> np.ndarray:
    """MDPS 基线：朝向当前目标概率最高网格的贪心奖励。"""
    _, fused_prob = fuse_global(env.local_unc, env.local_prob)
    return np.full(env.cfg.n_uav, float(fused_prob.max()))


def mdps_improved_reward(env) -> np.ndarray:
    """MDPS-improved：在 MDPS 目标概率上叠加 DPES 信息素（信息素场已在环境中维护）。"""
    _, fused_prob = fuse_global(env.local_unc, env.local_prob)
    signal = fused_prob + env.pheromone
    return np.full(env.cfg.n_uav, float(signal.max()))
