import numpy as np


def bayes_update_prob(p: float, detected: bool, p_d: float, p_f: float) -> float:
    """Eq.(6)：贝叶斯更新目标存在概率。detected=True 对应检测到目标。"""
    if detected:
        num = p * p_d
        den = p * p_d + (1.0 - p) * p_f
    else:
        num = p * (1.0 - p_d)
        den = p * (1.0 - p_d) + (1.0 - p) * (1.0 - p_f)
    return num / den if den > 0 else p


def entropy(p: float) -> float:
    """Eq.(7)：信息熵，归一化到 [0,1]。p=0 或 1 时熵为 0。"""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)


def fuse_global(local_unc: np.ndarray, local_prob: np.ndarray):
    """Eq.(8)(10)：每网格全局不确定度取最小；全局概率取最小不确定度 UAV 的概率（并列取最大概率）。"""
    g_unc = local_unc.min(axis=0)
    min_mask = (local_unc == g_unc[None, :])
    masked_prob = np.where(min_mask, local_prob, -1.0)
    g_prob = masked_prob.max(axis=0)
    return g_unc, g_prob
