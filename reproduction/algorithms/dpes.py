"""
dpes.py —— 双模式信息素高效搜索机制 (Dual-mode Pheromone-based Efficient Search)

复现对象：论文 §IV-A (paper.md S030–S034, 算法 1, 公式 13–17)。

一句话：
    DPES 用一套"信息素场"给每个网格打"引导信号"。它让 UAV 优先去
    (a) 目标可能较多的高价值网格、和 (b) 很久没去、目标可能又出现的网格；
    同时抑制 UAV 反复搜已经确认过的网格。

四条更新规则 ↔ 公式：
    公式 13  网格分类 C(g_i)
    公式 14  G_hv 高价值网格的吸引素更新     (蒸发 E_s + 扩散 G_s)
    公式 15  邻域扩散进入 g_i 的素总量 f_i(t)
    公式 16  G_lu 长时间未访问网格的吸引素   (不蒸发、不扩散、访问即清零)
    公式 17  G_cs 已确认状态网格的排斥素     (负值、蒸发、不扩散、超时重置)

参数来源：论文表 II (paper.md 中文表注, 公式 13 正文)
    D    = 200   重访阈值 (time steps)
    E_s  = 0.1   蒸发系数
    G_s  = 0.1   扩散系数
    d_hv = 0.1   G_hv 释放量 (吸引)
    d_lu = 0.003 G_lu 释放量 (吸引, 持续累加)
    d_cs = 0.2   G_cs 释放量 (排斥)
"""

import numpy as np

from ..env.search_env import (
    LY, LX, N_UAV,
    TARGET_CONFIRM_THRESHOLD,
)


# ============================================================
# 参数 (论文表 II + 公式 13 语义默认)
# ============================================================
D = 200                      # 重访阈值: t - t_last > D 视为 G_lu
E_S = 0.1                    # 蒸发系数 E_s
G_S = 0.1                    # 扩散系数 G_s
D_HV = 0.1                   # 高价值网格释放量 (吸引)
D_LU = 0.003                 # 长时间未访问释放量 (吸引)
D_CS = 0.2                   # 已确认状态释放量 (排斥)

# 公式 13 里 p_max / p_min:
#   p >= p_max(0.8) 或 p <= p_min(0.2) → 目标状态已确认 → G_cs
#   0.5 < p <  p_max                    → 中高价值       → G_hv
# p_max 用环境的确认阈值 (公式 11 的 ξ), p_min 取对称的 1 - p_max
P_MIN = 1.0 - TARGET_CONFIRM_THRESHOLD   # 0.2
P_MAX = TARGET_CONFIRM_THRESHOLD         # 0.8


# 网格类别枚举
G_LU, G_HV, G_CS, G_OTHER = 0, 1, 2, 3


def _neighbors(ix: int, iy: int):
    """返回 g_i=(ix,iy) 的合法 4 邻居 (上下左右), 越界的丢弃。"""
    out = []
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        nx, ny = ix + dx, iy + dy
        if 0 <= nx < LX and 0 <= ny < LY:
            out.append((nx, ny))
    return out


class PheromoneMap:
    """双模式信息素场。持有 (LY, LX) 的 dp，每步 update 一次。

    用法 (接入 env_wrapper):
        ph = PheromoneMap()
        ph.reset()                     # reset() 时调用
        ...
        ph.update(env)                 # 每步在 env.step 之后调用
        patch = ph.get_patch(env, n)   # 供 UAV n 的观测使用
    """

    def __init__(self, d: int = D, e_s: float = E_S, g_s: float = G_S,
                 d_hv: float = D_HV, d_lu: float = D_LU, d_cs: float = D_CS,
                 p_min: float = P_MIN, p_max: float = P_MAX):
        self.d = d
        self.e_s = e_s
        self.g_s = g_s
        self.d_hv = d_hv
        self.d_lu = d_lu
        self.d_cs = d_cs
        self.p_min = p_min
        self.p_max = p_max
        self.dp = np.zeros((LY, LX), dtype=np.float32)   # 信息素场 (正=吸引, 负=排斥)

    # ----------------------------------------------------------
    def reset(self):
        """重置信息素场为全 0 (每个 episode 开始调用)。"""
        self.dp.fill(0.0)

    # ----------------------------------------------------------
    def _classify(self, env) -> np.ndarray:
        """公式 13: 给每个网格分类。

        输入读 env 的:
            env.gtpm          → p_{n,i}(t)  全局目标概率 (融合后的)
            env.t_last_visit  → t_i^{last}  上次访问时间
            env.t             → t
        """
        t_since = env.t - env.t_last_visit          # (LY, LX) 距上次访问的时间
        p = env.gtpm                                # (LY, LX) 目标概率

        cls = np.full((LY, LX), G_OTHER, dtype=np.int8)

        # G_lu: 超时未访问 (此时无论 p 多少, 都是 G_lu, 优先级最高)
        cls[t_since > self.d] = G_LU

        # G_hv / G_cs: 只对"仍然新鲜" (t_since <= D) 的网格
        fresh = t_since <= self.d

        hv = fresh & (p > 0.5) & (p < self.p_max)
        cs = fresh & ((p <= self.p_min) | (p >= self.p_max))
        cls[hv] = G_HV
        cls[cs] = G_CS

        # 冲突处理: 若某格同时满足 hv 和 cs 判据, G_cs 优先 (状态已确认更明确)
        # 由于 (0.5<p<p_max) 与 (p<=p_min 或 p>=p_max) 互斥, 实际上不会重叠。
        return cls

    # ----------------------------------------------------------
    def update(self, env):
        """公式 14–17: 按类别更新每个网格的信息素。

        必须在 env.step(actions) 之后调用 (此时 uav_pos / gtpm / t_ext / t_last_visit 已更新)。
        """
        cls = self._classify(env)

        # 本步被 UAV 访问的格子集合 (dp 的"释放"只看当前 UAV 是否在该格)
        visited = set()
        for n in range(N_UAV):
            ix, iy = int(env.uav_pos[n, 0]), int(env.uav_pos[n, 1])
            visited.add((ix, iy))

        # ---- 公式 15: 预计算每格的"邻居扩散进来的总量 f_i" (只对 G_hv 有意义) ----
        f = np.zeros((LY, LX), dtype=np.float32)
        for iy in range(LY):
            for ix in range(LX):
                nb = _neighbors(ix, iy)
                if not nb:
                    continue
                # 公式 15: f_i = Σ_{i'∈N_i} (G_s/|N_i|)·[dp_{i'}(t-1) + d_{i'}^hv(t)]
                acc = 0.0
                for (nx, ny) in nb:
                    # d_{i'}^hv: 若邻居 i' 被访问且属于 G_hv / G_lu, 则本步释放
                    dep = 0.0
                    if (nx, ny) in visited:
                        if cls[ny, nx] == G_HV:
                            dep = self.d_hv
                        elif cls[ny, nx] == G_LU:
                            dep = self.d_lu
                    acc += self.g_s / len(nb) * (self.dp[ny, nx] + dep)
                f[iy, ix] = acc

        # ---- 逐格按类别更新 (公式 14 / 16 / 17) ----
        for iy in range(LY):
            for ix in range(LX):
                c = cls[iy, ix]
                if c == G_HV:
                    # 公式 14: dp = (1-E_s)·{(1-G_s)·[dp + d^hv + f]}
                    dep = self.d_hv if (ix, iy) in visited else 0.0
                    self.dp[iy, ix] = ((1 - self.e_s)
                                       * ((1 - self.g_s) * (self.dp[iy, ix] + dep + f[iy, ix])))
                elif c == G_LU:
                    # 公式 16: dp = dp + d^lu
                    self.dp[iy, ix] = self.dp[iy, ix] + self.d_lu
                    if (ix, iy) in visited:
                        self.dp[iy, ix] = 0.0            # 访问后立即清零
                elif c == G_CS:
                    # 公式 17: dp = -(1-E_s)·[dp + d^cs]  (排斥, 负值)
                    self.dp[iy, ix] = -(1 - self.e_s) * (self.dp[iy, ix] + self.d_cs)
                else:  # G_OTHER
                    self.dp[iy, ix] = 0.0                # 其他网格不释放信息素

    # ----------------------------------------------------------
    def get_field(self) -> np.ndarray:
        """返回整个信息素场副本 (用于可视化 / 调试)。"""
        return self.dp.copy()

    # ----------------------------------------------------------
    def get_patch(self, env, n: int, half: int = 2) -> np.ndarray:
        """返回 UAV n 感知域内的信息素 patch (公式 4 的传感域 + DP_n(t)).

        用与 env_wrapper 相同的 5x5 窗口约定:
            - center = UAV 自身 (ix, iy)
            - 越界格子填 0
        返回 (5, 5) float32。

        论文 S034: DP_n(t) = { dp_i(t) | g_i ∈ Φ_n(t) } —— 作为 Actor 输入之一。
        """
        ix, iy, _ = env.uav_pos[n]
        patch = np.zeros((2 * half + 1, 2 * half + 1), dtype=np.float32)
        for ddy in range(-half, half + 1):
            for ddx in range(-half, half + 1):
                ny, nx = iy + ddy, ix + ddx
                if 0 <= ny < LY and 0 <= nx < LX:
                    patch[half + ddy, half + ddx] = self.dp[ny, nx]
        return patch
