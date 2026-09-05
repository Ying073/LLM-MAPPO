"""
env_wrapper.py —— 把 SearchEnv 包成多智能体 MAPPO 接口

论文对应：
- 公式 18：O_n(t)，每架 UAV 的局部观测（每架 UAV 各自看一份）
- 公式 19：A_n(t) = {0..5} 离散动作空间（北/东/南/西/升/降）
- §IV-A (公式 18 扩展 + S034)：O_n(t) 之上再叠加 DPES 信息素 patch DP_n(t)，
  一起作为 Actor 的输入（见 dpes.py 的 get_patch）。

实际进入神经网络的观测 O_n(t) 由以下几部分组成（按论文"分散式 Actor"思想，
每架 UAV 只看自己附近的局部信息；Critic 拥有全局视野，见 networks.py）：

    自身状态(3)        = [ix_norm, iy_norm, h_norm]
    感知域 patch(50)   = 5x5 窗口内的 全局不确定度(25) + 目标存在状态(25)
    DPES 信息素 patch(25) = 5x5 窗口内的 信息素场 (正=吸引, 负=排斥)  ← M3 新增
    最近障碍(3)        = [dx_norm, dy_norm, dist_norm]
    最近 UAV(3)        = [dx_norm, dy_norm, dh_norm]
    自身能量(1)        = [energy]
    ─────────────────
    无 DPES：60 维  (M2 基线)
    有 DPES：85 维  (M3)
"""

import numpy as np

from .search_env import (
    SearchEnv, LX, LY, N_UAV, N_OBSTACLE, MAX_STEPS,
    HEIGHTS, SENSE_SIZE, DX, DY,
)
from ..reward.manual_reward import compute_manual_reward, SENSE_OFFSETS_BY_H
from ..algorithms.dpes import PheromoneMap


# 5x5 局部窗口中"实际可见格子"的偏移，用 UAV 当前高度档 h 索引
SENSE_OFFSETS_BY_H = {
    0: [(0, 0)],                                                  # 1 grid
    1: [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],               # 5 grids
    2: [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)],     # 9 grids
}


class MultiAgentWrapper:
    """把 SearchEnv 包成多智能体接口，让 MAPPO 直接训练。

    用法：
        env = MultiAgentWrapper(seed=0)
        obs_list = env.reset()         # list, 长度 N_UAV, 每个 shape (60,)
        actions = [3, 1, 0, ...]       # list, 长度 N_UAV
        next_obs_list, env_reward, done, info = env.step(actions)
    """

    PATCH = 5       # 5x5 局部窗口（覆盖所有档位的可见格 + 外圈 unknown）
    HALF = 2        # PATCH // 2 = 2，所以窗口中心是 UAV 自身
    # 基础 60 维 + DPES 信息素 patch 25 维 = 85 维 (M3)
    OBS_DIM = 3 + 25 + 25 + 25 + 3 + 3 + 1  # = 85

    def __init__(self, seed: int = 0, use_dpes: bool = True,
                 lrs_reward_fn=None):
        """
        Args:
            seed       : 环境随机种子
            use_dpes   : 是否启用 DPES 信息素（M3，公式 13–17）
            lrs_reward_fn : 可选，LLM 生成的稠密奖励函数
                       签名 `reward(env, n, action, prev_au) -> float`
                       取自 `reproduction.lrs.LRS.run()` 的 `best_fn`。
                       提供时走 LRS+安全惩罚路径，否则用 manual_reward。
                       对应论文 §IV-C：MAPPO 训练时固定使用 LRS 生成的 R^best。
        """
        self.env = SearchEnv(seed=seed)
        self.use_dpes = use_dpes
        self.dpes = PheromoneMap() if use_dpes else None
        self.energy = np.full(N_UAV, 1.0, dtype=np.float32)  # E_n(0)/E^ini ∈ [0,1]
        self.prev_area_uncertainty = None    # 上一步的 area uncertainty，手写奖励用
        self.lrs_reward_fn = lrs_reward_fn  # None → 用 manual_reward；非 None → LRS 路径

    # ----------------------------------------------------------
    # 把内部 env 状态转成 MAPPO 需要的接口
    # ----------------------------------------------------------
    def reset(self):
        """重置环境。返回每个 UAV 的初始观测列表。"""
        self.env.reset()
        self.energy[:] = 1.0
        self.prev_area_uncertainty = self.env.area_uncertainty()
        self.prev_searched = self.env.searched.copy()   # 跟踪"本步新增搜索确认"
        if self.dpes is not None:
            self.dpes.reset()
        return [self._uav_obs(n) for n in range(N_UAV)]

    def step(self, actions):
        """执行一个时间步。返回 (next_obs_list, env_reward, done, info)。

        论文 §IV-D：奖励函数由 MAPPO 网络外部分配，本环境只负责提供
        一份"中间 reward 信号"（手写稠密奖励，替代 LRS）。
        """
        actions = np.asarray(actions, dtype=np.int64)
        assert actions.shape == (N_UAV,)

        # (1) 推进基础 env，得到全局信息
        _, _, done, env_info = self.env.step(actions.tolist())

        # (1.5) [M3] 更新 DPES 信息素场 (公式 13–17)
        if self.dpes is not None:
            self.dpes.update(self.env)

        # (2) 每架 UAV 简化的能耗模型: 升/降 0.001，水平/悬停 0.0005
        #     (公式 1–3 在 M2 暂时用常数替代，物理能耗估计放到 M3 调优)
        for n, a in enumerate(actions):
            self.energy[n] = max(0.0, self.energy[n] - (0.001 if a in (4, 5) else 0.0005))

        # (3) 计算稠密奖励 —— LRS 路径 (M5) 或 手写稠密 (M2/M3/M4 单独跑时)
        cur_au = self.env.area_uncertainty()
        cur_searched = self.env.searched.copy()
        # "本步新确认的格子"集合，用于搜索奖励的因果信用分配
        newly_confirmed = cur_searched.astype(bool) & ~self.prev_searched.astype(bool)

        if self.lrs_reward_fn is not None:
            # M5 路径: R^best (R_best) 主项 + 安全/能耗惩罚
            shared_reward, per_agent_reward = self._lrs_path_reward(
                actions=actions,
                prev_au=self.prev_area_uncertainty,
            )
        else:
            # M2/M3 路径: manual_reward (含 W_SEARCH/W_COVER + 安全)
            shared_reward, per_agent_reward = compute_manual_reward(
                env=self.env,
                actions=actions,
                energy=self.energy.copy(),
                prev_au=self.prev_area_uncertainty,
                cur_au=cur_au,
                newly_confirmed=newly_confirmed,
            )
        self.prev_area_uncertainty = cur_au
        self.prev_searched = cur_searched

        # (4) 收集 per-agent 局部观测 + 给 Critic 的全局 state
        local_obs = [self._uav_obs(n) for n in range(N_UAV)]
        info = dict(env_info)
        info["per_agent_reward"] = per_agent_reward   # shape (N_UAV,)
        info["newly_confirmed_count"] = int(newly_confirmed.sum())  # [新增] 方便 debug
        return local_obs, shared_reward, done, info

    # ----------------------------------------------------------
    # [M5] LRS 路径: R^best(env, n, a, prev_au) + 安全惩罚
    # ----------------------------------------------------------
    # 为什么还要补"安全"项: 论文 R1/R3/Rbest 都不显式惩罚碰撞，
    # 但 MAPPO 训 W/O action masking 时若没有 -2 这种项，UAV 会反复撞墙。
    # 我们取最小集合：碰撞/越界/超限升降 + 移动能耗，仅作 baseline safety。
    SAFETY_W = {
        "collision":   -2.0,   # 想进入被拒
        "boundary":    -1.0,   # 试图升到顶/降到底
        "height_chg": -0.05,   # 普通升/降
        "move":        -0.02,   # 水平移动一步
        "energy_low":  -0.5,   # 能量 < 5% 时的额外罚
    }

    def _lrs_path_reward(self, actions, prev_au):
        """LRS + 安全路径。每架 UAV:
            per[n] = R_best(env_post_step, n, action[n], prev_au) + safety(action[n])
        """
        actions = np.asarray(actions, dtype=np.int64)
        per = np.zeros(N_UAV, dtype=np.float32)
        for n in range(N_UAV):
            ix, iy, h = self.env.uav_pos[n]
            a = int(actions[n])
            # R^best 主项（已含搜索/不确定度/协作）
            per[n] += float(self.lrs_reward_fn(self.env, n, a, prev_au))
            # 安全惩罚
            if 0 <= a <= 3:
                nx, ny = ix + int(DX[a]), iy + int(DY[a])
                if not (0 <= nx < LX and 0 <= ny < LY and self.env.occ[ny, nx] == 0):
                    per[n] += self.SAFETY_W["collision"]
                else:
                    per[n] += self.SAFETY_W["move"]
            elif (a == 4 and h >= len(HEIGHTS) - 1) or (a == 5 and h <= 0):
                per[n] += self.SAFETY_W["boundary"]
            elif a in (4, 5):
                per[n] += self.SAFETY_W["height_chg"]
            if self.energy[n] < 0.05:
                per[n] += self.SAFETY_W["energy_low"]
        shared_reward = float(per.mean())
        return shared_reward, per

    def get_global_state(self) -> np.ndarray:
        """给 Critic 用的全局 state: 把所有 UAV 的局部观测拼起来 (公式 25 输入)。"""
        return np.concatenate([self._uav_obs(n) for n in range(N_UAV)], axis=0)

    # ----------------------------------------------------------
    # 内部：构造第 n 架 UAV 的 60 维观测
    # ----------------------------------------------------------
    def _uav_obs(self, n: int) -> np.ndarray:
        ix, iy, h = self.env.uav_pos[n]

        # (1) 自身状态 3 维
        own = np.array([
            ix / (LX - 1),
            iy / (LY - 1),
            h / max(1, len(HEIGHTS) - 1),
        ], dtype=np.float32)

        # (2)(3) 5x5 patch: 可见格填实况, 其余填 0 (unknown)
        geum_patch = np.zeros((self.PATCH, self.PATCH), dtype=np.float32)
        zeta_patch = np.zeros((self.PATCH, self.PATCH), dtype=np.float32)
        for ddy, ddx in SENSE_OFFSETS_BY_H[h]:
            py = self.HALF + ddy
            px = self.HALF + ddx
            ny, nx = iy + ddy, ix + ddx
            if 0 <= py < self.PATCH and 0 <= px < self.PATCH and 0 <= ny < LY and 0 <= nx < LX:
                geum_patch[py, px] = self.env.geum[ny, nx]
                zeta_patch[py, px] = self.env.zeta[ny, nx]
        geum_flat = geum_patch.flatten()
        zeta_flat = zeta_patch.flatten()

        # (4) 最近障碍：方向 + 距离 (3 维)
        ob_dx, ob_dy, ob_d = _nearest_obstacle(self.env.occ, ix, iy)
        max_dim = max(LX, LY)
        obstacle = np.array([
            ob_dx / max_dim,
            ob_dy / max_dim,
            min(ob_d, max_dim) / max_dim,
        ], dtype=np.float32)

        # (5) 最近的其他 UAV：方向 + 距离 + 高度差 (3 维)
        u_dx, u_dy, u_dh = _nearest_uav(self.env.uav_pos, n)
        uav_other = np.array([
            u_dx / max_dim,
            u_dy / max_dim,
            u_dh / max(1, len(HEIGHTS) - 1),
        ], dtype=np.float32)

        # (6) 自身能量 (1 维)
        energy = np.array([self.energy[n]], dtype=np.float32)

        # (7) [M3] DPES 信息素 patch (25 维): 感知域内的信息素场
        if self.dpes is not None:
            dp_patch = self.dpes.get_patch(self.env, n, half=self.HALF)
            dp_flat = dp_patch.flatten()
        else:
            dp_flat = np.zeros(self.PATCH * self.PATCH, dtype=np.float32)

        if self.dpes is not None:
            return np.concatenate([own, geum_flat, zeta_flat, dp_flat, obstacle, uav_other, energy])
        else:
            # 无 DPES (M2 基线), 保持 60 维
            return np.concatenate([own, geum_flat, zeta_flat, obstacle, uav_other, energy])


# ---------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------
def _nearest_obstacle(occ: np.ndarray, ix: int, iy: int):
    """返回离 (ix, iy) 最近的障碍物 (dx, dy, dist)，用 Chebyshev 距离。"""
    if occ.sum() == 0:
        return 0, 0, max(LX, LY)
    # 用 numpy 向量化扫
    ys, xs = np.where(occ == 1)
    dx_arr = xs - ix
    dy_arr = ys - iy
    d_arr = np.maximum(np.abs(dx_arr), np.abs(dy_arr))
    idx = int(d_arr.argmin())
    return int(dx_arr[idx]), int(dy_arr[idx]), int(d_arr[idx])


def _nearest_uav(uav_pos: np.ndarray, n: int):
    """返回离 UAV n 最近的 *其他* UAV 的 (dx, dy, dh)。"""
    self_pos = uav_pos[n]
    deltas = uav_pos.copy()
    deltas[n] = self_pos + 9999    # 排除自身
    dx_arr = deltas[:, 0] - self_pos[0]
    dy_arr = deltas[:, 1] - self_pos[1]
    dh_arr = deltas[:, 2] - self_pos[2]
    d_arr = np.maximum(np.abs(dx_arr), np.abs(dy_arr))
    idx = int(d_arr.argmin())
    return int(dx_arr[idx]), int(dy_arr[idx]), int(dh_arr[idx])
