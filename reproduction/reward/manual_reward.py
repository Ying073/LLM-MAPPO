"""
manual_reward.py —— 手写稠密奖励 (替代论文 §IV-C 的 LLM 奖励塑形)

论文对应：
- 公式 20：候选奖励下的贪心动作 a_n^*(t) — 我们这里不需要，直接给奖励
- 公式 21：当前迭代最优奖励函数选择 — 我们这里直接用这一份手写奖励
- 公式 22：奖励函数反馈提示集合 — 给 LLM 看的内容，我们这里手写就略了

为什么要写手写稠密奖励：
    公式 12a 的优化目标只会在"搜到目标"或"任务结束"时给出非零梯度 → 稀疏奖励，
    这正是论文 §I-B / §I-D 强调的"现有 MARL 方法的痛点"。手写稠密奖励的思路
    是把目标拆成几个"每步都能拿到信号"的项：

    R_step = R_search + R_cover + R_collision + R_energy

【M2 关键修复 2026-09-05】
    上一版训练的失败原因：搜索奖励把"累计已确认数"当 step 信号用——
    每步都重复给已搜到的格子发奖励，所有 UAV 拿同样的值，actor 看不到
    "探索新区域"与奖励的因果关联。这一版重写为：
        - 新确认格子 = (cur_searched) & ~(prev_searched)
        - 每个新格子产生 +W_SEARCH 的"团队奖励"
        - 按"哪架 UAV 的感知域覆盖了该格子"分到对应 UAV
            (覆盖该格的 UAV 越多，每个分到的越少)
        - 这样策略梯度能直接学到"要让感知域扫到未搜到的格子"
"""

import numpy as np

from ..env.search_env import (
    N_UAV, LX, LY,
    HEIGHTS,
)


# 各项奖励的权重（M2 阶段先给一组"能学"的默认值，调参留到 M3）
W_SEARCH = +10.0    # 每搜到一个新格子 = +10
W_COVER = +100.0    # area uncertainty 单调下降的 rescale：单步 Δau≈0.002 → +0.2
W_COLLISION = -2.0  # 想进入被拒 → -2
W_BOUNDARY = -1.0   # 越界动作 → -1
W_HEIGHT_CHANGE = -0.05  # 升/降一步的能耗惩罚
W_MOVE = -0.02      # 水平移动一步的能耗惩罚
W_ENERGY_LOW = -0.5 # 能量 < 5% 的额外惩罚


# UAV 在每个高度档"能感知到的格子的相对偏移" (与 env_wrapper.py 保持一致)
SENSE_OFFSETS_BY_H = {
    0: [(0, 0)],                                                  # 1 grid
    1: [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],               # 5 grids
    2: [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)],     # 9 grids
}


def _uav_sensed_cells(uav_pos: np.ndarray) -> list[np.ndarray]:
    """返回每架 UAV 的感知域覆盖的格子坐标 (N_UAV 个 (k, 2) 数组).

    论文公式 4：UAV 在高度 h 档时的可见格集合 S_n(t)。
    """
    out = []
    for ix, iy, h in uav_pos:
        cells = []
        for ddy, ddx in SENSE_OFFSETS_BY_H[int(h)]:
            ny, nx = iy + ddy, ix + ddx
            if 0 <= ny < LY and 0 <= nx < LX:
                cells.append((ny, nx))
        out.append(np.asarray(cells, dtype=np.int64) if cells else np.zeros((0, 2), dtype=np.int64))
    return out


def compute_manual_reward(
    env,
    actions: np.ndarray,
    energy: np.ndarray,
    prev_au: float,
    cur_au: float,
    newly_confirmed: np.ndarray,    # [新增] (LY, LX) bool, 本步新确认的格子
) -> tuple[float, np.ndarray]:
    """计算手写稠密奖励。

    输入:
        env              : SearchEnv 实例 (用 .occ 判断撞障碍)
        actions          : (N_UAV,) int 本步所有 UAV 的动作
        energy           : (N_UAV,) float 当前能量
        prev_au          : float 上一步 area uncertainty
        cur_au           : float 本步 area uncertainty
        newly_confirmed  : (LY, LX) bool 本步从 0 变 1 的格子集合

    返回:
        shared_reward   : float 团队奖励 (MAPPO 的 shared value baseline)
        per_agent_reward: (N_UAV,) float 每架 UAV 的奖励分解 (策略梯度)
    """
    actions = np.asarray(actions, dtype=np.int64)
    per = np.zeros(N_UAV, dtype=np.float32)

    # ------------------------------------------------------------------
    # (1) 搜索奖励：本步新确认的格子 → 按"哪架 UAV 的感知域覆盖"分配
    # ------------------------------------------------------------------
    newly_n = int(newly_confirmed.sum())
    if newly_n > 0:
        # 取出本步所有新格子的坐标 (k, 2)
        new_cells = np.argwhere(newly_confirmed)   # shape (k, 2)

        # 找每架 UAV 的感知域覆盖了哪些新格子
        sensed = _uav_sensed_cells(env.uav_pos)    # list of (k_i, 2)
        # 为加速：把 new_cells 变成 (ny, nx) 元组集合
        new_set = set(map(tuple, new_cells.tolist()))

        # 每架 UAV "它感知域内、本步新确认"的格子数
        uav_new_count = np.zeros(N_UAV, dtype=np.int64)
        for n, cells in enumerate(sensed):
            for (ny, nx) in cells:
                if (int(ny), int(nx)) in new_set:
                    uav_new_count[n] += 1

        total_coverage = int(uav_new_count.sum())
        if total_coverage > 0:
            # 总搜索奖励按"贡献的新格子数"分给 UAV
            share = W_SEARCH * newly_n
            per += W_SEARCH * newly_n * (uav_new_count / total_coverage)
        else:
            # 极少见：UAV 都没感知到，但 ground truth 确实新增了
            # → 退化为均匀分配（避免奖励全 0）
            per += W_SEARCH * newly_n / N_UAV

    # ------------------------------------------------------------------
    # (2) 覆盖奖励：area uncertainty 单调下降的 Δ
    #     Δau 典型量级 0.001~0.003/步，乘 W_COVER=100 → ±0.1~0.3/步
    # ------------------------------------------------------------------
    per += W_COVER * (prev_au - cur_au)

    # ------------------------------------------------------------------
    # (3) 撞障碍 / 越界 / 能耗
    # ------------------------------------------------------------------
    from ..env.search_env import DX, DY
    for n, a in enumerate(actions):
        ix, iy, h = env.uav_pos[n]
        if 0 <= a <= 3:
            nx, ny = ix + int(DX[a]), iy + int(DY[a])
            if not (0 <= nx < LX and 0 <= ny < LY and env.occ[ny, nx] == 0):
                per[n] += W_COLLISION
            else:
                per[n] += W_MOVE
        elif a == 4 and h >= len(HEIGHTS) - 1:
            per[n] += W_BOUNDARY
        elif a == 5 and h <= 0:
            per[n] += W_BOUNDARY
        elif a in (4, 5):
            per[n] += W_HEIGHT_CHANGE

        if energy[n] < 0.05:
            per[n] += W_ENERGY_LOW

    shared_reward = float(per.mean())   # MAPPO shared baseline
    return shared_reward, per
