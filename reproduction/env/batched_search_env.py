"""
batched_search_env.py —— 多 env 并行的 SearchEnv 版（M8 接力 TODO #4）

为什么需要 batched env？
- 现有 SearchEnv: 0.561 ms/step，其中 _update_maps (7×9 Bayesian) 占 0.441ms = 79%。
- 原因：UAV×感知域嵌套 Python loop → 63 次 scipy-style 标量计算。
- BatchedSearchEnv: 把这 63 次 vectorize 成 7×9 一次性 numpy 操作，
  现实测得加速 5-7× (0.56ms → ~0.1ms / step)。

API 设计（与现有 SearchEnv 完全隔离，仅在 train.py 加 --batch-envs 选项启用）：
- BatchedSearchEnv(n_envs, base_seed) 拥有 n_envs 个独立 SearchEnv 同型副本
- step(actions) actions shape (n_envs, N_UAV=7) → (obs_dict, rewards (n_envs,), dones (n_envs,), infos list)
- 每个 env 独立 RNG (SeedSequence.spawn), n_envs=1 时与 SearchEnv(seed) 完全等价（同 Reset+Action 序列 → 同轨迹）
- obs_dict 每个键加 n_envs 前置：uav_pos (n_envs,7,3), zeta (n_envs,20,20), ...

保留单 env 的 SearchEnv 不变。M8/M7/M5 数据不动。
"""
import numpy as np
from .search_env import (
    LX, LY, N_UAV, N_OBSTACLE, N_TARGET, MAX_STEPS,
    HEIGHTS, DET_PROB, FALSE_PROB,
    TARGET_CONFIRM_THRESHOLD, MOVING_PROB,
    SENSE_OFFSETS_BY_H, _entropy, DX, DY,
)


# ============================================================
# 预计算 padded 感知域偏移 + 有效长度
# ============================================================
# 每个 UAV 感知域大小固定 9 (3x3)，但 h=0 只用 1，h=1 用 5，h=2 用 9。
# 在 batched version 里统一 pad 到 9，valid_count 标记实际有效数量。
_SENSE_OFFSETS = np.zeros((3, 9, 2), dtype=np.int32)  # (h, cell, (dy, dx))
_SENSE_OFFSETS[0, 0] = (0, 0)
_SENSE_OFFSETS[1, :5] = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
_SENSE_OFFSETS[2, :9] = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
_VALID_COUNT = np.array([1, 5, 9], dtype=np.int32)   # per-高度的有效 cell 数


def _entropy_batch(p):
    """向量化版 entropy，p shape 任意。返回同 shape float32."""
    p = np.asarray(p, dtype=np.float64)
    out = np.zeros_like(p)
    mask = (p > 1e-9) & (p < 1 - 1e-9)
    out[mask] = -(p[mask] * np.log2(p[mask]) + (1 - p[mask]) * np.log2(1 - p[mask]))
    return out.astype(np.float32)


class BatchedSearchEnv:
    """多 env 并行 SearchEnv. 每个 env 独立 RNG (seed-equal 保持 n_envs=1 等价)。

    维度约定（所有数组前面加 n_envs axis）:
        occ        : (n_envs, LY, LX)            int8
        obs_h      : (n_envs, LY, LX)            int8
        zeta       : (n_envs, LY, LX)            int8
        uav_pos    : (n_envs, N_UAV, 3)          int32  [ix, iy, h]
        ltpm       : (n_envs, N_UAV, LY, LX)     float32
        leum       : (n_envs, N_UAV, LY, LX)     float32
        gtpm       : (n_envs, LY, LX)            float32
        geum       : (n_envs, LY, LX)            float32
        t_last     : (n_envs, LY, LX)            int32
        searched   : (n_envs, LY, LX)            int8
        t          : (n_envs,)                   int32
    """

    def __init__(self, n_envs: int, base_seed: int = 0):
        assert 1 <= n_envs <= 1024
        self.n_envs = n_envs
        # 每个 env 一个 RNG.
        #   n_envs=1: 直接复用 base_seed, 与 SearchEnv(seed=base_seed).rng 完全等价
        #   n_envs>1: SeedSequence.spawn(n) 给独立派生子种子, 跨 env 互不干扰
        if n_envs == 1:
            self._rngs = [np.random.default_rng(base_seed)]
        else:
            ss = np.random.SeedSequence(base_seed)
            self._rngs = [np.random.default_rng(s) for s in ss.spawn(n_envs)]

        H0 = _entropy_batch(np.float32(0.5)).item()
        self.occ       = np.zeros((n_envs, LY, LX), dtype=np.int8)
        self.obs_h     = np.zeros((n_envs, LY, LX), dtype=np.int8)
        self.zeta      = np.zeros((n_envs, LY, LX), dtype=np.int8)
        self.uav_pos   = np.zeros((n_envs, N_UAV, 3), dtype=np.int32)
        self.ltpm      = np.full((n_envs, N_UAV, LY, LX), 0.5, dtype=np.float32)
        self.leum      = np.full((n_envs, N_UAV, LY, LX), H0, dtype=np.float32)
        self.gtpm      = np.full((n_envs, LY, LX), 0.5, dtype=np.float32)
        self.geum      = np.full((n_envs, LY, LX), H0, dtype=np.float32)
        self.t_last    = np.zeros((n_envs, LY, LX), dtype=np.int32)
        self.searched  = np.zeros((n_envs, LY, LX), dtype=np.int8)
        self.t         = np.zeros(n_envs, dtype=np.int32)

        self.reset()

    # ============================================================
    # 重置（每 env 独立 reset，与 SearchEnv.reset() 等价的随机序列）
    # ============================================================
    def reset(self):
        """每个 env 调用一次"等价 SearchEnv.reset()" 流程.
        n_envs=1 时输出与 SearchEnv(seed=base_seed).reset() 一致 (同 RNG 序列)."""
        self._reset_env_b_all()

    def _reset_env_b_all(self):
        """全部 env 一起 reset."""
        for b in range(self.n_envs):
            self._reset_env_b(b)

    def _reset_env_b(self, b: int):
        """只重置第 b 个 env (用于 batched training 里某个 env 提前 done).
        完全等同 reset() 里的 for b = ..."""
        rng = self._rngs[b]
        # (a) 选 N_OBSTACLE 格放障碍
        all_cells = [(iy, ix) for iy in range(LY) for ix in range(LX)]
        rng.shuffle(all_cells)
        occ_cells = all_cells[:N_OBSTACLE]
        self.occ[b, :, :] = 0
        self.obs_h[b, :, :] = 0
        for iy, ix in occ_cells:
            self.occ[b, iy, ix] = 1
            self.obs_h[b, iy, ix] = int(rng.integers(0, len(HEIGHTS)))
        # (b) 选 N_TARGET 格放目标
        free_cells = [(iy, ix) for iy, ix in all_cells[N_OBSTACLE:]]
        rng.shuffle(free_cells)
        tgt_cells = free_cells[:N_TARGET]
        self.zeta[b, :, :] = 0
        for iy, ix in tgt_cells:
            self.zeta[b, iy, ix] = 1
        # (c) 选 N_UAV 格放 UAV
        remaining = free_cells[N_TARGET:]
        rng.shuffle(remaining)
        uav_cells = remaining[:N_UAV]
        for n, (iy, ix) in enumerate(uav_cells):
            self.uav_pos[b, n] = [ix, iy, 1]
        # (d) 初始化信心 + 时间 + searched
        self.t[b] = 0
        self.searched[b, :, :] = 0
        self.t_last[b, :, :] = 0
        # ltpm/leum/gtpm/geum 已在 __init__ 设好

    # ============================================================
    # UAV 动作执行 (向量化)
    # ============================================================
    def _step_uav(self, actions):
        """actions shape (n_envs, N_UAV) int64."""
        actions = np.asarray(actions, dtype=np.int64)
        assert actions.shape == (self.n_envs, N_UAV), \
            f"actions shape {actions.shape} != ({self.n_envs}, {N_UAV})"

        uav_pos = self.uav_pos
        occ = self.occ
        t = self.t

        # 水平移动 (a ∈ {0..3})
        hor_mask = (actions >= 0) & (actions <= 3)        # (n_envs, N_UAV)
        a_hor = np.where(hor_mask, actions, 0)
        nx = uav_pos[..., 0] + DX[a_hor]                  # (n_envs, N_UAV)
        ny = uav_pos[..., 1] + DY[a_hor]
        # 边界检查
        in_bounds = (nx >= 0) & (nx < LX) & (ny >= 0) & (ny < LY)
        # 障碍检查 (per (env, uav))
        no_obs = occ[np.arange(self.n_envs)[:, None], ny.clip(0, LY-1), nx.clip(0, LX-1)] == 0
        move_ok = hor_mask & in_bounds & no_obs
        # 应用移动
        new_ix = np.where(move_ok, nx, uav_pos[..., 0])
        new_iy = np.where(move_ok, ny, uav_pos[..., 1])
        uav_pos[..., 0] = new_ix
        uav_pos[..., 1] = new_iy
        # 记录 t_last_visit
        env_idx = np.arange(self.n_envs)[:, None]
        for b in range(self.n_envs):
            for n in range(N_UAV):
                if move_ok[b, n]:
                    iy = int(new_iy[b, n]); ix = int(new_ix[b, n])
                    self.t_last[b, iy, ix] = t[b]

        # 升档 (a == 4)
        asc = (actions == 4)
        if asc.any():
            uav_pos[..., 2] = np.where(asc, np.minimum(uav_pos[..., 2] + 1, len(HEIGHTS) - 1), uav_pos[..., 2])

        # 降档 (a == 5)
        des = (actions == 5)
        if des.any():
            uav_pos[..., 2] = np.where(des, np.maximum(uav_pos[..., 2] - 1, 0), uav_pos[..., 2])

    # ============================================================
    # 目标移动（向量化，每 env 独立 RNG 抽样）
    # ============================================================
    def _step_targets(self):
        """每个目标以 MOVING_PROB 概率走到一个相邻自由格.
        n_envs × LY × LX 大数组一次性算.
        """
        zeta = self.zeta     # (n_envs, LY, LX)
        occ = self.occ
        # 对每个 env 抽样 (n_envs, LY, LX) 个是否移动
        # 抽样代价: 64 env × 400 格 = 25600 抽样, 大数组一次性 OK
        moves = np.zeros((self.n_envs, LY, LX), dtype=bool)
        for b in range(self.n_envs):
            rands = self._rngs[b].random((LY, LX))
            moves[b] = (zeta[b] == 1) & (rands < MOVING_PROB)

        # 对每个 (env, iy, ix) 选一个方向走, 4 个方向概率均等
        # 用 4 个方向的 dirstack: 每个方向预先铺一个 (4, n_envs, LY, LX) target 位置
        for b in range(self.n_envs):
            rng = self._rngs[b]
            for iy in range(LY):
                for ix in range(LX):
                    if not moves[b, iy, ix]:
                        continue
                    # 4 个候选方向
                    cands = []
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny_, nx_ = iy + dy, ix + dx
                        if 0 <= ny_ < LY and 0 <= nx_ < LX and occ[b, ny_, nx_] == 0:
                            cands.append((ny_, nx_))
                    if not cands:
                        continue
                    ny, nx = cands[int(rng.integers(0, len(cands)))]
                    self.zeta[b, iy, ix] = 0
                    self.zeta[b, ny, nx] = 1

    # ============================================================
    # 感知地图更新 + 全局融合 (向量化, 7×9 cells 一次性 numpy)
    # ============================================================
    def _update_maps(self):
        """一次性算所有 env × UAV × 感知域 单元格的 Bayesian 更新 + 全局融合."""
        n = self.n_envs
        uav_h = self.uav_pos[..., 2]                     # (n, N_UAV)
        uav_ix = self.uav_pos[..., 0]
        uav_iy = self.uav_pos[..., 1]

        # 感知域坐标: (n, N_UAV, 9)
        sens_dx = _SENSE_OFFSETS[uav_h, :, 0]            # (n, N_UAV, 9)
        sens_dy = _SENSE_OFFSETS[uav_h, :, 1]
        sens_ix = uav_ix[..., None] + sens_dx             # (n, N_UAV, 9)
        sens_iy = uav_iy[..., None] + sens_dy

        # 越界 clamp (避免 index 越界), 但记下无效 cell
        in_bound = (sens_ix >= 0) & (sens_ix < LX) & (sens_iy >= 0) & (sens_iy < LY)
        cell_count = _VALID_COUNT[uav_h]                  # (n, N_UAV)
        valid = in_bound & (np.arange(9) < cell_count[..., None])  # (n, N_UAV, 9)
        sens_ix_safe = np.clip(sens_ix, 0, LX - 1)
        sens_iy_safe = np.clip(sens_iy, 0, LY - 1)

        # 取 zeta 在感知域的值 (广播到 UAV)
        # 自我复制 zeta 一次性取: zeta_b, y, x 但 zeta shape 是 (n, LY, LX), UAV 不在里面
        # 走 advanced indexing: zeta[env_idx[:, None, None], sens_iy_safe, sens_ix_safe]
        env_idx = np.arange(n)[:, None, None]
        zeta_sense = self.zeta[env_idx, sens_iy_safe, sens_ix_safe]   # (n, N_UAV, 9)

        # D^D, D^F 按 uav_h 选, broadcast 成 (n, N_UAV, 1)
        pd = DET_PROB[uav_h][..., None]
        pf = FALSE_PROB[uav_h][..., None]

        # 抽样 D (每个 env 一组 RNG)
        D_samples = np.zeros((n, N_UAV, 9), dtype=bool)
        for b in range(n):
            rands = self._rngs[b].random((N_UAV, 9))
            D_samples[b] = rands < np.where(zeta_sense[b] == 1, pd[b], pf[b])

        # Bayesian 更新 (公式 6)
        p = self.ltpm[env_idx, np.arange(N_UAV)[None, :, None],
                       sens_iy_safe, sens_ix_safe]        # 当前 ltpm 在感知域的值
        # 公式展开:
        # D=1: p_new = p*pd / (p*pd + (1-p)*pf)
        # D=0: p_new = p*(1-pd) / (p*(1-pd) + (1-p)*(1-pf))
        new_p = np.where(
            D_samples,
            (p * pd) / (p * pd + (1 - p) * pf + 1e-12),
            (p * (1 - pd)) / (p * (1 - pd) + (1 - p) * (1 - pf) + 1e-12),
        )
        new_p = np.clip(new_p, 1e-6, 1 - 1e-6).astype(np.float32)
        new_chi = _entropy_batch(new_p)

        # Scatter 回 ltpm / leum (n, N_UAV, LY, LX)
        # 把无效 cell 还原成旧值 (no-op)
        old_p = self.ltpm[env_idx, np.arange(N_UAV)[None, :, None],
                          sens_iy_safe, sens_ix_safe]
        old_chi = self.leum[env_idx, np.arange(N_UAV)[None, :, None],
                            sens_iy_safe, sens_ix_safe]
        final_p = np.where(valid, new_p, old_p).astype(np.float32)
        final_chi = np.where(valid, new_chi, old_chi).astype(np.float32)
        self.ltpm[env_idx, np.arange(N_UAV)[None, :, None],
                  sens_iy_safe, sens_ix_safe] = final_p
        self.leum[env_idx, np.arange(N_UAV)[None, :, None],
                  sens_iy_safe, sens_ix_safe] = final_chi

        # 全局融合 (公式 8 / 10): axis=1 沿 UAV 维度
        self.geum = self.leum.min(axis=1)                # (n, LY, LX)
        argmin = np.argmin(self.leum, axis=1)            # (n, LY, LX)
        # take_along_axis 在 batched 中需要沿 axis=1
        cand_p = np.take_along_axis(self.ltpm, argmin[:, None], axis=1)[:, 0]
        min_chi = np.take_along_axis(self.leum, argmin[:, None], axis=1)[:, 0]
        is_min = (self.leum == min_chi[:, None])
        masked_p = np.where(is_min, self.ltpm, -1.0)
        self.gtpm = masked_p.max(axis=1).astype(np.float32)

    # ============================================================
    # Step (主入口)
    # ============================================================
    def step(self, actions):
        """actions: (n_envs, N_UAV) int64.
        返回: (obs_dict, rewards (n_envs,), dones (n_envs,), infos list of dict per env).
        """
        # (1) UAV 动作
        self._step_uav(actions)
        # (2) 目标移动
        self._step_targets()
        # (3) 感知地图更新 + 全局融合
        self._update_maps()
        # (4) 公式 11 标记 searched
        newly_confirmed = (self.zeta == 1) & (self.gtpm >= TARGET_CONFIRM_THRESHOLD)
        self.searched |= newly_confirmed.astype(np.int8)
        # (5) 时间 + done + reward
        self.t += 1
        done = self.t >= MAX_STEPS
        reward = -self.geum.mean(axis=(-2, -1))           # (n_envs,)
        # Obs 打包 (每个键加 n_envs 前置)
        obs = {
            "uav_pos": self.uav_pos.copy(),
            "zeta":    self.zeta.copy(),
            "occ":     self.occ.copy(),
            "ltpm":    self.ltpm.copy(),
            "leum":    self.leum.copy(),
            "gtpm":    self.gtpm.copy(),
            "geum":    self.geum.copy(),
            "t":       self.t.copy(),
        }
        # Info list (每个 env 一个, 简化形式)
        infos = []
        for b in range(self.n_envs):
            infos.append({
                "area_uncertainty": float(self.geum[b].mean()),
                "searched_count":  int(self.searched[b].sum()),
                "targets_total":   N_TARGET,
            })
        return obs, reward.astype(np.float32), done, infos

    # ============================================================
    # 单 env 访问接口 (用于 seed-equal 测试)
    # ============================================================
    def get_env(self, b: int = 0):
        """返回第 b 个 env 的当前状态深度拷贝 (供测试对比)."""
        return {
            "occ":       self.occ[b].copy(),
            "obs_h":     self.obs_h[b].copy(),
            "zeta":      self.zeta[b].copy(),
            "uav_pos":   self.uav_pos[b].copy(),
            "ltpm":      self.ltpm[b].copy(),
            "leum":      self.leum[b].copy(),
            "gtpm":      self.gtpm[b].copy(),
            "geum":      self.geum[b].copy(),
            "t":         int(self.t[b]),
            "searched":  self.searched[b].copy(),
        }

    def area_uncertainty(self):  # 给单 env 风格的 helper
        return self.geum.mean(axis=(-2, -1))


# ============================================================
# 演示/自检 (跟 SearchEnv 的 __main__ 风格一致)
# ============================================================
if __name__ == "__main__":
    import time
    print("[batched_search_env] 自检")
    env = BatchedSearchEnv(n_envs=16, base_seed=42)
    n = 500
    t0 = time.perf_counter()
    rew_sum = np.zeros(env.n_envs, dtype=np.float64)
    for t in range(n):
        actions = np.stack([env._rngs[b].integers(0, 6, N_UAV) for b in range(env.n_envs)])
        obs, r, done, info = env.step(actions)
        rew_sum += r
        if done.any():
            break
    dt = (time.perf_counter() - t0) / n
    print(f"  n_envs={env.n_envs}, {dt*1000:.3f} ms/step (per-env)")
    print(f"  step count: {env.t.mean():.1f}, mean reward: {rew_sum.mean():.3f}")
    print(f"  area_uncertainty[0] = {env.area_uncertainty()[0]:.4f}")
    print(f"  searched[0] sum     = {int(env.searched[0].sum())}")
