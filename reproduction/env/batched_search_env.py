"""
batched_search_env.py —— 多 env 并行的 SearchEnv 版（M8 接力 TODO #4）

为什么需要 batched env？
- 现有 SearchEnv: 0.561 ms/step，其中 _update_maps (7×9 Bayesian) 占 0.441ms = 79%。
- 原因：UAV×感知域嵌套 Python loop → 63 次 scipy-style 标量计算。
- BatchedSearchEnv: 并行保存多个环境状态；感知更新逐个有效格执行，
  从而与单环境保持完全相同的 Bayesian 更新顺序和随机数流。

API 设计（与现有 SearchEnv 完全隔离，仅在 train.py 加 --batch-envs 选项启用）：
- BatchedSearchEnv(n_envs, base_seed) 拥有 n_envs 个独立 SearchEnv 同型副本
- step(actions) actions shape (n_envs, N_UAV=7) → (obs_dict, rewards (n_envs,), dones (n_envs,), infos list)
- 每个 env 独立 RNG (SeedSequence.spawn), n_envs=1 时与 SearchEnv(seed) 完全等价（同 Reset+Action 序列 → 同轨迹）
- obs_dict 每个键加 n_envs 前置：uav_pos (n_envs,7,3), zeta (n_envs,20,20), ...

保留单 env 的 SearchEnv 不变。M8/M7/M5 数据不动。
"""
import numpy as np
from .search_env import (
    LX, LY, AREA, GRID, N_UAV, N_OBSTACLE, N_TARGET, MAX_STEPS,
    TARGET_SPEED, TIME_STEP,
    HEIGHTS, DET_PROB, FALSE_PROB,
    TARGET_CONFIRM_THRESHOLD, MOVING_PROB,
    SENSE_OFFSETS_BY_H, _entropy, DX, DY,
)


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
        self.target_xy_m = np.zeros((n_envs, N_TARGET, 2), dtype=np.float64)
        self.target_heading = np.zeros((n_envs, N_TARGET), dtype=np.float64)
        self.target_cells = np.zeros((n_envs, N_TARGET, 2), dtype=np.int32)
        self.target_found = np.zeros((n_envs, N_TARGET), dtype=bool)

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
        self.target_xy_m[b] = np.asarray(
            [[(ix + 0.5) * GRID, (iy + 0.5) * GRID] for iy, ix in tgt_cells], dtype=np.float64
        )
        self.target_heading[b] = rng.uniform(0.0, 2.0 * np.pi, size=N_TARGET)
        self.target_found[b].fill(False)
        self._refresh_zeta_b(b)
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
        # 每个 episode 必须从未知先验重新开始（算法 3 line 3）。
        h0 = _entropy_batch(np.float32(0.5)).item()
        self.ltpm[b].fill(0.5)
        self.leum[b].fill(h0)
        self.gtpm[b].fill(0.5)
        self.geum[b].fill(h0)

    def _refresh_zeta_b(self, b: int):
        self.zeta[b].fill(0)
        cells = np.floor(self.target_xy_m[b] / GRID).astype(np.int32)
        cells[:, 0] = np.clip(cells[:, 0], 0, LX - 1)
        cells[:, 1] = np.clip(cells[:, 1], 0, LY - 1)
        self.target_cells[b] = cells
        self.zeta[b, cells[:, 1], cells[:, 0]] = 1

    # ============================================================
    # UAV 动作执行
    # ============================================================
    def _step_uav(self, actions):
        """actions shape (n_envs, N_UAV) int64."""
        actions = np.asarray(actions, dtype=np.int64)
        assert actions.shape == (self.n_envs, N_UAV), \
            f"actions shape {actions.shape} != ({self.n_envs}, {N_UAV})"

        if np.any((actions < 0) | (actions >= 6)):
            raise ValueError("actions must be in range 0..5")

        old_pos = self.uav_pos.copy()
        candidate = old_pos.copy()
        # Apply individually legal proposals first. Joint UAV conflicts are
        # resolved from the frozen proposal below.
        for b in range(self.n_envs):
            occupied = {tuple(map(int, p)) for p in old_pos[b].tolist()}
            for n in range(N_UAV):
                ix, iy, h = (int(v) for v in old_pos[b, n])
                a = int(actions[b, n])
                if a < 4:
                    nx, ny = ix + int(DX[a]), iy + int(DY[a])
                    if not (0 <= nx < LX and 0 <= ny < LY):
                        continue
                    if self.occ[b, ny, nx] and h <= self.obs_h[b, ny, nx]:
                        continue
                    if (nx, ny, h) in occupied - {(ix, iy, h)}:
                        continue
                    candidate[b, n, :2] = [nx, ny]
                elif a == 4:
                    if h < len(HEIGHTS) - 1 and (ix, iy, h + 1) not in occupied:
                        candidate[b, n, 2] = h + 1
                else:
                    descend_h = h - 1
                    if h > 0 and not (
                        self.occ[b, iy, ix] and descend_h <= self.obs_h[b, iy, ix]
                    ) and (ix, iy, descend_h) not in occupied:
                        candidate[b, n, 2] = descend_h

        for b in range(self.n_envs):
            proposed = candidate[b].copy()
            blocked = np.zeros(N_UAV, dtype=bool)
            for n in range(N_UAV):
                conflict = any(n != m and np.array_equal(proposed[n], proposed[m]) for m in range(N_UAV))
                swap = any(
                    n != m and np.array_equal(proposed[n], old_pos[b, m])
                    and np.array_equal(proposed[m], old_pos[b, n]) for m in range(N_UAV)
                )
                blocked[n] = conflict or swap
            candidate[b, blocked] = old_pos[b, blocked]

        self.uav_pos[:] = candidate
        for b in range(self.n_envs):
            for ix, iy, _ in self.uav_pos[b]:
                self.t_last[b, iy, ix] = self.t[b]

    # ============================================================
    # 目标移动（向量化，每 env 独立 RNG 抽样）
    # ============================================================
    def _step_targets(self):
        """批量版 1 m/s 连续目标运动；目标身份和数量保持不变。"""
        step = TARGET_SPEED * TIME_STEP
        for b in range(self.n_envs):
            delta = np.column_stack((np.cos(self.target_heading[b]), np.sin(self.target_heading[b]))) * step
            proposed = self.target_xy_m[b] + delta
            occupied = {tuple(map(int, c)) for c in self.target_cells[b].tolist()}
            for k in range(N_TARGET):
                old_cell = tuple(map(int, self.target_cells[b, k]))
                occupied.remove(old_cell)
                x, y = proposed[k]
                reflected = False
                if not (0.0 <= x < AREA):
                    self.target_heading[b, k] = np.pi - self.target_heading[b, k]
                    reflected = True
                if not (0.0 <= y < AREA):
                    self.target_heading[b, k] = -self.target_heading[b, k]
                    reflected = True
                if reflected:
                    d = step * np.array([np.cos(self.target_heading[b, k]), np.sin(self.target_heading[b, k])])
                    proposed[k] = np.clip(self.target_xy_m[b, k] + d, 0.0, np.nextafter(AREA, 0.0))
                ix, iy = np.floor(proposed[k] / GRID).astype(np.int32)
                cell = (int(ix), int(iy))
                if self.occ[b, iy, ix] or cell in occupied:
                    self.target_heading[b, k] = (self.target_heading[b, k] + np.pi) % (2.0 * np.pi)
                    proposed[k] = self.target_xy_m[b, k]
                    cell = old_cell
                occupied.add(cell)
            self.target_xy_m[b] = proposed
            self._refresh_zeta_b(b)

    # ============================================================
    # 感知地图更新 + 全局融合
    # ============================================================
    def _update_maps(self):
        """Mirror SearchEnv's valid-cell order and RNG consumption exactly."""
        n = self.n_envs
        for b in range(n):
            rng = self._rngs[b]
            uav_ids = []
            xs = []
            ys = []
            heights = []
            for uav in range(N_UAV):
                ix, iy, h = (int(v) for v in self.uav_pos[b, uav])
                for dx, dy in SENSE_OFFSETS_BY_H[h]:
                    x, y = ix + dx, iy + dy
                    if 0 <= x < LX and 0 <= y < LY:
                        uav_ids.append(uav)
                        xs.append(x)
                        ys.append(y)
                        heights.append(h)

            uav_ids = np.asarray(uav_ids, dtype=np.intp)
            xs = np.asarray(xs, dtype=np.intp)
            ys = np.asarray(ys, dtype=np.intp)
            heights = np.asarray(heights, dtype=np.intp)
            pd = DET_PROB[heights]
            pf = FALSE_PROB[heights]
            detected = rng.random(len(xs)) < np.where(
                self.zeta[b, ys, xs] == 1, pd, pf
            )
            p = self.ltpm[b, uav_ids, ys, xs]
            p_new = np.where(
                detected,
                (p * pd) / (p * pd + (1.0 - p) * pf),
                (p * (1.0 - pd)) / (
                    p * (1.0 - pd) + (1.0 - p) * (1.0 - pf)
                ),
            )
            p_new = np.clip(p_new, 1e-6, 1.0 - 1e-6)
            self.ltpm[b, uav_ids, ys, xs] = p_new.astype(np.float32)
            self.leum[b, uav_ids, ys, xs] = _entropy_batch(p_new)

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
        current_confirmed = (self.zeta == 1) & (self.gtpm >= TARGET_CONFIRM_THRESHOLD)
        self.searched = current_confirmed.astype(np.int8)
        for b in range(self.n_envs):
            for k, (ix, iy) in enumerate(self.target_cells[b]):
                if current_confirmed[b, iy, ix]:
                    self.target_found[b, k] = True
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
                "found_count": int(self.target_found[b].sum()),
                "success_rate": float(self.target_found[b].mean()),
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
