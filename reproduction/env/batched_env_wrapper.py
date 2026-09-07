"""
batched_env_wrapper.py —— BatchedSearchEnv 的多智能体接口 (M8 接力 TODO #4)

跟 env/env_wrapper.py (Single env version) 的关系:
- 同接口风格: reset() / step(actions) / get_global_state()
- 区别: 所有输出加 n_envs 前置
  - obs:    (n_envs, N_UAV, OBS_DIM)  float32
  - reward: (n_envs,) shared reward (formula 25)
  - done:   (n_envs,) bool
  - per_agent_reward: (n_envs, N_UAV) per-agent dense reward
- 内部 obs 构造 (5x5 patch × 7 UAV) 用 np.ix_ 一次性算完
- manual_reward 路径逐 env 调 (M2/M3 兼容); LRS 路径 (M5) 见 _lrs_per_env

M2/M3 接入示例 (在 train.py):
    env = BatchedMultiAgentWrapper(n_envs=16, base_seed=seed, use_dpes=True, lrs_reward_fn=None)
    obs = env.reset()                                  # (16, 7, 85)
    actions = actor(obs.reshape(16*7, 85))              # (16*7, 6) probs
    actions = actions.argmax(-1).reshape(16, 7).astype(np.int64)
    obs, env_rew, done, per_agent_rew, infos = env.step(actions)
"""
import numpy as np

from .search_env import (LX, LY, N_UAV, HEIGHTS, E_INITIAL, TIME_STEP,
                         UAV_SPEED, propulsion_power)
from .batched_search_env import BatchedSearchEnv
from .env_wrapper import _nearest_obstacle, _nearest_uav, SENSE_OFFSETS_BY_H
from ..reward.manual_reward import compute_manual_reward


class BatchedMultiAgentWrapper:
    """多 env 并行的 MAPPO 接口. 7 UAV × n_envs 一次性 batched."""

    PATCH = 5
    HALF = 2
    OBS_DIM_NO_DPES = N_UAV * 3 + 25 + 25  # 71
    OBS_DIM = OBS_DIM_NO_DPES + 25          # 96

    def __init__(self, n_envs: int, base_seed: int = 0, use_dpes: bool = True,
                 lrs_reward_fn=None, use_paper_reward: bool = False):
        self.n_envs = n_envs
        self.use_dpes = use_dpes
        self.env = BatchedSearchEnv(n_envs, base_seed)
        # DPES 信息素: 用 BatchedPheromoneMap 一次性维护 (比 list-of-PheromoneMap 快 ~10×)
        if use_dpes:
            from ..algorithms.batched_dpes import BatchedPheromoneMap
            self.dpes = BatchedPheromoneMap(n_envs)
        else:
            self.dpes = None
        self.energy = np.full((n_envs, N_UAV), 1.0, dtype=np.float32)
        self.prev_au = np.zeros(n_envs, dtype=np.float32)
        self.prev_searched = np.zeros((n_envs, LY, LX), dtype=np.int8)
        self.lrs_reward_fn = lrs_reward_fn
        self.use_paper_reward = use_paper_reward
        # Step counter for LRS 的 prev_au
        self._t = np.zeros(n_envs, dtype=np.int32)

    # ============================================================
    # Reset
    # ============================================================
    def reset(self) -> np.ndarray:
        """重置所有 env, 返回初始 obs (n_envs, N_UAV, OBS_DIM)."""
        self.env.reset()
        self.energy[:] = 1.0
        self._t[:] = 0
        for b in range(self.n_envs):
            self.prev_au[b] = float(self.env.area_uncertainty()[b])
            self.prev_searched[b] = self.env.searched[b].copy()
        if self.dpes is not None:
            self.dpes.reset()
        return self._obs_batch()                       # (n_envs, N_UAV, 85 or 60)

    # ============================================================
    # Step
    # ============================================================
    def step(self, actions: np.ndarray):
        """actions shape (n_envs, N_UAV) int64.
        Returns:
            obs              : (n_envs, N_UAV, OBS_DIM) float32
            shared_reward    : (n_envs,) float32  formula 25
            done             : (n_envs,) bool
            per_agent_reward : (n_envs, N_UAV) float32
            infos            : list of dict per env
        """
        actions = np.asarray(actions, dtype=np.int64)
        assert actions.shape == (self.n_envs, N_UAV), \
            f"actions shape {actions.shape} != ({self.n_envs}, {N_UAV})"

        prev_geum = self.env.geum.copy()
        # (1) 推进底层 BatchedSearchEnv
        env_obs, env_shared_rew, done, infos = self.env.step(actions)
        self._t += 1

        # (1.5) DPES 更新 (batched, 用 BatchedPheromoneMap 一次性更新所有 env)
        if self.dpes is not None:
            self.dpes.update(self.env)

        # (2) Eq. (1)–(3) 推进能耗，energy 保存为 E_n/E_ini。
        delta = propulsion_power(UAV_SPEED) * TIME_STEP / E_INITIAL
        self.energy[:] = np.maximum(0.0, self.energy - delta)

        # (3) 计算稠密奖励 (M2/M3 走 manual_reward, M5 走 lrs_reward_fn)
        cur_au = self.env.area_uncertainty()             # (n_envs,)
        cur_searched = self.env.searched                  # (n_envs, LY, LX)
        newly = cur_searched.astype(bool) & ~self.prev_searched.astype(bool)  # (n_envs, LY, LX) bool
        newly_count = newly.sum(axis=(1, 2))              # (n_envs,)

        per_agent_reward = np.zeros((self.n_envs, N_UAV), dtype=np.float32)
        shared_reward = np.zeros(self.n_envs, dtype=np.float32)

        # 逐 env 调用 manual_reward (保留调用兼容性, manual_reward 不接受 batch 数组)
        # 时间复杂度: O(n_envs) 而不是 O(n_envs×N_UAV), 因为 manual_reward 内部已经 loop 7 UAV
        for b in range(self.n_envs):
            if self.use_paper_reward:
                from ..reward.paper_reward import compute_paper_rbest
                shared_b, components = compute_paper_rbest(prev_geum[b], self.env_for_b(b), actions[b])
                per_b = np.full(N_UAV, shared_b, dtype=np.float32)
                infos[b]["paper_reward_components"] = components
            elif self.lrs_reward_fn is not None:
                # M5 路径 (LRS): per-UAV 调一次
                shared_b, per_b = self._lrs_per_env(b, actions[b])
            else:
                # M2/M3 路径 (manual_reward)
                # 单 env 接口接受 (env, actions, energy, prev_au, cur_au, newly_confirmed)
                # BatchedSearchEnv 没 occ 等所有 SearchEnv 字段, 但 manual_reward 用到了 .occ
                # 解法: 传 self.env, manual_reward 内部读 .occ/.uav_pos/.zeta/.gtpm/.geum 都从 batched 取 [b]
                shared_b, per_b = compute_manual_reward(
                    env=self.env_for_b(b),
                    actions=actions[b],
                    energy=self.energy[b].copy(),
                    prev_au=float(self.prev_au[b]),
                    cur_au=float(cur_au[b]),
                    newly_confirmed=newly[b],
                )
            per_agent_reward[b] = per_b
            shared_reward[b] = shared_b

        # (4) 更新 prev
        self.prev_au[:] = cur_au
        self.prev_searched[:] = cur_searched

        # (5) 收集新 obs (推进后)
        obs = self._obs_batch()

        # (6) 把 newly_confirmed_count 写进 info
        for b in range(self.n_envs):
            infos[b]["per_agent_reward"] = per_agent_reward[b]
            infos[b]["newly_confirmed_count"] = int(newly_count[b])
        return obs, shared_reward, done, per_agent_reward, infos

    def get_action_masks(self) -> np.ndarray:
        masks = np.ones((self.n_envs, N_UAV, 6), dtype=bool)
        for b in range(self.n_envs):
            pos = self.env.uav_pos[b]
            occupied = {tuple(map(int, p)) for p in pos.tolist()}
            for n, (ix, iy, h) in enumerate(pos):
                for a, (dx, dy) in enumerate(((0, -1), (1, 0), (0, 1), (-1, 0))):
                    nx, ny = int(ix + dx), int(iy + dy)
                    if not (0 <= nx < LX and 0 <= ny < LY):
                        masks[b, n, a] = False
                    elif self.env.occ[b, ny, nx] and h <= self.env.obs_h[b, ny, nx]:
                        masks[b, n, a] = False
                    elif (nx, ny, int(h)) in occupied - {(int(ix), int(iy), int(h))}:
                        masks[b, n, a] = False
                masks[b, n, 4] = h < len(HEIGHTS) - 1 and (int(ix), int(iy), int(h + 1)) not in occupied
                descend_h = int(h - 1)
                masks[b, n, 5] = (
                    h > 0
                    and not (
                        self.env.occ[b, int(iy), int(ix)]
                        and descend_h <= self.env.obs_h[b, int(iy), int(ix)]
                    )
                    and (int(ix), int(iy), descend_h) not in occupied
                )
        return masks

    # ============================================================
    # 单 env view: 给 manual_reward 当 SearchEnv-like 接口
    # ============================================================
    def env_for_b(self, b: int) -> "_SingleEnvView":
        """返回第 b 个 env 的 view, 字段同步于 BatchedSearchEnv[b].

        关键点: manual_reward 读 .occ, .uav_pos, .zeta, .gtpm, .geum, .searched,
        .t_last_visit, .t, .area_uncertainty().
        """
        return _SingleEnvView(self.env, b)

    # ============================================================
    # Partial reset (某个 env done 时, 只重置那个 env)
    # ============================================================
    def partial_reset(self, b: int):
        """只重置 env b 的内部状态 (其他 env 继续推进).
        暴露给 train_batched.py 在某个 env done 时调用.
        """
        self.env._reset_env_b(b)
        self.energy[b, :] = 1.0
        self.prev_au[b] = float(self.env.area_uncertainty()[b])
        self.prev_searched[b] = self.env.searched[b].copy()
        self._t[b] = 0
        # BatchedPheromoneMap: 仅清零 env b 对应的那一张 dp 切片
        if self.dpes is not None:
            self.dpes.dp[b] = 0.0

    # ============================================================
    # LRS 路径 (per-env 调 lrs_reward_fn)
    # ============================================================
    SAFETY_W = {
        "collision":   -2.0,
        "boundary":    -1.0,
        "height_chg": -0.05,
        "move":        -0.02,
        "energy_low":  -0.5,
    }

    def _lrs_per_env(self, b: int, actions_b: np.ndarray):
        """LRS + 安全 per env. Signature 同 MultiAgentWrapper._lrs_path_reward."""
        DX = np.array([0, 1, 0, -1])
        DY = np.array([-1, 0, 1, 0])
        env_view = self.env_for_b(b)
        occ = env_view.occ
        uav_pos = env_view.uav_pos
        prev_au = float(self.prev_au[b])
        per = np.zeros(N_UAV, dtype=np.float32)
        actions_b = np.asarray(actions_b, dtype=np.int64)
        for n in range(N_UAV):
            ix, iy, h = uav_pos[n]
            a = int(actions_b[n])
            per[n] += float(self.lrs_reward_fn(env_view, n, a, prev_au))
            if 0 <= a <= 3:
                nx, ny = ix + int(DX[a]), iy + int(DY[a])
                if not (0 <= nx < LX and 0 <= ny < LY and occ[ny, nx] == 0):
                    per[n] += self.SAFETY_W["collision"]
                else:
                    per[n] += self.SAFETY_W["move"]
            elif (a == 4 and h >= len(HEIGHTS) - 1) or (a == 5 and h <= 0):
                per[n] += self.SAFETY_W["boundary"]
            elif a in (4, 5):
                per[n] += self.SAFETY_W["height_chg"]
            if self.energy[b, n] < 0.05:
                per[n] += self.SAFETY_W["energy_low"]
        shared = float(per.mean())
        return shared, per

    # ============================================================
    # DPES update single env (借用 PheromoneMap 单 env 接口)
    # ============================================================
    def _dpes_update_single(self, b: int):
        """PheromoneMap.update() 读 env.t / .t_last_visit / .gtpm / .uav_pos, 给 _SingleEnvView 即可."""
        self.dpes[b].update(self.env_for_b(b))

    # ============================================================
    # Batched observation 构造 (核心加速: 7 UAV × 5x5 patch 一次性算)
    # ============================================================
    def _obs_batch(self) -> np.ndarray:
        """返回 (n_envs, N_UAV, OBS_DIM) 的 np.ndarray."""
        n = self.n_envs
        env = self.env
        uav_pos = env.uav_pos     # (n, N_UAV, 3)
        zeta = env.zeta           # (n, LY, LX)
        occ = env.occ
        geum = env.geum           # (n, LY, LX)
        gtpm = env.gtpm

        uav_ix = uav_pos[..., 0]  # (n, N_UAV)
        uav_iy = uav_pos[..., 1]
        uav_h = uav_pos[..., 2]

        # (1) 公式 (18) 的 p(t)：每个 Actor 都接收所有 UAV 位置。
        all_pos = uav_pos.astype(np.float32).copy()
        all_pos[..., 0] /= (LX - 1)
        all_pos[..., 1] /= (LY - 1)
        all_pos[..., 2] /= max(1, len(HEIGHTS) - 1)
        all_pos = np.repeat(all_pos.reshape(n, 1, N_UAV * 3), N_UAV, axis=1)

        # (2)(3) 5x5 patch: geum (25) + zeta (25)
        # 用 numpy 一次性 broadcast scatter (不 Python loop)
        geum_patches = np.zeros((n, N_UAV, self.PATCH, self.PATCH), dtype=np.float32)
        es_patches = np.zeros((n, N_UAV, self.PATCH, self.PATCH), dtype=np.float32)
        for h_level in range(3):
            mask = (uav_h == h_level)        # (n, N_UAV)
            for ddy, ddx in SENSE_OFFSETS_BY_H[h_level]:
                py = self.HALF + ddy
                px = self.HALF + ddx
                tgt_y = uav_iy + ddy          # (n, N_UAV)
                tgt_x = uav_ix + ddx
                valid = mask & (tgt_y >= 0) & (tgt_y < LY) & (tgt_x >= 0) & (tgt_x < LX)
                # 取 zeta/geum 在 (tgt_y, tgt_x), 越界 clamp 避免 index 错
                src_y = np.clip(tgt_y, 0, LY - 1)
                src_x = np.clip(tgt_x, 0, LX - 1)
                env_idx = np.arange(n)
                z_at = zeta[env_idx[:, None], src_y, src_x]
                o_at = occ[env_idx[:, None], src_y, src_x]
                g_at = geum[env_idx[:, None], src_y, src_x]
                # 一次性 broadcast scatter (无效位置写 0)
                es_at = np.where(z_at == 1, 1.0, np.where(o_at == 1, -1.0, 0.0))
                es_patches[..., py, px] = np.where(valid, es_at, 0.0)
                geum_patches[..., py, px] = np.where(valid, g_at, 0.0)
        geum_flat = geum_patches.reshape(n, N_UAV, -1)
        es_flat = es_patches.reshape(n, N_UAV, -1)

        # 拼接 (有/无 DPES)
        if self.use_dpes:
            # (7) DPES 信息素 patch — 向量化按 UAV 取一个 (5,5) block
            dp_patches = np.zeros((n, N_UAV, self.PATCH, self.PATCH), dtype=np.float32)
            if self.dpes is not None:
                # BatchedPheromoneMap: dp (n_envs, LY, LX) 一次性取所有 env 的 patch
                dp_field = self.dpes.dp  # (n, LY, LX)
                for ddy in range(-self.HALF, self.HALF + 1):
                    for ddx in range(-self.HALF, self.HALF + 1):
                        py = self.HALF + ddy
                        px = self.HALF + ddx
                        tgt_y = uav_iy + ddy
                        tgt_x = uav_ix + ddx
                        # DP_n(t) 只包含当前高度的 sensing domain。
                        sense_valid = np.zeros_like(uav_h, dtype=bool)
                        for h_level in range(3):
                            if (ddy, ddx) in SENSE_OFFSETS_BY_H[h_level]:
                                sense_valid |= (uav_h == h_level)
                        valid = sense_valid & (tgt_y >= 0) & (tgt_y < LY) & (tgt_x >= 0) & (tgt_x < LX)
                        src_y = np.clip(tgt_y, 0, LY - 1)
                        src_x = np.clip(tgt_x, 0, LX - 1)
                        env_idx = np.arange(n)
                        dp_at = dp_field[env_idx[:, None], src_y, src_x]  # (n, N_UAV)
                        dp_patches[..., py, px] = np.where(valid, dp_at, 0.0)
            dp_flat = dp_patches.reshape(n, N_UAV, -1)
            obs = np.concatenate([all_pos, geum_flat, es_flat, dp_flat], axis=-1)
        else:
            obs = np.concatenate([all_pos, geum_flat, es_flat], axis=-1)
        return obs.astype(np.float32)

    def _nearest_obstacle_batch(self, uav_ix, uav_iy, occ):
        """对每 (env, uav), 找最近障碍. occ (n, LY, LX)."""
        n = self.n_envs
        # 全局障碍坐标 (因为每个 env 不一样, 只能 per-env 算)
        out = np.zeros((n, N_UAV, 3), dtype=np.float32)
        for b in range(n):
            ys, xs = np.where(occ[b] == 1)
            if len(ys) == 0:
                out[b, :, 0] = 0.0
                out[b, :, 1] = 0.0
                out[b, :, 2] = 1.0
                continue
            dx_arr = xs[None, :] - uav_ix[b][:, None]   # (N_UAV, n_obs)
            dy_arr = ys[None, :] - uav_iy[b][:, None]
            d_arr = np.maximum(np.abs(dx_arr), np.abs(dy_arr))
            idx = d_arr.argmin(axis=1)                  # (N_UAV,)
            out[b, :, 0] = dx_arr[np.arange(N_UAV), idx] / max(LX, LY)
            out[b, :, 1] = dy_arr[np.arange(N_UAV), idx] / max(LX, LY)
            out[b, :, 2] = np.minimum(d_arr[np.arange(N_UAV), idx], max(LX, LY)) / max(LX, LY)
        return out

    def _nearest_uav_batch(self, uav_ix, uav_iy, uav_h, uav_pos):
        """对每 (env, uav), 找最近的其他 UAV.
        uav_pos (n, N_UAV, 3)."""
        n = self.n_envs
        max_dim = max(LX, LY)
        out = np.zeros((n, N_UAV, 3), dtype=np.float32)
        for b in range(n):
            for n_idx in range(N_UAV):
                self_ix = uav_ix[b, n_idx]
                self_iy = uav_iy[b, n_idx]
                self_h = uav_h[b, n_idx]
                # 排除自己
                other_ix = np.delete(uav_pos[b, :, 0], n_idx)
                other_iy = np.delete(uav_pos[b, :, 1], n_idx)
                other_h = np.delete(uav_pos[b, :, 2], n_idx)
                if len(other_ix) == 0:
                    out[b, n_idx] = [0, 0, 0]
                    continue
                dx_arr = other_ix - self_ix
                dy_arr = other_iy - self_iy
                dh_arr = other_h - self_h
                d_arr = np.maximum(np.abs(dx_arr), np.abs(dy_arr))
                idx = int(d_arr.argmin())
                out[b, n_idx, 0] = dx_arr[idx] / max_dim
                out[b, n_idx, 1] = dy_arr[idx] / max_dim
                out[b, n_idx, 2] = dh_arr[idx] / max(1, len(HEIGHTS) - 1)
        return out

    # ============================================================
    # 给 Critic 用的全局 state (公式 25): 把所有 env × UAV 的 obs 拼起来
    # ============================================================
    def get_global_state(self) -> np.ndarray:
        """(n_envs, N_UAV * OBS_DIM) 一个 env 看全部 7 UAV 的拼接."""
        obs = self._obs_batch()  # (n, N_UAV, OBS_DIM)
        return obs.reshape(self.n_envs, N_UAV * obs.shape[-1]).astype(np.float32)


class _SingleEnvView:
    """BatchedSearchEnv[b] 的伪单 env view, 让 manual_reward/PheromoneMap 等单 env 接口能读.

    仅暴露 manual_reward.update / .occ / .uav_pos / .zeta / .gtpm / .geum / .searched /
    .t / .t_last_visit / .area_uncertainty() 这些字段/方法.
    """
    def __init__(self, batched: BatchedSearchEnv, b: int):
        self._b = b
        self._batch = batched
        # 直接绑引用 (manual_reward 不会修改这些)
        self.uav_pos = batched.uav_pos[b]
        self.zeta = batched.zeta[b]
        self.occ = batched.occ[b]
        self.gtpm = batched.gtpm[b]
        self.geum = batched.geum[b]
        self.searched = batched.searched[b]
        self.t_last_visit = batched.t_last[b]
        self.t = int(batched.t[b])
        self.obs_h = batched.obs_h[b]
        self.ltpm = batched.ltpm[b]
        self.leum = batched.leum[b]

    def area_uncertainty(self):
        return float(self._batch.area_uncertainty()[self._b])
