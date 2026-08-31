import numpy as np
from llm_mappo.utils.config import Config
from llm_mappo.utils.seed import set_seed
from llm_mappo.env.sensor import sensing_radius_grids, detection_prob, false_alarm_prob, sensing_domain
from llm_mappo.env.maps import fuse_global
from llm_mappo.env.energy import propulsive_power

ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0), (0, 0)]  # N,E,S,W,up,down


def _bayes_vec(p, detected, p_d, p_f):
    """Eq.(6) 向量化：对一组网格同时做贝叶斯更新。"""
    num = np.where(detected, p * p_d, p * (1.0 - p_d))
    den = np.where(detected, p * p_d + (1.0 - p) * p_f,
                   p * (1.0 - p_d) + (1.0 - p) * (1.0 - p_f))
    safe = np.where(den > 0, den, 1.0)
    return np.where(den > 0, num / safe, p)


def _entropy_vec(p):
    """Eq.(7) 向量化。"""
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)


class SearchEnv:
    """多 UAV 动态目标搜索环境（Gymnasium 风格接口）。"""

    def __init__(self, cfg: Config | None = None):
        cfg = cfg or Config()
        self.cfg = cfg.env          # EnvConfig
        self.dpes = cfg.dpes        # DPESConfig（T10 信息素使用）
        self.dt = self.cfg.dt

    # ---------- 重置 ----------
    def reset(self, seed=None):
        if seed is not None:
            set_seed(seed)
        self._step = 0
        gs = self.cfg.grid_size
        I = gs * gs
        L = self.cfg.cell_size
        # 目标：连续坐标 (m)，随机方向
        self.target_pos = np.random.rand(self.cfg.n_targets, 2) * (gs * L)
        ang = np.random.rand(self.cfg.n_targets) * 2 * np.pi
        self.target_dir = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        # 障碍物：占用网格集合
        flat = np.random.choice(I, self.cfg.n_obstacles, replace=False)
        self.obstacles = set(flat.tolist())
        # UAV 初始：随机不重叠空闲网格，高度 0
        free = np.array([i for i in range(I) if i not in self.obstacles])
        starts = np.random.choice(free, self.cfg.n_uav, replace=False)
        self.uav_pos = np.array([(s % gs, s // gs, 0) for s in starts])
        self.energy = np.full(self.cfg.n_uav, self.cfg.E_ini)
        # 每 UAV 的 LTPM/LEUM
        self.local_prob = np.full((self.cfg.n_uav, I), 0.5)
        self.local_unc = np.full((self.cfg.n_uav, I), 1.0)
        self.last_visit = np.full((self.cfg.n_uav, I), -self.cfg.max_steps)
        self._update_target_presence()
        return self._obs(), {}

    # ---------- 目标运动 ----------
    def _move_targets(self):
        self.target_pos += self.target_dir * self.cfg.target_speed * self.dt
        limit = self.cfg.grid_size * self.cfg.cell_size
        for k in range(self.cfg.n_targets):
            for d in range(2):
                if self.target_pos[k, d] < 0 or self.target_pos[k, d] > limit:
                    self.target_dir[k, d] *= -1
                    self.target_pos[k, d] = np.clip(self.target_pos[k, d], 0, limit)

    def _update_target_presence(self):
        gs = self.cfg.grid_size
        L = self.cfg.cell_size
        I = gs * gs
        presence = np.zeros(I, dtype=bool)
        for k in range(self.cfg.n_targets):
            gx = int(self.target_pos[k, 0] // L)
            gy = int(self.target_pos[k, 1] // L)
            if 0 <= gx < gs and 0 <= gy < gs:
                presence[gy * gs + gx] = True
        self.target_presence = presence

    def _target_in_grid(self, i: int) -> bool:
        return bool(self.target_presence[i])

    # ---------- 传感器扫描 ----------
    def _sensor_scan(self):
        gs = self.cfg.grid_size
        for n in range(self.cfg.n_uav):
            alt = int(self.uav_pos[n, 2])
            r = sensing_radius_grids(alt)
            p_d = detection_prob(alt)
            p_f = false_alarm_prob(alt)
            dom = sensing_domain(tuple(self.uav_pos[n]), r, gs)
            if not dom:
                continue
            idx = np.array([y * gs + x for x, y in dom])
            self.last_visit[n, idx] = self._step
            truth = self.target_presence[idx]
            probs = np.where(truth, p_d, p_f)
            detected = np.random.rand(len(idx)) < probs
            self.local_prob[n, idx] = _bayes_vec(self.local_prob[n, idx], detected, p_d, p_f)
            self.local_unc[n, idx] = _entropy_vec(self.local_prob[n, idx])

    # ---------- 能耗 ----------
    def _update_energy(self):
        for n in range(self.cfg.n_uav):
            self.energy[n] -= propulsive_power(self.cfg.uav_speed) * self.dt

    # ---------- 步进 ----------
    def step(self, actions: np.ndarray):
        gs = self.cfg.grid_size
        self._move_targets()
        for n, a in enumerate(actions):
            a = int(a)
            if a < 4:
                dx, dy = ACTIONS[a]
                self.uav_pos[n, 0] = np.clip(self.uav_pos[n, 0] + dx, 0, gs - 1)
                self.uav_pos[n, 1] = np.clip(self.uav_pos[n, 1] + dy, 0, gs - 1)
            elif a == 4:
                self.uav_pos[n, 2] = min(2, self.uav_pos[n, 2] + 1)
            elif a == 5:
                self.uav_pos[n, 2] = max(0, self.uav_pos[n, 2] - 1)
        self._update_target_presence()
        self._sensor_scan()
        self._update_energy()
        self._step += 1
        term = self._step >= self.cfg.max_steps
        trunc = bool((self.energy < propulsive_power(self.cfg.uav_speed) * self.dt).any())
        return self._obs(), np.zeros(self.cfg.n_uav), term, trunc, {}

    # ---------- 观测 / 指标 ----------
    def _obs(self):
        return [np.concatenate([self.local_prob[n], self.local_unc[n]])
                for n in range(self.cfg.n_uav)]

    def _searched_targets(self) -> float:
        """Eq.(11)：目标被成功搜索 = 存在且全局概率 >= xi。"""
        _, fused_prob = fuse_global(self.local_unc, self.local_prob)
        return float(np.sum(self.target_presence & (fused_prob >= self.cfg.xi)))

    def _avg_uncertainty(self) -> float:
        """Eq.(9)：区域平均不确定度。"""
        fused_unc, _ = fuse_global(self.local_unc, self.local_prob)
        return float(fused_unc.mean())

    # ---------- 动作掩码 ----------
    def action_masks(self) -> np.ndarray:
        gs = self.cfg.grid_size
        masks = np.ones((self.cfg.n_uav, 6))
        for n in range(self.cfg.n_uav):
            x, y, alt = self.uav_pos[n]
            if x == 0:
                masks[n, 3] = 0          # 西
            if x == gs - 1:
                masks[n, 1] = 0          # 东
            if y == 0:
                masks[n, 2] = 0          # 南
            if y == gs - 1:
                masks[n, 0] = 0          # 北
            if alt == 2:
                masks[n, 4] = 0          # 升
            if alt == 0:
                masks[n, 5] = 0          # 降
        return masks
