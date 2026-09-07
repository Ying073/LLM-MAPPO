"""
search_env.py —— M1 里程碑完整版 (v1 稳定版, 保留 numpy 实现)

复现对象：LLM-MAPPO 论文 §III 系统模型 (论文 paper.md S016–S026, 公式 4–11)。

关于 GPU 化的取舍 (M6 经验):
- 本环境的 7×9 贝叶斯更新 + 400 格的 Python 循环, 整个 500 步在 CPU 上仅 1.99ms/step
- 曾尝试把 _update_maps / _step_targets 整体搬到 torch tensor (v2), 测得 3.12ms/step (CUDA)
- GPU launch overhead 远大于计算本身, 反而比 CPU 慢
- 真正 GPU 化收益在 PPO update (mappo.py, mappo.update() 的 batched matmul)
- 结论: 搜索环境保持 numpy, GPU 仅给 actor/critic 前向+反向用

所有"为什么这么写"都在函数 docstring 的【对应论文】里。
跑通的标志: python search_env.py 弹出一张俯视图。
"""

import numpy as np


# ============================================================
# 一、几何 / 运动参数 (对应论文 §III 段落 S016, 表 I, 公式 4)
# ============================================================
LX = 20                                # 网格列数 (= L_X)
LY = 20                                # 网格行数 (= L_Y)
AREA = 2000.0                          # 区域边长 X = Y = 2000 m
GRID = AREA / LX                       # 每格边长 L = 100 m

N_UAV = 7                              # UAV 数量 N
N_OBSTACLE = 20                        # 静态障碍物数量
N_TARGET = 15                          # 动态目标数量
TARGET_SPEED = 1.0                     # 目标速度 1 m/s
UAV_SPEED = 10.0                       # UAV 速度 10 m/s
TIME_STEP = 1.0                        # 表 I: 每个 time step 为 1 s
E_INITIAL = 6.0e4                      # 表 I: UAV 初始能量 6e4 J

# Eq. (1) 推进功率参数（论文实验设置）。
P_B = 79.86
P_D = 88.63
U_TIP = 120.0
V_0 = 4.03
FUSELAGE_DRAG = 0.6
AIR_DENSITY = 1.225
ROTOR_SOLIDITY = 0.05
ROTOR_DISK_AREA = 0.503
MAX_STEPS = 500                        # 一个 episode 的时间步数 T

# 表 I: 高度 -> 感知域 / 检测概率 / 虚警概率 (公式 4 / 公式 12g–i)
HEIGHTS = np.array([50.0, 100.0, 150.0])          # 三档高度 (m)
DET_PROB = np.array([0.9, 0.8, 0.7])              # P^D, 公式 5
FALSE_PROB = np.array([0.1, 0.2, 0.3])            # P^F, 公式 5
SENSE_SIZE = np.array([1, 5, 9])                  # 感知域网格数 (1/5/9)

# 水平 4 方向向量, 下标 0–3 对应动作 0–3
DX = np.array([0, 1, 0, -1])      # 北(0) 东(1) 南(2) 西(3): 列方向变化
DY = np.array([-1, 0, 1, 0])      # 屏幕 y 轴向下, "北" 是 iy-1

# 公式 11 中的目标确认阈值 ξ (paper S025)
TARGET_CONFIRM_THRESHOLD = 0.8

# 旧实现用“每步 0.5 概率跳一个 100 m 网格”，等效速度约 50 m/s，
# 与表 I 的 1 m/s 不一致。保留名字仅兼容旧导入；新实现使用连续坐标匀速运动。
MOVING_PROB = 1.0


# ============================================================
# 预计算常量
# ============================================================
SENSE_OFFSETS_BY_H = {
    0: [(0, 0)],                                          # 50 m
    1: [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],        # 100 m (5 格十字)
    2: [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)],  # 150 m (3x3)
}


# ============================================================
# 辅助函数
# ============================================================
def _entropy(p):
    """信息熵, 对应论文 公式 7:
        χ = -p·log2(p) - (1-p)·log2(1-p)
    p=0 或 1 时取 0 (避免 log(0) 报错).
    """
    p = np.asarray(p, dtype=np.float64)
    out = np.zeros_like(p)
    mask = (p > 1e-9) & (p < 1 - 1e-9)
    out[mask] = -(p[mask] * np.log2(p[mask]) + (1 - p[mask]) * np.log2(1 - p[mask]))
    return out.astype(np.float32)


def propulsion_power(speed: float) -> float:
    """论文 Eq. (1) 的多旋翼推进功率 P(v)，单位 W。"""
    v = float(speed)
    profile = P_B * (1.0 + 3.0 * v * v / (U_TIP * U_TIP))
    induced_inner = np.sqrt(1.0 + v ** 4 / (4.0 * V_0 ** 4)) - v * v / (2.0 * V_0 * V_0)
    induced = P_D * np.sqrt(max(0.0, induced_inner))
    parasite = 0.5 * FUSELAGE_DRAG * AIR_DENSITY * ROTOR_SOLIDITY * ROTOR_DISK_AREA * v ** 3
    return float(profile + induced + parasite)


# ============================================================
# 主环境类
# ============================================================
class SearchEnv:
    """多 UAV 动态目标搜索环境 (numpy 纯 CPU 实现).

    状态 (numpy 数组, 与 wrapper / lrs.py 接口兼容):
        uav_pos : (N_UAV, 3)               每架 UAV 的 (ix, iy, h)
        zeta    : (LY, LX)   int           目标存在状态 0/1
        occ     : (LY, LX)   int           障碍占用 0/1
        obs_h   : (LY, LX)   int           障碍物高度档位
        ltpm    : (N_UAV, LY, LX) float    每架 UAV 的局部目标概率 p
        leum    : (N_UAV, LY, LX) float    每架 UAV 的局部不确定度 χ
        gtpm    : (LY, LX)   float         全局目标概率
        geum    : (LY, LX)   float         全局不确定度
        t_last_visit : (LY, LX) int        每个网格"上次被访问"的时间步
        searched : (LY, LX) int            该网格是否曾被确认存在目标

    device 参数 (M6): 仅作为记录, 不影响行为; 实际计算都在 CPU numpy.
    GPU 加速留给 PPO update (mappo.py).
    """

    def __init__(self, seed: int = 0, device: str = "cpu"):
        self.device = device  # 仅记录, 实际计算 CPU
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ============================================================
    # 二、初始化 (对应论文 S016+S019)
    # ============================================================
    def reset(self):
        """重置一局. 顺序: 先布障碍 → 排除障碍格后布目标 → 再排除两者后布 UAV."""
        # (a) 随机选 N_OBSTACLE 个不重复网格放障碍物; 随机高度档位
        all_cells = [(iy, ix) for iy in range(LY) for ix in range(LX)]
        self.rng.shuffle(all_cells)
        occ_cells = all_cells[:N_OBSTACLE]
        self.occ = np.zeros((LY, LX), dtype=np.int8)
        self.obs_h = np.zeros((LY, LX), dtype=np.int8)
        for iy, ix in occ_cells:
            self.occ[iy, ix] = 1
            self.obs_h[iy, ix] = int(self.rng.integers(0, len(HEIGHTS)))

        # (b) 在剩下的格子里随机选 N_TARGET 个放目标
        free_cells = [(iy, ix) for iy, ix in all_cells[N_OBSTACLE:]]
        self.rng.shuffle(free_cells)
        tgt_cells = free_cells[:N_TARGET]
        # 每个目标保留独立身份和连续米制坐标。随机方向在一个 episode 内保持，
        # 每步位移 TARGET_SPEED * TIME_STEP；遇边界/障碍时反射。
        self.target_xy_m = np.asarray(
            [[(ix + 0.5) * GRID, (iy + 0.5) * GRID] for iy, ix in tgt_cells],
            dtype=np.float64,
        )
        self.target_heading = self.rng.uniform(0.0, 2.0 * np.pi, size=N_TARGET)
        self.target_found = np.zeros(N_TARGET, dtype=bool)
        self.target_first_found_step = np.full(N_TARGET, -1, dtype=np.int32)
        self.zeta = np.zeros((LY, LX), dtype=np.int8)
        self._refresh_zeta()

        # (c) 再从剩余格子里随机选 N_UAV 个, 初始高度档 h=1 (100 m)
        remaining = free_cells[N_TARGET:]
        self.rng.shuffle(remaining)
        uav_cells = remaining[:N_UAV]
        self.uav_pos = np.zeros((N_UAV, 3), dtype=np.int32)
        for n, (iy, ix) in enumerate(uav_cells):
            self.uav_pos[n] = [ix, iy, 1]

        # (d) 初始化 LTPM/LEUM/GTPM/GEUM (p=0.5 时不确定度最大)
        H0 = _entropy(0.5)
        self.ltpm = np.full((N_UAV, LY, LX), 0.5, dtype=np.float32)
        self.leum = np.full((N_UAV, LY, LX), H0, dtype=np.float32)
        self.gtpm = np.full((LY, LX), 0.5, dtype=np.float32)
        self.geum = np.full((LY, LX), H0, dtype=np.float32)

        # (e) 辅助变量
        self.t_last_visit = np.zeros((LY, LX), dtype=np.int32)
        self.searched = np.zeros((LY, LX), dtype=np.int8)
        self.cumulative_searched = 0
        self.t = 0

        return self._get_obs()

    def _refresh_zeta(self):
        """从固定数量的目标身份重建网格存在状态 ζ_i(t)。"""
        self.zeta.fill(0)
        cells = np.floor(self.target_xy_m / GRID).astype(np.int32)
        cells[:, 0] = np.clip(cells[:, 0], 0, LX - 1)
        cells[:, 1] = np.clip(cells[:, 1], 0, LY - 1)
        self.target_cells = cells
        self.zeta[cells[:, 1], cells[:, 0]] = 1

    def _step_targets(self):
        """按表 I 的 1 m/s 匀速移动目标，并保持目标数量与身份不变。"""
        step = TARGET_SPEED * TIME_STEP
        delta = np.column_stack((np.cos(self.target_heading), np.sin(self.target_heading))) * step
        proposed = self.target_xy_m + delta
        occupied_cells = {tuple(map(int, cell)) for cell in self.target_cells.tolist()}
        for k in range(N_TARGET):
            old_cell = tuple(map(int, self.target_cells[k]))
            occupied_cells.remove(old_cell)
            x, y = proposed[k]
            reflected = False
            if not (0.0 <= x < AREA):
                self.target_heading[k] = np.pi - self.target_heading[k]
                reflected = True
            if not (0.0 <= y < AREA):
                self.target_heading[k] = -self.target_heading[k]
                reflected = True
            if reflected:
                delta_k = step * np.array([np.cos(self.target_heading[k]), np.sin(self.target_heading[k])])
                proposed[k] = np.clip(self.target_xy_m[k] + delta_k, 0.0, np.nextafter(AREA, 0.0))
            ix, iy = np.floor(proposed[k] / GRID).astype(np.int32)
            cell = (int(ix), int(iy))
            if self.occ[iy, ix] or cell in occupied_cells:
                self.target_heading[k] = (self.target_heading[k] + np.pi) % (2.0 * np.pi)
                proposed[k] = self.target_xy_m[k]
                ix, iy = np.floor(proposed[k] / GRID).astype(np.int32)
                cell = (int(ix), int(iy))
            occupied_cells.add(cell)
        self.target_xy_m = proposed
        self._refresh_zeta()

    def get_action_masks(self) -> np.ndarray:
        """返回论文式安全动作掩码，形状为 (N_UAV, 6)。"""
        masks = np.ones((N_UAV, 6), dtype=bool)
        occupied = {tuple(map(int, p)) for p in self.uav_pos.tolist()}
        for n, (ix, iy, h) in enumerate(self.uav_pos):
            for a in range(4):
                nx, ny = int(ix + DX[a]), int(iy + DY[a])
                if not (0 <= nx < LX and 0 <= ny < LY):
                    masks[n, a] = False
                    continue
                if self.occ[ny, nx] and h <= self.obs_h[ny, nx]:
                    masks[n, a] = False
                    continue
                if (nx, ny, int(h)) in occupied - {(int(ix), int(iy), int(h))}:
                    masks[n, a] = False
            masks[n, 4] = h < len(HEIGHTS) - 1 and (int(ix), int(iy), int(h + 1)) not in occupied
            descend_h = int(h - 1)
            masks[n, 5] = (
                h > 0
                and not (self.occ[int(iy), int(ix)] and descend_h <= self.obs_h[int(iy), int(ix)])
                and (int(ix), int(iy), descend_h) not in occupied
            )
        return masks

    # ============================================================
    # 三、感知模型 (公式 4 感知域 + 公式 5 检测/虚警)
    # ============================================================
    def _sensing_domain(self, n: int):
        """返回 UAV n 感知域内所有 (ix, iy).

        对应论文 公式 4:
            Φ_n(t) = { g_i : || p_i - p_n(t) || ≤ S_n(z_n(t)) }
            S_n = z·tan(θ/2) / L

        表 I 把 θ 取特定值, 结果就是 3 档高度对应 1 / 5 / 9 个网格.
        """
        ix, iy, h = self.uav_pos[n]
        offsets = SENSE_OFFSETS_BY_H[h]
        return [(ix + dx, iy + dy) for dx, dy in offsets
                if 0 <= ix + dx < LX and 0 <= iy + dy < LY]

    def _detect(self, n: int, ix: int, iy: int) -> int:
        """对单个格子采样检测结果 D ∈ {0, 1}.

        对应论文 公式 5:
            P(D=1 | ζ=1) = P^D(z)         (检测概率, 表 I 给 0.9/0.8/0.7)
            P(D=1 | ζ=0) = P^F(z)         (虚警概率, 表 I 给 0.1/0.2/0.3)
        """
        h = int(self.uav_pos[n, 2])
        if self.zeta[iy, ix] == 1:
            p = DET_PROB[h]
        else:
            p = FALSE_PROB[h]
        return int(self.rng.random() < p)

    # ============================================================
    # 四、局部地图更新与全局融合 (公式 6、7、8、10)
    # ============================================================
    def _update_maps(self):
        """每个时间步调用一次: 先各 UAV 局部更新, 再融合全局.

        对应论文 公式 6 (贝叶斯更新):
            D=1:  p <- p·P^D / (p·P^D + (1-p)·P^F)
            D=0:  p <- p·(1-P^D) / (p·(1-P^D) + (1-p)·(1-P^F))
            不在感知域 → p 不变
        对应论文 公式 7 (熵):
            χ = -p·log2(p) - (1-p)·log2(1-p)
        """
        for n in range(N_UAV):
            dom = self._sensing_domain(n)
            h = int(self.uav_pos[n, 2])
            pd = DET_PROB[h]
            pf = FALSE_PROB[h]
            for ix, iy in dom:
                D = self._detect(n, ix, iy)
                p = self.ltpm[n, iy, ix]
                if D == 1:
                    p_new = (p * pd) / (p * pd + (1 - p) * pf)
                else:
                    p_new = (p * (1 - pd)) / (p * (1 - pd) + (1 - p) * (1 - pf))
                p_new = float(np.clip(p_new, 1e-6, 1 - 1e-6))
                self.ltpm[n, iy, ix] = p_new
                self.leum[n, iy, ix] = _entropy(p_new)
        self._fuse_maps()

    def _fuse_maps(self):
        """融合出全局地图 gtpm / geum.

        对应论文 公式 8 (全局不确定度 = min):
            χ_i(t) = min_n χ_{n,i}(t)
        对应论文 公式 10 (全局目标概率):
            p_i(t) = p_{k,i}(t), 其中 k 满足 χ_{k,i} = min_n χ_{n,i};
            若多个并列最低, 取其中最大的 p (提升该格优先级)
        """
        # 公式 8: 全局不确定度 = 每格所有 UAV 的局部不确定度的最小值
        self.geum = self.leum.min(axis=0)

        # 公式 10: 找每格不确定度最低的 UAV, 取其 p; 并列时取 max p
        argmin = np.argmin(self.leum, axis=0)              # (LY, LX)
        min_leum = np.take_along_axis(self.leum, argmin[None], axis=0)[0]
        candidate_p = np.take_along_axis(self.ltpm, argmin[None], axis=0)[0]
        is_min = (self.leum == min_leum[None])              # (N_UAV, LY, LX)
        masked_p = np.where(is_min, self.ltpm, -1.0)
        self.gtpm = masked_p.max(axis=0).astype(np.float32)

    def area_uncertainty(self) -> float:
        """对应论文 公式 9: 区域平均不确定度 = GEUM 的均值."""
        return float(self.geum.mean())

    # ============================================================
    # 五、仿真推进 (动作 + 目标移动)
    # ============================================================
    def step(self, actions):
        """推进一个时间步. 参数/返回与 v1 一致."""
        actions = np.asarray(actions, dtype=np.int64)
        assert actions.shape == (N_UAV,), f"actions must be length {N_UAV}"

        # (1) 同步计算候选位置；冲突动作保持原位，保证 UAV-UAV/UAV-障碍避碰。
        masks = self.get_action_masks()
        candidate = self.uav_pos.copy()
        for n, a in enumerate(actions):
            if not 0 <= a < 6:
                raise ValueError(f"action {a} out of range 0..5")
            if not masks[n, a]:
                continue
            if a < 4:
                candidate[n, 0] += int(DX[a])
                candidate[n, 1] += int(DY[a])
            elif a == 4:
                candidate[n, 2] += 1
            else:
                candidate[n, 2] -= 1
        for n in range(N_UAV):
            conflict = any(n != m and np.array_equal(candidate[n], candidate[m]) for m in range(N_UAV))
            swap = any(
                n != m and np.array_equal(candidate[n], self.uav_pos[m])
                and np.array_equal(candidate[m], self.uav_pos[n]) for m in range(N_UAV)
            )
            if conflict or swap:
                candidate[n] = self.uav_pos[n]
        self.uav_pos = candidate
        for ix, iy, _ in self.uav_pos:
            self.t_last_visit[iy, ix] = self.t

        # (2) 动态目标按米制连续坐标移动。
        self._step_targets()

        # (3) 更新感知地图 (公式 6 → 公式 8 → 公式 10)
        self._update_maps()

        # (4) 公式 11: 标记被"成功搜索"的目标网格 (ζ=1 且 gtpm ≥ ξ)
        current_confirmed = (self.zeta == 1) & (self.gtpm >= TARGET_CONFIRM_THRESHOLD)
        self.searched = current_confirmed.astype(np.int8)
        for k, (ix, iy) in enumerate(self.target_cells):
            if current_confirmed[iy, ix] and not self.target_found[k]:
                self.target_found[k] = True
                self.target_first_found_step[k] = self.t + 1
        self.cumulative_searched += int(current_confirmed.sum())

        # (5) 推进时间, 计算 done
        self.t += 1
        done = self.t >= MAX_STEPS
        reward = -self.area_uncertainty()

        info = {
            "area_uncertainty": self.area_uncertainty(),
            "searched_count": int(self.searched.sum()),
            "found_count": int(self.target_found.sum()),
            "success_rate": float(self.target_found.mean()),
            "target_search_time": int(self.target_first_found_step.max()) if self.target_found.all() else MAX_STEPS,
            "cumulative_searched": int(self.cumulative_searched),
            "targets_total": N_TARGET,
        }
        return self._get_obs(), reward, done, info

    # ============================================================
    # 六、观测 + 可视化
    # ============================================================
    def _get_obs(self):
        return {
            "uav_pos": self.uav_pos.copy(),
            "zeta": self.zeta.copy(),
            "occ": self.occ.copy(),
            "ltpm": self.ltpm.copy(),
            "leum": self.leum.copy(),
            "gtpm": self.gtpm.copy(),
            "geum": self.geum.copy(),
            "t": self.t,
        }

    def render(self, ax=None):
        """俯视图: 背景 = GEUM 热图, 障碍=灰, 目标=红, UAV=蓝数字编号."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(self.geum, cmap="Greys", origin="lower", vmin=0, vmax=1, alpha=0.6)
        for iy in range(LY):
            for ix in range(LX):
                if self.occ[iy, ix]:
                    ax.add_patch(plt.Rectangle((ix - 0.5, iy - 0.5), 1, 1,
                                               facecolor="black", alpha=0.4 + 0.2 * (self.obs_h[iy, ix] / 2)))
        for iy in range(LY):
            for ix in range(LX):
                if self.zeta[iy, ix]:
                    ax.scatter(ix, iy, marker="o", c="red", s=80, edgecolors="darkred")
        for n in range(N_UAV):
            ix, iy, h = self.uav_pos[n]
            ax.scatter(ix, iy, marker="^", c="blue", s=120, edgecolors="navy")
            ax.annotate(str(n), (ix, iy), color="white", ha="center", va="center", fontsize=7, weight="bold")
        ax.set_xlim(-0.5, LX - 0.5)
        ax.set_ylim(-0.5, LY - 0.5)
        ax.set_xticks(range(LX))
        ax.set_yticks(range(LY))
        ax.grid(True, color="gray", linewidth=0.3, alpha=0.5)
        ax.set_aspect("equal")
        return ax


# ============================================================
# 演示脚本
# ============================================================
if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    env = SearchEnv(seed=0)
    obs = env.reset()
    print(f"[init] t={env.t}, area_uncertainty={env.area_uncertainty():.4f}, "
          f"obstacles={int(env.occ.sum())}, targets={int(env.zeta.sum())}")

    rewards = []
    t0 = time.time()
    for t in range(MAX_STEPS):
        actions = [int(env.rng.integers(0, 6)) for _ in range(N_UAV)]
        obs, r, done, info = env.step(actions)
        rewards.append(r)
        if done:
            break
    dt = time.time() - t0
    print(f"[end ] t={env.t}, area_uncertainty={info['area_uncertainty']:.4f}, "
          f"searched={info['searched_count']}/{info['targets_total']}, "
          f"mean_reward={np.mean(rewards):.4f}, total={dt:.2f}s, "
          f"per-step={dt / env.t * 1000:.2f}ms")

    fig, ax = plt.subplots(figsize=(6, 6))
    env.render(ax=ax)
    ax.set_title(f"t={env.t}, area uncertainty={env.area_uncertainty():.3f}")
    plt.show()
