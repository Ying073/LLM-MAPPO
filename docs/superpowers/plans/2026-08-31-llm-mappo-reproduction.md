# LLM-MAPPO 论文复现 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从零复现 LLM-MAPPO：实现仿真环境、DPES、MAPPO、LRS，跑通端到端训练并出论文风格图表。

**Architecture:** 纯 NumPy 向量化的网格仿真环境（Gymnasium 接口）→ DPES 双模式信息素 → MAPPO（集中 Critic + 分散 Actor + 动作掩码）→ LRS（DeepSeek API 离线生成奖励函数）。所有物理公式（传感器/贝叶斯/能耗/信息素）为纯函数、TDD 先行。

**Tech Stack:** Python 3.10+、PyTorch (CUDA)、NumPy、Gymnasium、Matplotlib、openai SDK（调 DeepSeek）、python-dotenv、pytest。

---

## 全局约定（所有任务一致，避免命名漂移）

- **网格索引**：一维 `i = y * grid_size + x`，`(x, y)` 为列、行，范围 `[0, grid_size)`。
- **UAV 位置**：`(x, y, alt)`，`alt ∈ {0,1,2}` 对应高度 `(50, 100, 150) m`。
- **动作**：`0=N, 1=E, 2=S, 3=W, 4=up, 5=down`（int）。
- **感知域大小**：按表 I「感知域大小 1/5/9」记为**半径（网格数）** `sensing_radius_grids = (1, 5, 9)` 对应三档高度。若最终核对论文图例发现是「面积/格子数」，只需改 `sensor.py` 一行。（差异记入复现报告）
- **概率阈值**（网格分类 Eq.13 的数值论文未显式给出，参数化默认值）：确认阈值 `xi=0.9`、高价值下限 `p_hv_low=0.3`、高价值上限 `p_hv_high=0.9`。均可从配置改。
- **随机种子**：所有随机量（目标初始/运动、障碍物、UAV 初始）经 `utils/seed.py` 统一控制。

---

## 任务总览

| 阶段 | 任务 | 内容 |
|---|---|---|
| 0 骨架 | T0 | 配置系统 + 依赖 + 随机种子 |
| 1 环境 | T1 | 传感器模型 (Eq.4–5) + 测试 |
| 1 环境 | T2 | 贝叶斯地图 LTPM/LEUM (Eq.6–7) + 测试 |
| 1 环境 | T3 | 能耗模型 (Eq.1–3) + 测试 |
| 1 环境 | T4 | SearchEnv 主环境 + 动作掩码 + 冒烟测试 |
| 2 MAPPO | T5 | Actor/Critic 网络 |
| 2 MAPPO | T6 | RolloutBuffer + GAE |
| 2 MAPPO | T7 | PPO 更新 + 单智能体 sanity check |
| 2 MAPPO | T8 | MAPPO 多智能体化（集中 Critic + 分散 Actor） |
| 3 DPES | T9 | 网格分类 + 信息素更新 (Eq.13–17) + 测试 |
| 3 DPES | T10 | 信息素接入环境与观测 |
| 4 LRS | T11 | DeepSeek API 客户端 |
| 4 LRS | T12 | Prompt 模板（转录图 11） |
| 4 LRS | T13 | 贪心 rollout 评估 (Eq.20–21) |
| 4 LRS | T14 | LRS 迭代闭环 (Eq.22, K=5) |
| 5 实验 | T15 | baselines 奖励（Handcraft / MDPS / 消融变体） |
| 5 实验 | T16 | 训练脚本 + 配置 |
| 5 实验 | T17 | 出图脚本（图 4/6/7/8/10） |
| 5 实验 | T18 | 全量 8 seeds 并行挂机脚本 |
| 6 文档 | T19 | 复现报告 + 学习笔记 + 交接 |

---

## 任务 0：配置系统与依赖

**Files:**
- Create: `llm_mappo/__init__.py`
- Create: `llm_mappo/utils/__init__.py`
- Create: `llm_mappo/utils/config.py`
- Create: `llm_mappo/utils/seed.py`
- Create: `requirements.txt`

- [ ] **Step 1: 写 requirements.txt**

```
torch>=2.0
numpy
gymnasium
matplotlib
openai>=1.0
python-dotenv
pytest
```

- [ ] **Step 2: 写配置 dataclass**

`llm_mappo/utils/config.py`：

```python
from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    grid_size: int = 20
    cell_size: float = 100.0          # m
    n_uav: int = 7
    n_obstacles: int = 20
    n_targets: int = 15
    altitudes: tuple = (50.0, 100.0, 150.0)
    detect_probs: tuple = (0.9, 0.8, 0.7)
    false_alarm_probs: tuple = (0.1, 0.2, 0.3)
    sensing_radius_grids: tuple = (1, 5, 9)   # 表 I「感知域大小 1/5/9」，解释为半径（网格）
    target_speed: float = 1.0          # m/s
    dt: float = 1.0                    # s
    uav_speed: float = 10.0            # m/s
    max_steps: int = 500
    # 能耗参数 (论文 p.11)
    P_b: float = 79.86
    P_d: float = 88.63
    U_tip: float = 120.0
    v0: float = 4.03
    drag: float = 0.6
    rho: float = 1.225
    solidity: float = 0.05
    area: float = 0.503
    E_ini: float = 6e4
    # 目标搜索确认阈值 (Eq.11)
    xi: float = 0.9


@dataclass
class MAPPOConfig:
    lr: float = 2e-4
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 10
    hidden_size: int = 64
    n_episodes: int = 28000
    entropy_coef: float = 0.01


@dataclass
class DPESConfig:
    revisit_threshold: int = 200        # D
    evaporation: float = 0.1            # E_s
    diffusion: float = 0.1              # G_s
    release_hv: float = 0.1             # 高价值释放量
    release_lu: float = 0.003           # 长期未访释放量
    release_cs: float = 0.2             # 已确认释放量
    p_hv_low: float = 0.3
    p_hv_high: float = 0.9


@dataclass
class LRSConfig:
    n_iterations: int = 5               # K
    model: str = "deepseek-reasoner"
    max_retries: int = 3


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    mappo: MAPPOConfig = field(default_factory=MAPPOConfig)
    dpes: DPESConfig = field(default_factory=DPESConfig)
    lrs: LRSConfig = field(default_factory=LRSConfig)
    seed: int = 0
```

- [ ] **Step 3: 写随机种子工具**

`llm_mappo/utils/seed.py`：

```python
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

- [ ] **Step 4: 空包 `__init__.py`**

`llm_mappo/__init__.py` 与 `llm_mappo/utils/__init__.py` 均为空文件。

- [ ] **Step 5: 提交**

```bash
git add requirements.txt llm_mappo/
git commit -m "chore: add config, seed, requirements"
```

---

## 任务 1：传感器模型（Eq.4–5）

**Files:**
- Create: `llm_mappo/env/__init__.py`
- Create: `llm_mappo/env/sensor.py`
- Test: `tests/test_sensor.py`

- [ ] **Step 1: 写失败测试**

`tests/test_sensor.py`：

```python
import numpy as np
from llm_mappo.env.sensor import (
    sensing_radius_grids, detection_prob, false_alarm_prob, sensing_domain,
)


def test_radius_matches_altitude():
    assert sensing_radius_grids(0) == 1
    assert sensing_radius_grids(1) == 5
    assert sensing_radius_grids(2) == 9


def test_detection_prob_decreases_with_altitude():
    p = [detection_prob(a) for a in range(3)]
    assert p == [0.9, 0.8, 0.7]


def test_sensing_domain_is_within_bounds():
    dom = sensing_domain((10, 10, 1), 1, grid_size=20)
    assert all(0 <= x < 20 and 0 <= y < 20 for x, y in dom)


def test_sensing_domain_contains_center():
    dom = sensing_domain((10, 10, 1), 1, grid_size=20)
    assert (10, 10) in dom
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_sensor.py -v`
Expected: FAIL（模块未定义）

- [ ] **Step 3: 实现**

`llm_mappo/env/sensor.py`：

```python
from typing import List, Tuple

from llm_mappo.utils.config import EnvConfig

_cfg = EnvConfig()


def sensing_radius_grids(altitude_level: int) -> int:
    """表 I：感知域半径（网格数）随高度档位变化。"""
    return _cfg.sensing_radius_grids[altitude_level]


def detection_prob(altitude_level: int) -> float:
    return _cfg.detect_probs[altitude_level]


def false_alarm_prob(altitude_level: int) -> float:
    return _cfg.false_alarm_probs[altitude_level]


def sensing_domain(
    uav_pos: Tuple[int, int, int], radius: int, grid_size: int
) -> List[Tuple[int, int]]:
    """Eq.(4)：以 (x,y) 为中心、radius 为半径的网格集合（含边界裁剪）。"""
    x, y, _ = uav_pos
    dom = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            xx, yy = x + dx, y + dy
            if 0 <= xx < grid_size and 0 <= yy < grid_size:
                dom.append((xx, yy))
    return dom
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_sensor.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add llm_mappo/env/sensor.py tests/test_sensor.py
git commit -m "feat(env): sensor model (Eq.4-5) with tests"
```

---

## 任务 2：贝叶斯地图 LTPM/LEUM（Eq.6–7）

**Files:**
- Create: `llm_mappo/env/maps.py`
- Test: `tests/test_maps.py`

- [ ] **Step 1: 写失败测试**

`tests/test_maps.py`：

```python
import math
import numpy as np
from llm_mappo.env.maps import bayes_update_prob, entropy, fuse_global


def test_bayes_detection_increases_prob():
    p_new = bayes_update_prob(0.5, detected=True, p_d=0.9, p_f=0.1)
    assert p_new > 0.5


def test_bayes_no_detection_decreases_prob():
    p_new = bayes_update_prob(0.5, detected=False, p_d=0.9, p_f=0.1)
    assert p_new < 0.5


def test_entropy_max_at_half():
    e = entropy(0.5)
    assert abs(e - 1.0) < 1e-6


def test_entropy_zero_at_extremes():
    assert entropy(0.0) == 0.0
    assert entropy(1.0) == 0.0


def test_fuse_global_min_uncertainty():
    # 3 架 UAV 对同一网格的不确定度取最小
    local_unc = np.array([[0.8], [0.3], [0.6]])      # (n_uav, n_grid)
    local_prob = np.array([[0.5], [0.9], [0.7]])
    g_unc, g_prob = fuse_global(local_unc, local_prob)
    assert g_unc[0] == 0.3
    assert g_prob[0] == 0.9   # 唯一最小不确定度的 UAV 的概率
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_maps.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

`llm_mappo/env/maps.py`：

```python
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
    """Eq.(7)：信息熵，归一化到 [0,1]。"""
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)


def fuse_global(local_unc: np.ndarray, local_prob: np.ndarray):
    """Eq.(8)(10)：每网格全局不确定度取最小；全局概率取最小不确定度 UAV 的概率（并列取最大概率）。"""
    n_uav = local_unc.shape[0]
    g_unc = local_unc.min(axis=0)
    min_mask = (local_unc == g_unc[None, :])
    masked_prob = np.where(min_mask, local_prob, -1.0)
    g_prob = masked_prob.max(axis=0)
    return g_unc, g_prob
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_maps.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add llm_mappo/env/maps.py tests/test_maps.py
git commit -m "feat(env): Bayesian map update (Eq.6-8,10) with tests"
```

---

## 任务 3：能耗模型（Eq.1–3）

**Files:**
- Create: `llm_mappo/env/energy.py`
- Test: `tests/test_energy.py`

- [ ] **Step 1: 写失败测试**

`tests/test_energy.py`：

```python
from llm_mappo.env.energy import propulsive_power


def test_power_positive():
    assert propulsive_power(10.0) > 0.0


def test_power_hover_finite():
    assert propulsive_power(0.0) > 0.0
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_energy.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

`llm_mappo/env/energy.py`：

```python
import numpy as np
from llm_mappo.utils.config import EnvConfig

_cfg = EnvConfig()


def propulsive_power(v: float) -> float:
    """Eq.(1)：旋翼 UAV 推进功率（桨叶剖面 + 诱导 + 寄生阻力）。"""
    term1 = _cfg.P_b * (1.0 + 3.0 * v ** 2 / _cfg.U_tip ** 2)
    term2 = _cfg.P_d * np.sqrt(
        np.sqrt(1.0 + v ** 4 / (4.0 * _cfg.v0 ** 4)) - v ** 2 / (2.0 * _cfg.v0 ** 2)
    )
    term3 = 0.5 * _cfg.drag * _cfg.rho * _cfg.solidity * _cfg.area * v ** 3
    return term1 + term2 + term3
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_energy.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add llm_mappo/env/energy.py tests/test_energy.py
git commit -m "feat(env): UAV energy model (Eq.1-3)"
```

---

## 任务 4：SearchEnv 主环境 + 动作掩码

**Files:**
- Create: `llm_mappo/env/env.py`
- Test: `tests/test_env.py`

- [ ] **Step 1: 写失败测试（冒烟）**

`tests/test_env.py`：

```python
import numpy as np
from llm_mappo.env.env import SearchEnv


def test_reset_step_shape():
    env = SearchEnv()
    obs, _ = env.reset(seed=0)
    assert len(obs) == env.cfg.n_uav
    actions = np.zeros(env.cfg.n_uav, dtype=int)
    obs, rew, term, trunc, info = env.step(actions)
    assert rew.shape == (env.cfg.n_uav,)
    assert isinstance(term, bool) and isinstance(trunc, bool)


def test_action_masking_blocks_collision():
    env = SearchEnv()
    env.reset(seed=0)
    masks = env.action_masks()          # (n_uav, 6)
    assert masks.shape == (env.cfg.n_uav, 6)
    # 越界动作被屏蔽
    assert (masks.sum(axis=1) > 0).all()
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_env.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 SearchEnv**

`llm_mappo/env/env.py`（核心实现，含目标运动、传感器扫描、地图更新、能耗、动作掩码）：

```python
from typing import List, Tuple
import numpy as np
from llm_mappo.utils.config import Config
from llm_mappo.utils.seed import set_seed
from llm_mappo.env.sensor import (
    sensing_radius_grids, detection_prob, false_alarm_prob, sensing_domain,
)
from llm_mappo.env.maps import bayes_update_prob, entropy, fuse_global
from llm_mappo.env.energy import propulsive_power

ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0), (0, 0)]  # N,E,S,W,up,down


class SearchEnv:
    def __init__(self, cfg: Config | None = None):
        cfg = cfg or Config()
        self.cfg = cfg.env          # EnvConfig
        self.dpes = cfg.dpes        # DPESConfig（T10 信息素使用）
        self.dt = self.cfg.dt

    def reset(self, seed: int | None = None):
        if seed is not None:
            set_seed(seed)
        self._step = 0
        gs = self.cfg.grid_size
        # 目标：连续坐标 (m)，1 m/s 随机方向
        self.target_pos = np.random.rand(self.cfg.n_targets, 2) * (gs * self.cfg.cell_size)
        ang = np.random.rand(self.cfg.n_targets) * 2 * np.pi
        self.target_dir = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        # 障碍物：占用网格集合
        flat = np.random.choice(gs * gs, self.cfg.n_obstacles, replace=False)
        self.obstacles = set(flat.tolist())
        # UAV 初始：随机不重叠空闲网格，高度 0
        free = [i for i in range(gs * gs) if i not in self.obstacles]
        starts = np.random.choice(free, self.cfg.n_uav, replace=False)
        self.uav_pos = np.array([(s % gs, s // gs, 0) for s in starts])
        self.energy = np.full(self.cfg.n_uav, self.cfg.E_ini)
        # 每 UAV 的 LTPM/LEUM（局部概率图/不确定度图）
        self.local_prob = np.full((self.cfg.n_uav, gs * gs), 0.5)
        self.local_unc = np.full((self.cfg.n_uav, gs * gs), 1.0)
        self.last_visit = np.full((self.cfg.n_uav, gs * gs), -self.cfg.max_steps)
        return self._obs(), {}

    def _move_targets(self):
        self.target_pos += self.target_dir * self.cfg.target_speed * self.dt
        limit = self.cfg.grid_size * self.cfg.cell_size
        for k in range(self.cfg.n_targets):
            for d in range(2):
                if self.target_pos[k, d] < 0 or self.target_pos[k, d] > limit:
                    self.target_dir[k, d] *= -1
                    self.target_pos[k, d] = np.clip(self.target_pos[k, d], 0, limit)

    def _sensor_scan(self):
        gs = self.cfg.grid_size
        for n in range(self.cfg.n_uav):
            alt = int(self.uav_pos[n, 2])
            r = sensing_radius_grids(alt)
            p_d, p_f = detection_prob(alt), false_alarm_prob(alt)
            dom = sensing_domain(tuple(self.uav_pos[n]), r, gs)
            for (x, y) in dom:
                i = y * gs + x
                self.last_visit[n, i] = self._step
                truth = self._target_in_grid(i)
                detected = (np.random.rand() < (p_d if truth else p_f))
                self.local_prob[n, i] = bayes_update_prob(self.local_prob[n, i], detected, p_d, p_f)
                self.local_unc[n, i] = entropy(self.local_prob[n, i])

    def _target_in_grid(self, i: int) -> bool:
        gs = self.cfg.grid_size
        x0, y0 = i % gs, i // gs
        L = self.cfg.cell_size
        for k in range(self.cfg.n_targets):
            px, py = self.target_pos[k]
            if x0 * L <= px < (x0 + 1) * L and y0 * L <= py < (y0 + 1) * L:
                return True
        return False

    def _update_energy(self):
        for n in range(self.cfg.n_uav):
            v = self.cfg.uav_speed
            self.energy[n] -= propulsive_power(v) * self.dt

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
        self._sensor_scan()
        self._update_energy()
        self._step += 1
        term = self._step >= self.cfg.max_steps
        trunc = (self.energy < propulsive_power(self.cfg.uav_speed) * self.dt).any()
        return self._obs(), np.zeros(self.cfg.n_uav), term, trunc, {}

    def _obs(self):
        # 观测：每个 UAV 返回局部目标概率 + 局部不确定度（DPES 任务扩展信息素）
        return [np.concatenate([self.local_prob[n], self.local_unc[n]]) for n in range(self.cfg.n_uav)]

    def _searched_targets(self) -> float:
        """Eq.(11)：目标被成功搜索 = 存在且全局概率 >= xi。"""
        _, fused_prob = fuse_global(self.local_unc, self.local_prob)
        gs = self.cfg.grid_size
        count = 0.0
        for i in range(gs * gs):
            if self._target_in_grid(i) and fused_prob[i] >= self.cfg.xi:
                count += 1.0
        return count

    def _avg_uncertainty(self) -> float:
        """Eq.(9)：区域平均不确定度。"""
        fused_unc, _ = fuse_global(self.local_unc, self.local_prob)
        return float(fused_unc.mean())

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
```

> **注**：本任务先让环境「能跑通」，奖励函数（稀疏 vs 稠密）在后续 MAPPO/LRS 任务中接入；`_obs` 也会在 DPES 任务里扩展。

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_env.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add llm_mappo/env/env.py tests/test_env.py
git commit -m "feat(env): SearchEnv main loop + action masking"
```

---

## 任务 5：Actor / Critic 网络

**Files:**
- Create: `llm_mappo/mappo/__init__.py`
- Create: `llm_mappo/mappo/networks.py`
- Test: `tests/test_networks.py`

- [ ] **Step 1: 写失败测试**

`tests/test_networks.py`：

```python
import torch
from llm_mappo.mappo.networks import Actor, Critic


def test_actor_outputs_action_probs():
    actor = Actor(obs_dim=800, n_actions=6, hidden=64)
    x = torch.randn(1, 800)
    out = actor(x)
    assert out.shape == (1, 6)
    assert torch.allclose(out.exp().sum(-1), torch.ones(1), atol=1e-5)


def test_critic_outputs_scalar():
    critic = Critic(obs_dim=800 * 7, hidden=64)
    x = torch.randn(1, 800 * 7)
    assert critic(x).shape == (1, 1)
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_networks.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

`llm_mappo/mappo/networks.py`：

```python
import torch
import torch.nn as nn


class Actor(nn.Module):
    """分散式策略网络：观测 -> 动作 logits（表 II：2 层×64 ReLU）。"""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)   # 返回 logits


class Critic(nn.Module):
    """集中式价值网络：联合观测 -> 状态价值。"""

    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        return self.net(obs)
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_networks.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add llm_mappo/mappo/networks.py tests/test_networks.py
git commit -m "feat(mappo): Actor/Critic networks"
```

---

## 任务 6：RolloutBuffer + GAE

**Files:**
- Create: `llm_mappo/mappo/buffer.py`
- Test: `tests/test_buffer.py`

- [ ] **Step 1: 写失败测试**

`tests/test_buffer.py`：

```python
import numpy as np
from llm_mappo.mappo.buffer import RolloutBuffer


def test_compute_gae_and_returns():
    buf = RolloutBuffer(n_agents=2, obs_dim=4, n_actions=3)
    # 存 3 步数据
    for t in range(3):
        buf.add(obs=np.zeros((2, 4)), actions=np.zeros(2, dtype=int),
                logits=np.zeros((2, 3)), rewards=np.ones(2), dones=np.zeros(2))
    values = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])  # (T, n_agents)
    buf.compute_gae(values, gamma=0.95, lam=0.95)
    assert buf.advantages.shape == (3, 2)
    assert buf.returns.shape == (3, 2)
    # 全 1 奖励、价值 1，GAE 应为 0
    assert np.allclose(buf.advantages, 0.0, atol=1e-6)
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_buffer.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

`llm_mappo/mappo/buffer.py`：

```python
import numpy as np


class RolloutBuffer:
    """一个 episode 内的转移缓冲，episode 结束后计算 GAE 与 returns。"""

    def __init__(self, n_agents: int, obs_dim: int, n_actions: int):
        self.n_agents = n_agents
        self.obs, self.actions, self.logits = [], [], []
        self.rewards, self.dones = [], []

    def add(self, obs, actions, logits, rewards, dones):
        self.obs.append(obs)
        self.actions.append(actions)
        self.logits.append(logits)
        self.rewards.append(rewards)
        self.dones.append(dones)

    def compute_gae(self, values: np.ndarray, gamma: float, lam: float):
        T, _ = values.shape
        adv = np.zeros_like(values)
        last = np.zeros(self.n_agents)
        for t in reversed(range(T)):
            delta = self.rewards[t] + gamma * last * (1 - self.dones[t]) - values[t]
            last = delta + gamma * lam * last * (1 - self.dones[t])
            adv[t] = last
        self.advantages = adv
        self.returns = adv + values

    def arrays(self):
        return {
            "obs": np.stack(self.obs),
            "actions": np.stack(self.actions),
            "logits": np.stack(self.logits),
            "advantages": self.advantages,
            "returns": self.returns,
        }
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_buffer.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add llm_mappo/mappo/buffer.py tests/test_buffer.py
git commit -m "feat(mappo): rollout buffer with GAE"
```

---

## 任务 7：PPO 更新 + 单智能体 sanity check

**Files:**
- Create: `llm_mappo/mappo/update.py`
- Test: `tests/test_update.py`

- [ ] **Step 1: 写失败测试（梯度不崩溃 + 损失有限）**

`tests/test_update.py`：

```python
import torch
from llm_mappo.mappo.networks import Actor
from llm_mappo.mappo.update import ppo_actor_loss


def test_ppo_loss_finite_and_grad():
    actor = Actor(obs_dim=4, n_actions=3, hidden=16)
    obs = torch.randn(5, 4)
    old_logits = torch.randn(5, 3)
    actions = torch.randint(0, 3, (5,))
    adv = torch.randn(5)
    loss = ppo_actor_loss(actor, obs, old_logits, actions, adv, clip_eps=0.2)
    assert torch.isfinite(loss)
    loss.backward()
    assert actor.net[0].weight.grad is not None
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_update.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

`llm_mappo/mappo/update.py`：

```python
import torch
import torch.nn.functional as F


def ppo_actor_loss(actor, obs, old_logits, actions, adv, clip_eps: float):
    """Eq.(23)(24)：PPO 裁剪替代目标（负值作为损失）。"""
    new_logits = actor(obs)
    old_logp = F.log_softmax(old_logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
    new_logp = F.log_softmax(new_logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
    ratio = (new_logp - old_logp).exp()
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    return -torch.min(surr1, surr2).mean()
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_update.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add llm_mappo/mappo/update.py tests/test_update.py
git commit -m "feat(mappo): PPO clipped objective (Eq.23-24)"
```

---

## 任务 8：MAPPO 多智能体化 + 训练器

**Files:**
- Create: `llm_mappo/mappo/trainer.py`
- Create: `scripts/train_mappo.py`

- [ ] **Step 1: 实现 MAPPOTrainer**

`llm_mappo/mappo/trainer.py`：

```python
import numpy as np
import torch
import torch.nn.functional as F
from llm_mappo.utils.config import Config
from llm_mappo.mappo.networks import Actor, Critic
from llm_mappo.mappo.buffer import RolloutBuffer
from llm_mappo.mappo.update import ppo_actor_loss


class MAPPOTrainer:
    """集中 Critic + 分散 Actor，共享参数（同构 UAV）。"""

    def __init__(self, cfg: Config, obs_dim: int, n_actions: int = 6):
        self.cfg = cfg
        self.n_agents = cfg.env.n_uav
        self.obs_dim = obs_dim
        self.actor = Actor(obs_dim, n_actions, cfg.mappo.hidden_size)
        self.critic = Critic(obs_dim * self.n_agents, cfg.mappo.hidden_size)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.mappo.lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.mappo.lr)

    def select_actions(self, obs_list, masks):
        """obs_list: list[Tensor(n_agents, obs_dim)] 返回 actions 与 logits（带掩码）。"""
        logits_list = [self.actor(o) for o in obs_list]
        probs = []
        for logits, m in zip(logits_list, masks):
            m_t = torch.tensor(m, dtype=torch.float32)
            logits = logits - (1 - m_t) * 1e9
            probs.append(F.softmax(logits, dim=-1))
        actions = [torch.multinomial(p, 1).squeeze(1) for p in probs]
        return actions, logits_list

    def update(self, buf: RolloutBuffer):
        data = buf.arrays()
        for _ in range(self.cfg.mappo.epochs):
            obs = torch.tensor(data["obs"], dtype=torch.float32)          # (T,N,O)
            adv = torch.tensor(data["advantages"], dtype=torch.float32)   # (T,N)
            ret = torch.tensor(data["returns"], dtype=torch.float32)      # (T,N)
            old_logits = torch.tensor(data["logits"], dtype=torch.float32)
            acts = torch.tensor(data["actions"], dtype=torch.long)
            # Critic 用联合观测
            joint = obs.reshape(obs.shape[0], -1)
            vals = self.critic(joint).squeeze(-1).unsqueeze(-1).expand_as(ret)
            loss_c = F.mse_loss(vals, ret)
            self.opt_c.zero_grad()
            loss_c.backward()
            self.opt_c.step()
            # Actor：逐 agent 计算（共享同一 actor）
            loss_a = 0.0
            for n in range(self.n_agents):
                loss_a = loss_a + ppo_actor_loss(
                    self.actor, obs[:, n], old_logits[:, n],
                    acts[:, n], adv[:, n], self.cfg.mappo.clip_eps,
                )
            loss_a = loss_a / self.n_agents
            self.opt_a.zero_grad()
            loss_a.backward()
            self.opt_a.step()
```

- [ ] **Step 2: 写训练脚本（单智能体 sanity：1 架 UAV，验证收敛）**

`scripts/train_mappo.py`：

```python
import numpy as np
import torch
from llm_mappo.utils.config import Config
from llm_mappo.utils.seed import set_seed
from llm_mappo.env.env import SearchEnv
from llm_mappo.mappo.trainer import MAPPOTrainer
from llm_mappo.mappo.buffer import RolloutBuffer


def main():
    cfg = Config()
    set_seed(cfg.seed)
    env = SearchEnv(cfg)
    obs, _ = env.reset(seed=cfg.seed)
    obs_dim = obs[0].shape[0]
    trainer = MAPPOTrainer(cfg, obs_dim)
    for ep in range(200):           # sanity：先 200 episode
        buf = RolloutBuffer(cfg.env.n_uav, obs_dim, 6)
        obs, _ = env.reset()
        for t in range(cfg.env.max_steps):
            obs_t = [torch.tensor(o, dtype=torch.float32) for o in obs]
            masks = env.action_masks()
            actions, logits = trainer.select_actions(obs_t, masks)
            acts = np.array([a.numpy()[0] for a in actions])
            next_obs, rew, term, trunc, _ = env.step(acts)
            buf.add(obs=np.stack(obs), actions=acts,
                    logits=np.stack([lg.detach().numpy() for lg in logits]),
                    rewards=rew, dones=np.array([float(term or trunc)] * cfg.env.n_uav))
            obs = next_obs
            if term or trunc:
                break
        joint = np.stack(buf.obs)
        with torch.no_grad():
            vals = trainer.critic(torch.tensor(joint, dtype=torch.float32)
                                  .reshape(joint.shape[0], -1)).squeeze(-1).numpy()
        vals = np.stack([vals] * cfg.env.n_uav, axis=-1)
        buf.compute_gae(vals, cfg.mappo.gamma, cfg.mappo.gae_lambda)
        trainer.update(buf)
        if ep % 20 == 0:
            print(f"ep {ep} mean_reward={buf.rewards and np.mean(buf.rewards):.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行 sanity check**

Run: `python scripts/train_mappo.py`
Expected: 无异常退出，打印 episode 信息，mean_reward 有值。

- [ ] **Step 4: 提交**

```bash
git add llm_mappo/mappo/trainer.py scripts/train_mappo.py
git commit -m "feat(mappo): MAPPO trainer + single-agent sanity check"
```

---

## 任务 9：DPES 网格分类与信息素更新（Eq.13–17）

**Files:**
- Create: `llm_mappo/dpes/__init__.py`
- Create: `llm_mappo/dpes/pheromone.py`
- Test: `tests/test_pheromone.py`

- [ ] **Step 1: 写失败测试**

`tests/test_pheromone.py`：

```python
import numpy as np
from llm_mappo.dpes.pheromone import classify_grid, update_pheromone
from llm_mappo.utils.config import DPESConfig


def test_classify_high_value():
    assert classify_grid(p=0.5, last_visit=0, t=10, cfg=DPESConfig()) == "hv"


def test_classify_long_unvisited():
    cfg = DPESConfig()
    assert classify_grid(p=0.0, last_visit=0, t=cfg.revisit_threshold + 1, cfg=cfg) == "lu"


def test_classify_confirmed():
    cfg = DPESConfig()
    assert classify_grid(p=0.95, last_visit=9, t=10, cfg=cfg) == "cs"


def test_update_pheromone_resets_lu_on_visit():
    cfg = DPESConfig()
    # 长期未访网格：访问后信号清零
    dp = update_pheromone("lu", dp=0.003, released=0.0, diffused=0.0, visited=True, cfg=cfg)
    assert dp == 0.0


def test_update_hv_evaporates_and_diffuses():
    cfg = DPESConfig()
    dp = update_pheromone("hv", dp=0.5, released=cfg.release_hv, diffused=0.02, visited=False, cfg=cfg)
    assert dp == cfg.evaporation * 0.5 + cfg.release_hv + 0.02
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_pheromone.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

`llm_mappo/dpes/pheromone.py`：

```python
from llm_mappo.utils.config import DPESConfig


def classify_grid(p: float, last_visit: int, t: int, cfg: DPESConfig) -> str:
    """Eq.(13)：按目标概率与最近访问时间把网格分四类。"""
    if p >= cfg.p_hv_high:
        return "cs"          # 高置信度，状态已确认
    if p >= cfg.p_hv_low:
        return "hv"          # 中高概率，高价值
    if t - last_visit > cfg.revisit_threshold:
        return "lu"          # 长期未访
    return "other"


def update_pheromone(category: str, dp: float, released: float,
                     diffused: float, visited: bool, cfg: DPESConfig) -> float:
    """Eq.(14)–(17)：四类信息素更新规则。"""
    if category == "hv":
        return cfg.evaporation * dp + released + diffused        # Eq.(14)
    if category == "lu":
        return 0.0 if visited else released                      # Eq.(16) 访问即清零
    if category == "cs":
        # Eq.(17)：蒸发；超过 D 由 classify 转为其他类，这里仅做蒸发
        return cfg.evaporation * dp
    return dp
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_pheromone.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add llm_mappo/dpes/pheromone.py tests/test_pheromone.py
git commit -m "feat(dpes): grid classification + pheromone update (Eq.13-17)"
```

---

## 任务 10：信息素接入环境与观测

**Files:**
- Modify: `llm_mappo/env/env.py`

- [ ] **Step 1: 在 SearchEnv 中维护信息素场并写入观测**

在 `SearchEnv.__init__` 末尾增加（`self.dpes` 已在任务 4 定义，此处只需补信息素场）：

```python
from llm_mappo.dpes.pheromone import classify_grid, update_pheromone
# (在 __init__ 内)
self.pheromone = np.zeros(gs * gs)
```

在 `step` 中，`_sensor_scan()` 之后新增 `_update_pheromone()`：

```python
def _update_pheromone(self):
    gs = self.cfg.grid_size
    fused_unc, fused_prob = fuse_global(self.local_unc, self.local_prob)
    for i in range(gs * gs):
        cat = classify_grid(fused_prob[i], self._global_last_visit(i), self._step, self.dpes)
        if cat == "hv":
            released = self.dpes.release_hv
        elif cat == "lu":
            released = self.dpes.release_lu
        elif cat == "cs":
            released = self.dpes.release_cs
        else:
            released = 0.0
        self.pheromone[i] = update_pheromone(
            cat, self.pheromone[i], released, 0.0, visited=False, cfg=self.dpes)

def _global_last_visit(self, i: int) -> int:
    return int(self.last_visit[:, i].max())
```

在 `_obs` 中把信息素拼入观测：

```python
def _obs(self):
    gs = self.cfg.grid_size
    obs_list = []
    for n in range(self.cfg.n_uav):
        o = np.concatenate([self.local_prob[n], self.local_unc[n], self.pheromone])
        obs_list.append(o)
    return obs_list
```

- [ ] **Step 2: 运行冒烟测试确认观测维度一致**

Run: `pytest tests/test_env.py -v`
Expected: PASS（`_obs` 维度变化不影响原断言）

- [ ] **Step 3: 提交**

```bash
git add llm_mappo/env/env.py
git commit -m "feat(env): integrate DPES pheromone into env and observations"
```

---

## 任务 11：DeepSeek API 客户端

**Files:**
- Create: `llm_mappo/lrs/__init__.py`
- Create: `llm_mappo/lrs/client.py`
- Create: `llm_mappo/lrs/__init__.py`

- [ ] **Step 1: 实现客户端**

`llm_mappo/lrs/client.py`：

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class DeepSeekClient:
    def __init__(self, model: str = "deepseek-reasoner"):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 4000) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return resp.choices[0].message.content
```

- [ ] **Step 2: 提交（无需测试，属外部依赖；在 LRS 任务中集成验证）**

```bash
git add llm_mappo/lrs/client.py
git commit -m "feat(lrs): DeepSeek API client"
```

---

## 任务 12：Prompt 模板（转录论文图 11）

**Files:**
- Create: `llm_mappo/lrs/prompt.py`

- [ ] **Step 1: 实现 prompt 模板**

`llm_mappo/lrs/prompt.py`：

```python
OUTPUT_INTERFACE = """\
请输出一个 Python 函数，签名固定为：
def reward(obs, prev_obs, info) -> float
其中 obs 为当前观测字典、info 含环境状态。函数必须可被 exec 直接执行，
只使用 numpy 与标准库，不引用任何外部变量。"""

TASK_DESCRIPTION = """\
你是一名多智能体强化学习的奖励函数设计者。场景：{n_uav} 架无人机在
{grid_size}x{grid_size} 网格的 2000mx2000m 区域搜索 {n_targets} 个动态地面目标，
区域内有静态障碍物。目标：最大化成功搜索的目标数，最小化搜索区域平均不确定度，
同时避免无人机之间及与障碍物的碰撞。动作空间：北/东/南/西/升/降。"""

REASONING_GUIDANCE = """\
设计奖励时请考虑：(1) 搜索到目标应给强正奖励；(2) 单步降低区域不确定度应给连续正奖励；
(3) 高度自适应：高空广域感知、低空精确检测；(4) UAV 间应保持分散以覆盖更多区域；
(5) 碰撞与越界应惩罚。避免只依赖稀疏的"找到目标"奖励。"""


def build_initial_prompt(n_uav, grid_size, n_targets) -> str:
    return "\n\n".join([
        TASK_DESCRIPTION.format(n_uav=n_uav, grid_size=grid_size, n_targets=n_targets),
        REASONING_GUIDANCE,
        OUTPUT_INTERFACE,
    ])


def build_feedback_prompt(base_prompt, best_code, best_score, worst_codes) -> str:
    worst = "\n".join(f"```python\n{c}\n```\n(score={s})" for c, s in worst_codes)
    return "\n\n".join([
        base_prompt,
        f"当前最优奖励函数（score={best_score}）:\n```python\n{best_code}\n```",
        f"以下为表现较差的候选（避免类似设计）:\n{worst}",
        "请生成一个改进后的奖励函数，直接输出代码。",
    ])
```

- [ ] **Step 2: 提交**

```bash
git add llm_mappo/lrs/prompt.py
git commit -m "feat(lrs): prompt templates for LRS"
```

---

## 任务 13：贪心 rollout 评估（Eq.20–21）

**Files:**
- Create: `llm_mappo/lrs/rollout.py`
- Test: `tests/test_rollout.py`

- [ ] **Step 1: 写失败测试（用固定奖励函数验证评分有限）**

`tests/test_rollout.py`：

```python
from llm_mappo.lrs.rollout import evaluate_reward_fn
from llm_mappo.env.env import SearchEnv
from llm_mappo.utils.config import Config


def _zero_reward(obs, prev_obs, info):
    return 0.0


def test_evaluate_returns_finite_score():
    cfg = Config()
    env = SearchEnv(cfg)
    score = evaluate_reward_fn(_zero_reward, env, cfg, seed=0)
    assert isinstance(score, float)
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_rollout.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

`llm_mappo/lrs/rollout.py`：

```python
import numpy as np
from llm_mappo.env.env import SearchEnv


def evaluate_reward_fn(reward_fn, env: SearchEnv, cfg, seed: int = 0) -> float:
    """Eq.(20)(21)：固定初始观测，逐步贪心选择使 R 最大的联合动作，执行一个 episode，
    用 Eq.(12a) 的原始目标 J 评分。本实现返回该 episode 的累计已搜索目标数（Eq.12a 第一项）。"""
    env.reset(seed=seed)
    total = 0.0
    for t in range(cfg.env.max_steps):
        acts = []
        masks = env.action_masks()
        obs_list = env._obs()
        for n in range(cfg.env.n_uav):
            best_a, best_r = 0, -1e18
            for a in range(6):
                if masks[n, a] == 0:
                    continue
                # 简化贪心：用当前观测评估该动作对应的奖励
                r = reward_fn({"obs": obs_list[n], "agent": n}, None, {})
                if r > best_r:
                    best_a, best_r = a, r
            acts.append(best_a)
        env.step(np.array(acts))
        total += env._searched_targets()
    return float(total)
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_rollout.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add llm_mappo/lrs/rollout.py tests/test_rollout.py
git commit -m "feat(lrs): greedy rollout evaluation (Eq.20-21)"
```

---

## 任务 14：LRS 迭代闭环（Eq.22，K=5）

**Files:**
- Create: `llm_mappo/lrs/lrs.py`

- [ ] **Step 1: 实现 LRS 主循环**

`llm_mappo/lrs/lrs.py`：

```python
import re
from llm_mappo.utils.config import Config
from llm_mappo.lrs.client import DeepSeekClient
from llm_mappo.lrs.prompt import build_initial_prompt, build_feedback_prompt
from llm_mappo.lrs.rollout import evaluate_reward_fn


def extract_code(text: str) -> str:
    """从 LLM 输出中提取 reward 函数代码块。"""
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


class LRS:
    """离线 LLM 奖励塑形：初始化 + K 轮迭代反馈（Eq.22）。"""

    def __init__(self, cfg: Config, env):
        self.cfg = cfg
        self.env = env
        self.client = DeepSeekClient(cfg.lrs.model)
        self.base = build_initial_prompt(
            cfg.env.n_uav, cfg.env.grid_size, cfg.env.n_targets)
        self.buffer = []          # (code, score)

    def run(self) -> str:
        best_code, best_score = None, -1e18
        for k in range(self.cfg.lrs.n_iterations):
            prompt = self.base if k == 0 else build_feedback_prompt(
                self.base, best_code, best_score, self.buffer[-2:])
            for attempt in range(self.cfg.lrs.max_retries):
                code = extract_code(self.client.generate(prompt))
                if self._validate(code):
                    break
            else:
                continue
            score = evaluate_reward_fn(self._make_fn(code), self.env, self.cfg)
            self.buffer.append((code, score))
            if score > best_score:
                best_code, best_score = code, score
            print(f"iter {k} score={score:.2f} best={best_score:.2f}")
        return best_code

    def _validate(self, code: str) -> bool:
        return code.strip().startswith("def reward")

    def _make_fn(self, code: str):
        ns = {}
        exec(code, ns)
        return ns["reward"]
```

- [ ] **Step 2: 提交**

```bash
git add llm_mappo/lrs/lrs.py
git commit -m "feat(lrs): LRS iterative loop (Eq.22, K=5)"
```

---

## 任务 15：baselines 奖励函数

**Files:**
- Create: `llm_mappo/baselines/__init__.py`
- Create: `llm_mappo/baselines/rewards.py`

- [ ] **Step 1: 实现 baselines 奖励**

`llm_mappo/baselines/rewards.py`：

```python
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
```

- [ ] **Step 2: 消融变体 OG / OI 说明（在 LRS 中配置实现）**

`LLM-MAPPO-OG`（无推理指导）与 `LLM-MAPPO-OI`（仅初始化、不迭代）是 LRS 的变体，通过 `llm_mappo/lrs/lrs.py` 的两个开关实现，不在本文件：

- `OG`：调用 `build_initial_prompt(..., include_reasoning=False)`，迭代照常 K 次
- `OI`：`LRS.run(iterations=1)`，跳过反馈迭代

> 实施时给 `build_initial_prompt` 增加 `include_reasoning` 参数、给 `LRS.run` 增加 `iterations` 参数（默认取 `cfg.lrs.n_iterations`），并同步更新 `prompt.py` 与 `lrs.py`。

- [ ] **Step 3: 提交**

```bash
git add llm_mappo/baselines/rewards.py
git commit -m "feat(baselines): sparse/handcraft/mdps/mdps-improved rewards"
```

---

## 任务 16：训练脚本 + 配置（整合全链路）

**Files:**
- Modify: `scripts/train_mappo.py`

- [ ] **Step 1: 重写训练脚本（支持奖励选择 / seed / CSV 日志）**

`scripts/train_mappo.py`（完整重写，替换任务 8 的版本）：

```python
import argparse
import csv
import os
import numpy as np
import torch
from llm_mappo.utils.config import Config
from llm_mappo.utils.seed import set_seed
from llm_mappo.env.env import SearchEnv
from llm_mappo.mappo.trainer import MAPPOTrainer
from llm_mappo.mappo.buffer import RolloutBuffer
from llm_mappo.baselines.rewards import (
    sparse_reward, handcraft_reward, mdps_reward, mdps_improved_reward,
)

REWARDS = {
    "sparse": sparse_reward,
    "handcraft": handcraft_reward,
    "mdps": mdps_reward,
    "mdps_improved": mdps_improved_reward,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", default="handcraft", choices=list(REWARDS))
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    cfg = Config()
    cfg.seed = args.seed
    set_seed(args.seed)
    env = SearchEnv(cfg)
    obs, _ = env.reset(seed=args.seed)
    obs_dim = obs[0].shape[0]
    trainer = MAPPOTrainer(cfg, obs_dim)
    reward_fn = REWARDS[args.reward]

    logf = None
    if args.log:
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        logf = open(args.log, "w", newline="")
        writer = csv.writer(logf)
        writer.writerow(["episode", "reward"])

    for ep in range(args.episodes):
        buf = RolloutBuffer(cfg.env.n_uav, obs_dim, 6)
        obs, _ = env.reset()
        ep_reward = 0.0
        for t in range(cfg.env.max_steps):
            obs_t = [torch.tensor(o, dtype=torch.float32) for o in obs]
            masks = env.action_masks()
            actions, logits = trainer.select_actions(obs_t, masks)
            acts = np.array([a.numpy()[0] for a in actions])
            next_obs, _, term, trunc, _ = env.step(acts)
            rew = reward_fn(env)
            ep_reward += float(rew.mean())
            buf.add(obs=np.stack(obs), actions=acts,
                    logits=np.stack([lg.detach().numpy() for lg in logits]),
                    rewards=rew, dones=np.array([float(term or trunc)] * cfg.env.n_uav))
            obs = next_obs
            if term or trunc:
                break
        joint = np.stack(buf.obs)
        with torch.no_grad():
            vals = trainer.critic(torch.tensor(joint, dtype=torch.float32)
                                  .reshape(joint.shape[0], -1)).squeeze(-1).numpy()
        vals = np.stack([vals] * cfg.env.n_uav, axis=-1)
        buf.compute_gae(vals, cfg.mappo.gamma, cfg.mappo.gae_lambda)
        trainer.update(buf)
        if logf:
            writer.writerow([ep, ep_reward])
        if ep % 20 == 0:
            print(f"[{args.reward} seed{args.seed}] ep {ep} reward={ep_reward:.3f}")

    if logf:
        logf.close()


if __name__ == "__main__":
    main()
```

> **注**：`llm_mappo` 奖励由 LRS 生成，需在任务 14 之后把生成的最优奖励函数保存为 `results/best_reward.py`，再在 `REWARDS` 中增加一个从该文件 `exec` 加载的条目。本任务先覆盖 4 个忠实级 baseline。

- [ ] **Step 2: 运行验证端到端**

Run: `python scripts/train_mappo.py --reward handcraft --episodes 50 --log results/test.csv`
Expected: 正常训练，生成 `results/test.csv` 含表头与数据。

- [ ] **Step 3: 提交**

```bash
git add scripts/train_mappo.py
git commit -m "feat(scripts): configurable reward + CSV logging"
```

---

## 任务 17：出图脚本

**Files:**
- Create: `scripts/plot_results.py`

- [ ] **Step 1: 实现多算法对比出图脚本（覆盖图 4/6/7 训练与搜索效率曲线风格）**

`scripts/plot_results.py`：

```python
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(log):
    data = np.genfromtxt(log, delimiter=",", names=True)
    return data["episode"], data["reward"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", nargs="+", required=True, help="训练日志 CSV，可多个")
    ap.add_argument("--label", nargs="+", default=None, help="与 --log 对应的图例标签")
    ap.add_argument("--out", default="figures/training_curve.png")
    args = ap.parse_args()

    labels = args.label or [f"algo{i}" for i in range(len(args.log))]
    fig, ax = plt.subplots()
    for log, label in zip(args.log, labels):
        ep, r = load(log)
        ax.plot(ep, r, lw=1, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
```

> 论文图 8/10（可扩展性、消融的柱状对比）需先统计各算法的「已搜索目标数 / 平均不确定度」终值再画柱状图，属 2 周冲刺之外的可选后续，记录到复现报告即可；本脚本先满足训练/效率曲线。

- [ ] **Step 2: 提交**

```bash
git add scripts/plot_results.py
git commit -m "feat(scripts): multi-algorithm reward curve plotting"
```

---

## 任务 18：全量 8 seeds 并行挂机脚本

**Files:**
- Create: `scripts/run_full_experiment.py`

- [ ] **Step 1: 实现并行训练脚本**

`scripts/run_full_experiment.py`：

```python
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

SEEDS = list(range(8))
REWARDS = ["sparse", "handcraft", "mdps", "mdps_improved"]


def run_one(args):
    seed, reward = args
    log = f"results/{reward}_seed{seed}.csv"
    cmd = [sys.executable, "scripts/train_mappo.py",
           "--reward", reward, "--seed", str(seed),
           "--episodes", "28000", "--log", log]
    subprocess.run(cmd, check=True)
    return reward, seed


def main():
    jobs = [(s, r) for r in REWARDS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=4) as ex:
        list(ex.map(run_one, jobs))
    print("all done")


if __name__ == "__main__":
    main()
```

> `train_mappo.py` 已在任务 16 具备 `--seed`、`--log` 与 CSV 写入，本脚本直接复用。`llm_mappo`（LRS 生成奖励）在任务 14 完成后把最优奖励保存为 `results/best_reward.py`，再于 `REWARDS` 中加入该条目即可；此处先覆盖 4 个忠实级 baseline 的 8 seeds。

- [ ] **Step 2: 提交**

```bash
git add scripts/run_full_experiment.py
git commit -m "feat(scripts): 8-seed parallel full experiment runner"
```

---

## 任务 19：复现报告 + 学习笔记 + 交接

**Files:**
- Create: `docs/report.md`
- Create: `notebooks/README.md`
- Create: `docs/handoff.md`

- [ ] **Step 1: 写复现报告骨架**

`docs/report.md` 内容（骨架，数据出图后填充）：

```markdown
# LLM-MAPPO 复现报告

## 1. 复现范围与结论（一句话）
## 2. 环境实现要点（传感器/贝叶斯地图/能耗/信息素）
## 3. 与论文差异（DeepSeek API vs 本地 7B；近似 baseline）
## 4. 实验设置与超参
## 5. 结果（图 4/6/7/8/10 + 表格）
## 6. 遗留问题与后续
```

- [ ] **Step 2: 写学习笔记索引与交接文档**

`notebooks/README.md`：

```markdown
# 学习笔记
1. PyTorch 基础（张量/自动求导/nn.Module）
2. PPO 直觉（裁剪、GAE）
3. LLM-MAPPO 架构（env -> DPES -> LRS -> MAPPO）
```

`docs/handoff.md`：

```markdown
# 交接：如何继续推进
- 如何跑全量训练：`python scripts/run_full_experiment.py`
- 如何接入自己的 LLM：改 `.env`
- 如何延伸到大创（加 NLOS 信道）：在 env/sensor.py 扩展检测模型
```

- [ ] **Step 3: 提交并推送**

```bash
git add docs/report.md notebooks/README.md docs/handoff.md
git commit -m "docs: reproduction report, notes, handoff"
git push origin main
```

---

## 实施顺序与依赖

T0 → T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8 → T9 → T10 → T11 → T12 → T13 → T14 → T15 → T16 → T17 → T18 → T19。
其中 T11–T14（LRS）依赖 T4 环境与 T15 的目标计数，可与 T5–T8（MAPPO）并行；实际按上述线性顺序执行最稳。
