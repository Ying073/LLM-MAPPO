# M2 (MAPPO) 代码 ↔ 论文对照表

> 复现对象：LLM-MAPPO 论文 §IV-C/D, 公式 18–29 (paper.md S031–S083, equations.md E018–E029)。
>
> 上一阶段（M1 仿真环境）见 `env/README_search_env.md`。本阶段 M2 在 M1 的环境上接 MAPPO 训练基线，**手写稠密奖励替代 LLM 奖励塑形**（M4 再升级）。

## TL;DR —— M2 的 4 个代码模块 ↔ 论文 4 块设计

```
   论文公式 18                            论文公式 23–26
   局部观测 O_n(t)  ───►   Actor          ───►    PPO 裁剪更新
   (每 UAV 看一份, 60 维)        网络         ratio = π/π_old, clip
                  ↓
   论文公式 25                            论文公式 12a (任务目标)
   全局 state s(t)  ───►   Critic         ───►    用 v(s) 算 GAE / return
   (所有 UAV 状态拼成)          网络         再驱动 Actor 更新方向
                  ↓
   论文公式 12a 的稠密拆解
   手写奖励 R_step = R_search + R_cover + R_collision + R_energy
   ──► replace/                reward/manual_reward.py
        LRS
```

---

## 1. M2 新增的 5 个文件

```
reproduction/
├── algorithms/
│   ├── networks.py    # Actor + Critic (CTDE 范式)
│   ├── buffer.py      # RolloutBuffer + GAE 优势计算
│   └── mappo.py       # MAPPO 训练器 (PPO update 主循环)
├── env/
│   └── env_wrapper.py # 把 SearchEnv 包成多智能体接口 (公式 18/19)
├── reward/
│   └── manual_reward.py # 手写稠密奖励 (替代 LRS)
└── train.py           # 主训练脚本
```

---

## 2. 每个函数 / 类 ↔ 对应哪条公式

### 2.1 `env_wrapper.MultiAgentWrapper` ↔ **公式 18, 19**
**公式 18：**
$$ \mathcal{O}_n(t) = \{ x_n(t), y_n(t), z_n(t), \Phi_n(t), \zeta_i, p_i(t), \chi_i(t) \}_{i \in \Phi_n(t)} $$

**公式 19：**
$$ \mathcal{A}_n(t) = \{0, 1, 2, 3, 4, 5\} = \{\text{北, 东, 南, 西, 升, 降}\} $$

**实现做了什么:**
- 每架 UAV 的 60 维观测拆成 6 块：`自身(3) + 5x5 patch(50) + 障碍(3) + UAV(3) + 能量(1)`
- `step(actions)` 推 `SearchEnv` + 计算手写奖励 + 返回 `(next_obs_list, shared_reward, done, info)`
- `get_global_state()` 把所有 UAV 局部观测拼起来给 Critic

**为什么这样:**
- 5x5 patch 是"统一窗口"——不论 h 是 0/1/2，可见格子都用同一个 5x5 数组装，没看到的格子填 0（unknown）。这样 Actor 网络的输入维度固定，不会因为 UAV 升降档而改变网络结构。
- patch 同时塞 `geum` (全局不确定度) 和 `zeta`（真实目标存在），让 Actor 既能"看哪些格子搜过"又能"看哪些格子真有目标"。
- `get_global_state` 返回 `60 × N_UAV = 420` 维——给 Critic 看全貌。这是 MAPPO 的 CTDE 范式：Actor 看局部，Critic 看全局。

### 2.2 `algorithms/networks.Actor` ↔ **公式 23** 的 LHS
**做了什么:**
- `MLP(60 → 64 → 64 → 6)` + Tanh 激活
- 输出 `logits` (未经 softmax 的动作偏好)，外面套 `Categorical(logits=...)` 分布
- `get_action()` 一次性返回 `action, logp, entropy`

**为什么这样:**
- 没用独热编码，直接用 6 维 logits + Categorical 分布——这是 PPO 在离散动作上的标准做法。
- 多架 UAV **共享参数**（一个 `Actor` 类处理所有 UAV），靠 batch 维区分。
- Actor 和 Critic 都没用 BN / LN——MAPPO 原始论文里就是干这种最朴素结构能 work。

### 2.3 `algorithms/networks.Critic` ↔ **公式 25** 的 LHS
**做了什么:**
- `MLP(global_dim → 128 → 64 → 1)` + Tanh
- 输入 = 全局 state (所有 UAV 局部 obs 拼接)

**为什么这样:**
- 标量输出 → `value: (B,)` → `loss = (V(s) - R̃)²` (公式 25)
- Critic 容量比 Actor 大（128 vs 64），符合"Critic 难学、需要更大容量"的直觉。

### 2.4 `algorithms.buffer.RolloutBuffer` ↔ **GAE + PPO replay**
**做了什么:**
- 收集 `T = rollout_len` (一个完整 episode 500 步) 的轨迹
- 字段：`obs / global_s / actions / logp / reward / done / value`
- `compute_advantages(last_value)` 用 GAE 公式递归算 $A_t$ 和 $\tilde R_t$：
  $$ \delta_t = r_t + \gamma V(s_{t+1})(1-\text{done}_t) - V(s_t) $$
  $$ A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l} $$
  $$ \tilde R_t = A_t + V(s_t) $$
- `get_minibatches()` 把 `T × N` 个样本 shuffle 后按 batch 切

**为什么这里把 per-agent reward 求平均再算 GAE:**
- CritIC 输出是"团队价值"，所以训练时用 collective reward；per-agent reward 只用于网络 forward 的梯度方向。
- 每个 UAV 共享同一个优势 $A_t$，但 Actor 接收自己的观测 $O_n(t)$ 后输出独立动作概率——这是 MAPPO 的核心约束（每个 agent 看局部）。

### 2.5 `algorithms/mappo.MAPPO.update` ↔ **公式 23, 24, 25, 26** 一次性跑完
**做了什么 (一次 PPO update):**
- 取 buffer → 算 GAE → K=4 epoch + minibatch SGD:
- **公式 23:** `L_clip = E[min(r·A, clip(r, 1-ε, 1+ε)·A)]`
  - 这里 `r = exp(new_logp − old_logp)`
- **公式 24:** Adam 更新 Actor `θ ← θ + α · ∇L_clip`
- **公式 25:** `L_vf = E[(V(s) − R̃)²]`
- **公式 26:** Adam 更新 Critic `φ ← φ − α · ∇L_vf`

**为什么这样:**
- 用 on-policy：每次 rollout 完后立即用这一份数据更新 K 次，然后弃掉。
- `entropy bonus` 加 -0.01 * entropy 到 Actor loss，鼓励探索（防止策略过早坍缩到某个动作）。
- Actor / Critic 用**独立 Adam**，学习率不同 (3e-4 vs 1e-3) —— Critic 一般学得快，需要更大学习率。

### 2.6 `reward/manual_reward.compute_manual_reward` ↔ **替代 公式 20–22**
**做了什么:** 每步返回 `(shared_reward, per_agent_reward)`
- `R_search = +10 * (新确认目标数 / N_UAV) * 0.1` —— 鼓励搜到目标
- `R_cover = +1 * (prev_au - cur_au)` —— 鼓励让 area uncertainty 单调下降
- `R_collision = -2`（撞障碍动作用了但被拒）/ `-1`（升到顶/降到地）
- `R_energy = -0.05 * 升/降一步` / `-0.02 * 水平移动一步`
- `R_depleted = -0.5`（能量 < 5% 时持续扣分）

**为什么这样:**
- 论文 §IV-C 说：稀疏奖励 (公式 12a 的目标) 是 MARL 老大难问题。LLM 奖励塑形是论文的核心创新 (LRS)。M2 阶段我们还没接 LLM，所以直接按公式 12a 的拆解思路手写一份稠密奖励。
- `R_search` 用累计 searched 数 / 7 而不是"本步新增"，是因为我们没记录"上一步已确认数"——粗略但够用。
- 各项权重先给一组能跑的默认值 (W_*)；M3/M4 调参。

### 2.7 `train.py` ↔ **训练主循环**
**做了什么:**
1. `env.reset()` → 拿到 7 份初始局部观测
2. 一回合循环 500 步：选动作 → step → 存进 buffer
3. 用最后一帧的 V(s) 做 GAE bootstrap → 调 MAPPO.update
4. 画训练曲线 → `training_curve.png`

**为什么这样:**
- 一个 episode 长度 = 500 步 (`env.MAX_STEPS`)，刚好就是一个完整回合。训练时一个 episode = 一次 PPO update。
- 日志：每 10 个 episode 打一行均值（reward / searched / au_loss）。

---

## 3. 一个 episode 里的数据流

```
reset() ─► env 初始化 →
         obs[0..6] 各 60 维 (+ prev_au)
                                        ┌── Actor → actions, logp ──┐
                                        │                            │
for t in 0..499: ─────────────►─┐       │                            ▼
                                ├─ store─┤── Critic → V(s) ─┐    env.step
                                │        │                   │
                                ▼        ◄───────────────────┘
                          buffer[T][N_UAV][obs_dim]
                                │
                                ▼ (episode 末尾)
                          GAE + returns → A_t, R̃_t
                                │
                                ▼
                          MAPPO.update (K=4 epoch)
                          ┌── update Actor (公式 23)
                          └── update Critic (公式 25)
                                │
                                ▼
                          training_curve.png
```

---

## 4. 超参与决策点

| 决策点 | 默认值 | 出处 | 后续调机会 |
|---|---|---|---|
| 观测维度 (60) | 自身(3) + 5x5 patch(50) + 障碍(3) + UAV(3) + 能量(1) | 公式 18 的合理近似 | M2 试跑后看是否够 |
| 动作空间 (6 离散) | {北,东,南,西,升,降} | 公式 19、§III-D | 论文就这么定的 |
| Actor 容量 | MLP(60, 64, 64, 6) | 公式 23 通用 | 容量不够再加层 |
| Critic 容量 | MLP(420, 128, 64, 1) | 公式 25 通用 | 容量不够再加层 |
| clip ε | 0.2 | 公式 23、PPO 原论文 | 极少调 |
| γ | 0.99 | 标准 MARL | M3 |
| λ_GAE | 0.95 | 标准 GAE | M3 |
| actor_lr | 3e-4 | PPO 默认 | M3 |
| critic_lr | 1e-3 | critic 一般要更大 | M3 |
| 更新 epoch K | 4 | PPO 默认 | M3 |
| rollout_len | 500 | env.MAX_STEPS | 单 episode 一次性 update |
| 奖励权重 | W_search=10, W_cover=1, W_collision=-2, W_energy=-0.05/-0.02 | 公式 12a 的拆解 | **M3/M4 重点调** |
| 更新总 episode | 200 (这次) | 论文 28000 太长，先验证 | M3 改 28000 |

---

## 5. 跑通的标志

按这份 README 跑训练（venv 已装好 numpy + matplotlib + torch）：

```bash
"C:\Users\lenovo\.workbuddy\binaries\python\envs\llm_mappo\Scripts\python.exe" "C:\Users\lenovo\AI\大创\LLM-MAPPO_论文阅读与复现\reproduction\train.py"
```

期望看到（每 10 个 episode 一行）：

```
[init] N_UAV=7, obs_dim=60, global_dim=420, act_dim=6, rollout_len=500, device=cpu
[ep    1/  200] reward=... searched=... area_unc=... actor_loss=... critic_loss=...
[ep   10/  200] reward=... searched=... area_unc=... actor_loss=... critic_loss=...
...
[train] saved training curve to: ...\training_curve.png
```

**学习成功的标志**（M2 仅作为 baseline，没要求跑 28000 episode）：
1. `reward` 总体趋势**变正**
2. `searched` 从随机策略的 ~13 上升到 ~14–15
3. `area_unc` 终止值**下降**
4. `actor_loss` 和 `critic_loss` 应该**先降后稳**，不应爆炸

如果以上任意一条不满足，最常见的两个原因：
- 奖励权重 → `reward/manual_reward.py` 里调 W_*
- 学习率太大/太小 → `algorithms/mappo.py:MAPPO.__init__` 改 `actor_lr/critic_lr`

---

## 5b. v1 训练失败 → v2 修复（2026-09-05 踩坑记录）

**第一版训练现象**：100 episode 后 `reward` 横盘、`searched` 反而下滑、`area_unc` 从 0.15 升到 0.5——**策略在退化**。但 `critic_loss` 收敛得很快（269 → 29）。

**根因**：手动奖励的**信用分配完全错乱**，三个连环 bug：

```python
# ❌ v1 manual_reward.py（已删除）
newly = int(env.searched.sum())                         # BUG 1: 累计值，不是"本步新增"
per[n] += W_SEARCH * (newly / N_UAV) * 0.1             # BUG 2: 所有 UAV 都拿同样值
                                                        # BUG 3: *0.1 把信号压扁
```

具体后果：
1. `env.searched.sum()` 是累计确认数 → 每步都重复给已搜到的格子发奖励，actor 看不到"探索新区域"才有奖励的因果
2. UAV A 飞到 X 搜到目标，UAV B 原地不动也拿 1/7 → **没有 credit assignment**，策略梯度学不到"要去哪里"
3. `*0.1` 把信号压扁到 0.14/步，被噪音淹没

**v2 修复**（`reward/manual_reward.py` + `env/env_wrapper.py`）：
```python
# ✅ v2：因果信号 + 信用分配
newly_confirmed = cur_searched & ~prev_searched       # 本步新增
new_cells = np.argwhere(newly_confirmed)              # (k, 2)
# 每架 UAV 感知域内、本步新确认的格子数
uav_new_count[n] = |{cell ∈ UAV_n 感知域 : cell ∈ new_cells}|
per[n] += W_SEARCH * newly_n * (uav_new_count[n] / total_coverage)
```

**关键设计原则**：
- **每个奖励项都必须是"本步动作的因果贡献"**，不能有累积
- **感知域覆盖该新格子的 UAV 才分到奖励**——这是公式 18/4 的工程映射
- **覆盖奖励 rescale**：单步 Δau≈0.002，乘 W_COVER=100 → ±0.2/步，与搜索奖励同量级

---

## 6. 故意简化的地方（之后怎么升级）

| 简化 | 现在 | 论文 / 升级时做法 | 升级到 |
|---|---|---|---|
| Critic 输入 | 全局 state 拼所有 UAV obs | 可改成"局部 obs + 全局统计量 (max/min/mean geum)" | M3 |
| Reward 权重 | 拍脑袋一套 | M3 接 DPES 后用 LRS 自动优化 | M4 |
| 网络结构 | 朴素 MLP | 可加 attention 层（MAPPO 论文 variant） | M3 |
| 训练规模 | 200 episodes | 论文 28000（要 GPU） | M3 |
| Action mask | search_env 已实现"撞了别动" | M2 没传给 Actor（采样后被拒） | M3 改成策略概率=0 |
| 局部观测 patch | 固定 5x5 | 加历史帧堆叠 (t-1, t-2) | M3 |
| 信任域 | clip ε | 也可加 KL 约束 (TRPO) | — |

---

## 7. 与论文对齐的得分

| 模块 | 完成度 | 备注 |
|---|---|---|
| Actor 公式 23/24 | ✅ 100% | 公式一致 |
| Critic 公式 25/26 | ✅ 100% | 公式一致 |
| 公式 18 局部观测 | ⚠️ 70% | 实现完整但维度是工程近似 |
| 公式 19 动作空间 | ✅ 100% | 6 离散动作 |
| 公式 12a 目标 | ⚠️ 50% | 手写稠密版, M4 升级到 LRS |

下一步走 M3（DPES 双模式信息素，公式 13–17）→ M4（LRS 离线 LLM 奖励塑形）。
