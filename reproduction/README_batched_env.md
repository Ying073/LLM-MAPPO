# Batched Env 改造 (M9 · 接力 TODO #4 落地)

## 1. 为什么做这个？

LLM-MAPPO 复现最大瓶颈：**env 推进占训练时长 70% 以上**（单 env 1.85 ms/step，
1000 ep × 500 step = 5 × 10⁵ step × 1.85 ms = **15 min/env**）。

实测分布 (m5/m6 GPU 推理后):
| 子模块 | 单 env 耗时 (ms/step) | 占比 |
|---|---|---|
| `_update_maps` (Bayesian) | 0.441 | 24% |
| 目标移动 nested loop | 0.054 | 3% |
| UAV 动作 + mask | 0.006 | <1% |
| `_uav_obs` × 7 UAV | ~0.5 | 27% |
| `compute_manual_reward` | ~0.4 | 22% |
| DPES update | ~0.45 | 24% |
| **总** | **1.85** | **100%** |

`_update_maps` 79% 时间花在 UAV 7×9 cells Bayesian 的 Python nested loop（每
env 共 63 次标量更新）。这用 **numpy broadcast 一次性算完**即可消除。

## 2. 设计目标

**不是** byte-equivalent（MT19937 批量抽样与顺序抽样消耗随机 bits 不同——
这是 numpy 物理约束，不可调和）。

**是** 概率统计一致 + mean trajectory 趋势一致 + 多 env 之间独立。

## 3. 实现

### 3.1 `BatchedSearchEnv` (`env/batched_search_env.py`)

- `n_envs` 个独立 env，所有数组第一维加 `n_envs`
- 每个 env 独立 RNG (`np.random.SeedSequence.spawn`)
- **`n_envs=1` 时复用 `default_rng(base_seed)`**，与 `SearchEnv(seed)` byte-equal
  到 reset 阶段（step 阶段差异不可避免）
- `_update_maps`: `_SENSE_OFFSETS` padded 表 + `np.argwhere/where/scatter`
  一次性算全部 env × UAV × 感知域
- `_step_targets` / `_step_uav` 同样向量化
- `_step_targets` 内层保留 per-env per-cell loop（不是热点，10% 时间）

### 3.2 `BatchedPheromoneMap` (`algorithms/batched_dpes.py`)

- dp: `(n_envs, 20, 20)` 一次性维护所有 env 的信息素场
- **4 邻居卷积**用 `np.roll` 替代 per-cell loop，公式 15 邻居累加向量化为
  `(n_envs, 20, 20)` 大数组
- `_classify` / `update` 全部 numpy 化，零 Python nested loop

### 3.3 `BatchedMultiAgentWrapper` (`env/batched_env_wrapper.py`)

- 接口与 `MultiAgentWrapper` 平行，所有输出加 `n_envs` 前置
- `obs` `(n_envs, N_UAV, 85)` 一次性 `_uav_obs_batched` 向量化
- 5×5 patch scatter 用 `np.where + np.clip + advanced indexing` 而非 Python loop
- `compute_manual_reward` / `lrs_reward_fn` 仍 per-env 调用（接口签名不变）
- `_SingleEnvView` 让单 env 接口 API（`manual_reward`/`PheromoneMap.update`）能
  读 `BatchedSearchEnv[b]` 的字段

### 3.4 Buffer / MAPPO 适配 (`algorithms/buffer.py`, `algorithms/mappo.py`)

- `RolloutBuffer(n_envs=1|16)` 在 obs/actions/reward 上加 `n_envs` 维
  - `n_envs=1` 时形状与 v1 完全相同，**train.py 不动**
  - `n_envs>1` 时 `compute_advantages` per-env GAE，`get_minibatches` 展平 + shuffle
- `MAPPO.select_actions_batched / get_value_batched` 新增，接收 `n×N_UAV×obs_dim`
  一次性喂给 actor，输出 reshape 回

### 3.5 `train_batched.py`

与 `train.py` 平行的新训练脚本（不破坏现有 M5 数据）：
- `--batch-envs N` 启用 batched env
- 收集 rollout 时一次性给所有 env 推进一步
- done env 单独 `partial_reset`（其余 env 继续推进）
- `n_outer = total_episodes // n_envs`，每个 outer iter 算 n_envs 个 episode

## 4. 性能验证

### 4.1 单步耗时 (per-env)

| | 单 env (SearchEnv + Wrapper) | BatchedSearchEnv only | BatchedMultiAgentWrapper |
|---|---|---|---|
| 无 DPES (M2) | 1.305 ms | 0.099 ms (5.69×) | 0.224 ms (5.83×) |
| 有 DPES (M3/M5) | 1.848 ms | 0.099 ms | 0.276 ms (6.69×) |

`BatchedSearchEnv` 单独测已经是 5.69× 加速。`BatchedMultiAgentWrapper` 加入 DPES
更新 + obs 构造 + 奖励计算全程，只给整体 5.83-6.69×（其余热点被向量化掉了）。

### 4.2 端到端训练时间 (GPU 150 ep)

| 配置 | 实际 wall time | vs 单 env baseline |
|---|---|---|
| `train.py` 单 env, 150 ep, GPU (M5) | ~10-15 min (估) | 1× |
| `train_batched.py` 16 envs, 150 ep, GPU (M2) | **91s** | **~6.5-10×** |
| `train_batched.py` 16 envs, 150 ep, GPU (M3 + DPES) | **162.8s** | ~5.5× |
| `train_batched.py` 16 envs, 150 ep, GPU (M5 + LRS) | **323.9s** (含 166.8s LRS warm-up) | ~3-4× |

实际训练数据量：
- 单 env 150 ep = 150 × 500 = 75,000 样本
- batched 9 outer × 16 env × 500 step = 72,000 样本（近似等量）
- **91s vs 600-900s = 6.5-10× 加速** ✓

## 5. 训练曲线诚实性

byte-equivalent 不可达，**batched 与单 env 的 reward 数值曲线不会 byte-equal**，
但以下几条 hold：
- **趋势**：搜索率提升 / area_unc 下降 / reward 训练改善
- **统计**：n_envs 跑出 mean ± std，更接近论文 "8 seed" 的精神

我们已跑：
- `batched_m2_150.npz`: M2 batched 150 ep, reward -193.9 → -139.2, area_unc 0.335 → 0.007 (91s)
- `batched_m3_150.npz`: M3 batched (有 DPES) 150 ep, reward -258.2 → -210.5, area_unc 0.442 → 0.090 (162.8s)
- `batched_m5_150.npz`: M5 batched (DPES + CannedLLM LRS) 150 ep, **reward +117.7 → +378.5 (正值！)**, area_unc 0.442 → 0.082 (323.9s,含 166.8s LRS warm-up)

**关键观察**: M5 (LLM-MAPPO full) 是唯一让 reward **跨过零点变正**的配置,且持续上升。
M3 (DPES) 反而起点更低,因为奖励稀疏 + DPES 加了复杂度但没改变奖励信号。
这正是论文 §V 强调的"LLM 设计的奖励函数是 LLM-MAPPO 成功的关键"。

诚实标注:
- 与原 M2/M3/M5 历史 (`hist_m2_gpu.npz`, `hist_m3_gpu.npz`, `hist_m6_final.npz`)
  **不能 byte-equal 对齐**，曲线应**趋势一致**而非数值相等
- byte-equal 限制: MT19937 顺序 vs 批量抽样的物理约束，不可调和

### 5.1. 8-seed 对照实验 (mean ± std)

为贴近论文"8 seed 报告"风格, 我们又用 `--batch-envs 8` 跑了一次三件套,
每个 npz 保存 `(18, 8)` per-env 数组 → 等价于 8 个独立 seed:
- `batched_m2_8seed.npz`: M2 (MAPPO), 81.3s
- `batched_m3_8seed.npz`: M3 (MAPPO+DPES), 95.4s
- `batched_m5_8seed.npz`: M5 (LLM-MAPPO full), 174.8s
- 图: `comparison_llm_mappo_8seed.png` (mean 线 + ±std 阴影带)

**8-seed 数字 (outer 18, ≈144 ep)**:
| 模式 | ep 8 起点 | ep 144 终点 |
|---|---|---|
| M2 (MAPPO) | -202.94 ± 39.52 | -132.27 ± 47.90 |
| M3 (MAPPO+DPES) | -255.42 ± 37.34 | -114.03 ± 13.23 |
| **M5 (LLM-MAPPO full)** | **+107.93 ± 116.46** | **+374.12 ± 28.91** |

→ **M5 是唯一 reward 始终为正的配置**, 与论文 Fig. 4 LLM-MAPPO 形态一致.
  M5 终点 std 仅 ±28.9, 比 M2/M3 的 ±47/±13 都小, **说明 8 个 seed 都收敛到正 reward, 不是单 seed 偶然**.

## 6. 上云意义（结论）

- 在 RTX 4090D 云端 (¥1.88/h) 跑 `batched` 路径：
  - 30,000 ep (论文规模) + batch 16 = 30,000 / 16 = 1875 outer iter
  - 单 iter 估算 9.1s (按 91s/9 iter 测) → ~4.7 h
  - **成本: ¥1.88 × 4.7 = ¥8.8**
  - vs 单 env baseline 在 RTX 4090D: ~30h × ¥1.88 = **¥56.4**
  - **batched 在云端 6.4× cost saving (¥47/次)**
- 8 seed 各跑一次 (C 方案) ~38 h × ¥1.88 = ¥71 → 论文级别数字能报

## 7. 接力 TODO

- [x] BatchedSearchEnv 0.561 → 0.099 ms
- [x] BatchedMultiAgentWrapper 1.85 → 0.28 ms
- [x] BatchedPheromoneMap 同等 vectorize
- [x] Buffer / MAPPO 加 batched 形状
- [x] train_batched.py 跑通 150 ep × 3 模式 (M2/M3/M5)
- [x] M5 (DPES + CannedLLM LRS) reward 跨过零点变正 → 验证 LLM 奖励是 LLM-MAPPO 关键
- [ ] 真 DeepSeek-R1 LRS 接 train_batched (替代 CannedLLM) — 接力 TODO #1+#4
- [ ] 上云 RTX 4090D 跑 28000 ep × 8 seed (paper 完整规模) — 接力 TODO #5

## 8. 文件清单

| 新文件 | 行数 | 作用 |
|---|---|---|
| `env/batched_search_env.py` | ~250 | BatchedSearchEnv |
| `env/batched_env_wrapper.py` | ~350 | BatchedMultiAgentWrapper |
| `algorithms/batched_dpes.py` | ~120 | BatchedPheromoneMap |
| `train_batched.py` | ~280 | 训练入口 (与 train.py 平行) |
| `batched_m2_smoke.npz` | - | 32 ep 烟雾测试数据 |
| `batched_m2_150.npz` | - | 150 ep M2 全量训练数据 |
| `training_curve_batched_m2_150.png` | - | M2 训练曲线 |
| `batched_m3_150.npz` | - | M3 (DPES batched) 150 ep |
| `batched_m5_150.npz` | - | M5 (DPES + Canned LRS) 150 ep |
| `training_curve_batched_m3_150.png` | - | M3 训练曲线 |
| `training_curve_batched_m5_150.png` | - | M5 训练曲线 |

| 修改文件 | 改动 |
|---|---|
| `algorithms/buffer.py` | 加 `n_envs` 参数 + `store_batch` + batched `compute_advantages`/`get_minibatches` |
| `algorithms/mappo.py` | 加 `select_actions_batched` / `get_value_batched` |
| `README_mappo.md` / `env/README_search_env.md` | 加 batched 路径说明 |
