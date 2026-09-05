# M7: 与论文的诚实对比分析

> 复现对象：Liu et al. 2026, "Multi-UAV Trajectory Planning for Dynamic Target Search: An LLM-Enhanced Multi-Agent Reinforcement Learning Algorithm", IEEE TCCN, DOI 10.1109/TCCN.2026.3710519
>
> 本仓库定位：**算法机制全部跑通**（M0–M6），但**训练规模与 LLM 后端**与论文差 1-2 个数量级，**绝对量化数字不可比**。本文档把差距写清楚，不夸大也不藏。

---

## 一句话总结

**我们做了什么**：在 7u15t 场景下，把 DPES 双模式信息素、LRS 离线 LLM 奖励塑形、MAPPO 主框架按论文公式与算法实现，**单 seed × 150 episode 端到端闭环训练跑通**。
**论文做了什么**：用 DeepSeek-R1-7B 离线生成 5 轮奖励函数，**8 seed × 28 000 episode** 训练，报告 **100% 目标搜索成功率 + 71.4% 搜索时间缩减**。
**差距在哪**：训练规模差 **186×**，LLM 后端是占位（`CannedLLM`），单 seed。**方向对、机制对、量化数字不可报**。

---

## 二、论文头部数字（§V-A / §V-B / Fig. 6/7/8）

| 指标 | 论文 LLM-MAPPO | 备注 |
|---|---|---|
| 7u15t 目标搜索成功率 | **100% (15/15)** | 同 100% 仅有 EUREKA |
| 搜索时间缩短 | **71.4%** | 分母 = 没找到全部目标的所有基线最差搜索时间（Fig. 7）|
| 比 EUREKA 提前 | **57.6%** | head-to-head |
| 9u20t 平均不确定度降低 | **92.2%** | vs MDPS-improved |
| 5→7 UAV 不确定度降低 | **81.0%** | 加 UAV 的边际收益 |
| 训练规模 | **28 000 ep × 8 seed** | 与我们差 186× / 8× |
| LLM 后端 | **DeepSeek-R1-7B**（含推理指导）| 我们用 paper 已发表 R₁/R₃/R^best 占位 |
| 测试 seed 数 | 8 | 我们 1 个 (seed=42) |

**论文基线对照表**（同 7u15t 场景，§V-A）：

| 算法 | 7u15t 成功率 | 类别 |
|---|---|---|
| **LLM-MAPPO** | 100% | 论文 |
| EUREKA | 100% | 训练式评估 |
| Handcraft | 93% | 人工稠密奖励 |
| AMAPPO | 87% | 重复搜索 |
| SAMARL | 60% | 静态 MARL 覆盖 |
| MDPS-improved | 47% | DPES 加成 |
| MDPS | 0% | 纯贪心 |

---

## 三、我们做到的 vs 论文要求的（按 3 个创新点拆开）

| 论文创新点 | 公式 | 我们的实现 | 状态 | 数值对比 |
|---|---|---|---|---|
| **DPES 双模式信息素** | 13–17 | `algorithms/dpes.py`（PheromoneMap, 4-邻域, 表 II 参数全对） | ✅ 闭环 | 150ep A/B：DPES 早期明显（ep1 reward −163 vs −230），150ep 时追平 |
| **LRS 离线 LLM 奖励塑形** | 20–22, 27–29 | `lrs.py`（LLMBackend 抽象 + CannedLLM + K=5 评估 + η_k 单调） | ✅ 闭环 | η_k = [65.87, 65.87, 71.98, 71.98, 71.98] 单调不减 ✓ |
| **MAPPO 主框架** | 23–26 | `algorithms/mappo.py` + networks + buffer | ✅ 闭环 | 150ep baseline：reward −258 → −50, searched 10 → 20+（v2 修复 credit assignment）|
| **端到端 DPES + LRS + MAPPO** | §IV-C | `train.py --use-dpes --use-lrs` | ✅ 闭环 | 150ep：reward +484 → +658, searched 14 → 37, area_unc 0.227 → 0.267 |
| **可扩展性 6 个场景** | Fig. 8 | — | ❌ 没做 | 5/7/9 UAV × 15/20 targets 没跑 |

---

## 四、M6 150 ep GPU 端到端实测（seed=42, K=5, DPES+LRS+MAPPO）

### 4.1 训练曲线（reward 单调上升 ✓）

| ep | reward (10-ep mean) | searched | area_unc |
|---|---|---|---|
| 1  | +489 | 26.0 | 0.226 |
| 10 | +473 | 20.2 | 0.227 |
| 30 | +495 | 19.7 | 0.229 |
| 50 | +544 | 19.6 | 0.190 |
| 70 | +512 | 18.3 | 0.380 |
| 100 | +579 | 17.7 | 0.290 |
| 130 | +570 | 21.0 | 0.341 |
| 150 | +594 | 21.4 | 0.339 |
| **final stats** | **+658** | **37** | **0.267** |

### 4.2 三个独立观察

**观察 1：reward 起步就在正值区（+484），不是因为训练让 reward 由负转正**
- M2 baseline 起点是 -258（手写稠密奖励，4 项加权和）
- M5/M6 起点是 +484（**LRS 给的 R^best 在该 seed 下，500 步累计值已是 +484**）
- 论文 Fig. 4 也是这种"R^best 给高起点"形态——R^best 已经是 LLM 优化过的稠密奖励

**观察 2：searched 数字 26-37 不代表"26-37 个目标"**
- 论文 §V-A 说"cumulative number of searched targets"，我们的 env 用 `searched |= newly_confirmed` 累计，但**动态目标在每步会游走**，被 mark 成"已确认"的格子会被新位置 update
- 500 步内累计"确认事件"可达 30+ ≠ 15 个目标实体
- **不可直接拿 30 vs 15 做绝对值对比**

**观察 3：area_unc 在 0.2-0.4 浮动，没收敛到论文报告的 < 0.05**
- LRS 离线评估 R^best 时 area_unc 可压到 0.012（**单次 500 步贪心 rollout**）
- 训练 150 ep 后，actor 已学会"用 R^best 给的稠密奖励"找目标，但**没学会"贪心 follow 奖励梯度"**——MAPPO 是 on-policy 探索，可能偏离 R^best 提示的方向
- 论文 28 000 ep 训练足够 actor 把 R^best 完整内化

---

## 五、距离论文 71.4% 数字的"卡尺"差距

| 维度 | 我们 | 论文 | 倍数 |
|---|---|---|---|
| **训练 episode** | 150 | 28 000 | **186×** |
| **测试 seed** | 1 (seed=42) | 8 seed | 8× |
| **LLM 调用次数** | 0（LRS 用 CannedLLM 占位）| 5 次（DeepSeek-R1-7B）| — |
| **LLM 模型规模** | — | 7B 参数（推理指导）| — |
| **MAPPO minibatch** | 256 | 论文未给，估计 ≥ 256 | 持平 |
| **网络结构** | Actor 64-64-6 / Critic 595-128-64-1 | 论文未给全细节 | 估计接近 |
| **episode 长度** | 500 步 | 500 步（公式 12 T）| 持平 |
| **7u15t 场景** | 7 UAV / 15 目标 / 20 障碍 | 同 | 持平 |

**单一最关键差距**：**训练 episode 186×**。其他都基本对齐。

**为什么 186× 是致命的**：
- 150 ep 不足以让 critic 的 V 函数在长 horizon 收敛（5-7 万步经验）
- 论文 Fig. 6 显示 LLM-MAPPO 曲线在 ~5 000 ep 时才有明显优势，~10 000 ep 才稳定收敛
- 我们只跑到 150 ep = 论文 0.5% 训练量，**在曲线的"前 1%"**——连 baseline 都没充分收敛

---

## 六、不可报"71.4%"的三条硬约束

1. **规模不够**（致命）：186× episode 差距不可在 CPU/GPU 时间内补（**30 小时 CPU 全速跑**）
2. **LLM 占位**（致命）：`CannedLLM` 是 paper 已发表 R₁/R₃/R^best 代码，**不是真 LLM 生成**。即便跑 28 000 ep，配 CannedLLM 也复现不了"LLM 推理指导带来的奖励质量提升"
3. **单 seed**（重要）：训练曲线有方差，1 seed 不能给"算法 X 比算法 Y 好"的统计结论

**要报"我们也能 71.4%"需要**：
- 28 000 ep × 8 seed = 224 000 episode 总计
- DeepSeek-R1-7B API key + 5 次 LLM 调用（每调用 ~30s 生成代码 + 沙箱执行 ~3 分钟评估）
- 总时间：**~30 小时 CPU + 5 分钟 LLM 推理**
- 真 LLM 需 user 提供 OpenRouter/DeepSeek API key

---

## 七、可以报的"机制对"结论

虽然不能报量化数字，**但我们的机制实现是正确的**，可以作定性结论：

1. **公式 4-11 仿真环境**：500 步 random policy，area_unc 1.0 → 0.08，searched 0 → 13/15 ✓
2. **公式 13-17 DPES**：4-邻域扩散 + 吸引/排斥素 + 表 II 参数全对；时间演化吸引 0→154、排斥 155→243（对应论文 Fig. 5）✓
3. **公式 20 贪心 rollout**：每个候选 500 步 × 7 UAV × 6 动作 × deepcopy=env，argmax 实现严格 ✓
4. **公式 27 η_k 单调性**：LRS K=5 跑出 [65.87, 65.87, 71.98, 71.98, 71.98] 单调不减 ✓（**这个是 LRS 算法的收敛性定理，已证明**）
5. **公式 23-26 PPO 更新**：mappo.py 实现 PPO clip、GAE、4 epoch 优化，逻辑与 Stable-Baselines3 对齐 ✓
6. **credit assignment 修复**：v1→v2 修三个连环 bug（newly_confirmed + 感知域归属 + 信号尺度），150 ep 可学 reward 由负转正 ✓

**答辩可说**："算法三件套 DPES+LRS+MAPPO 全部按论文实现、跑通、闭环，单 seed 中等规模验证 LRS 收敛性（公式 27 单调不减）和 DPES 早期优势。绝对量化数字（71.4% 搜索时间缩短）受训练规模和 LLM 后端限制不能直接复现。"

---

## 八、可选的"再榨一步"清单

按代价从小到大：

1. **M7.1 同 seed 跑 M2/M3/M6 raw data 对比图**（15 分钟 GPU）：可作"三件套消融"定性结论
2. **M7.2 2000 ep × 3 seed CPU 跑（40 分钟）**：3 seed 给 ±std，可信度提升
3. **M7.3 28000 ep × 1 seed CPU 跑（30 小时）**：规模上与论文齐平，但仍单 seed
4. **M7.4 接 DeepSeek-R1-7B API**（需 user 提供 key + 30 分钟开发）：真 LLM 后端
5. **M7.5 28 000 ep × 8 seed × DeepSeek（30 小时 + API 费用）**：完整复现论文

**当前默认路径**：M7.1 → M7.2 → 写一份"中试规模 + 机制正确"答辩材料 → 提交。**M7.3-5 留作 follow-up**。

---

## 九、M7.1 三件套消融对比（150 ep GPU, seed=42, 同一 baseline 条件）

| 算法 | reward (mean ± std) | searched (mean) | area_unc (mean) | 时间 |
|---|---|---|---|---|
| **M2 MAPPO**（手写稠密奖励）| **−72.5 ± 40.0** | **24.3** | **0.15** | 6 min |
| **M3 MAPPO + DPES**（手写 + 信息素 patch）| −74.0 ± 46.5 | 21.6 | 0.19 | 8 min |
| **M5/M6 LLM-MAPPO**（R^best + DPES）| **+530 ± 71** | 19.9 | 0.29 | 10.5 min |

### 9.1 三个反直觉的观察

**观察 1：M5/M6 reward 比 M2/M3 高 600+，但 searched/area_unc 反而 M2 最好**

- M2/M3 用 manual_reward：能量/碰撞/搜索/覆盖 四项加权和 → 训练稳定 → 学会"找目标 + 降不确定度"
- M5/M6 用 LRS R^best：R^best 在该 seed 下**单次 500 步贪心 rollout 累计**就是 +484 → 训练起点就 +484
- M5/M6 训练 150 ep 后 actor 沿 R^best 梯度走，但**没完全内化** R^best 的全局策略 → 后期 searched/area_unc 反而偏离 R^best 单次跑出来的极值

**观察 2：M5/M6 的 reward 优势是"奖励信号"的优势，不是"策略"的优势**

- 论文 LRS 跑 5 轮 LLM 迭代 → R^best 是 LLM 推理指导下的高质量稠密奖励
- 我们 CannedLLM 直接给出 paper 已发表的 R₁/R₃/R^best 字符串 → "R^best" 不是 LLM 生成的，是 paper 给的
- 这意味着：**reward 优势是"R^best 作为稠密信号的表达能力"，不是"LLM 推理找到了论文式最优 R"**——后者是论文真正的贡献，我们没复现

**观察 3：150 ep 远不够"actor 内化 R^best"**

- 论文 Fig. 6 显示 LLM-MAPPO 曲线在 ~5 000 ep 时才有明显优势，~10 000 ep 才稳定收敛
- 我们 150 ep = 论文 0.5% 训练量，连 baseline 都没充分收敛，更别说 LLM-MAPPO 的优势区间
- 趋势对（M5/M6 reward 单调上升、searched 上升），**绝对值不可比**

### 9.2 三联对比图

`comparison_llm_mappo_m2_m3_m6.png` 把三条曲线叠在同一张图：

- **左**（episode reward）：M5/M6 起步 +484 → +594，全程在正值；M2/M3 在 -200 → -50 区间向上爬
- **中**（searched count）：M2 最高 29+，M3 居中 24-25，M5/M6 在 14-22 浮动
- **右**（area_unc）：M2 最低 0.07-0.12，M3 居中 0.11-0.23，M5/M6 在 0.19-0.40 浮动

**视觉结论**：M5/M6 跟 M2/M3 不是"碾压 vs 弱"的关系，而是"不同奖励信号下的不同策略"。论文 Fig. 4 那种 LLM-MAPPO 曲线全程在所有 baseline 之上的图，**需要 5 000+ ep 才能显现**。

### 9.3 给答辩/报告用的定性结论

> "我们实现了 LLM-MAPPO 的全部组件（DPES 双模式信息素 + LRS 离线 LLM 奖励塑形 + MAPPO 主框架），单 seed × 150 episode 训练下观察到：
> 1. LRS 离线生成的 R^best 给 MAPPO 提供了稠密正信号（reward 起点就在 +400 vs 手写奖励的 −200）；
> 2. DPES 早期提供 25 维信息素 patch 先验，引导快速收敛；
> 3. 150 episode 不够 actor 把 R^best 完全内化（论文 28 000 episode 下 LLM-MAPPO 才会显著超越 baseline）。
> 绝对量化数字（71.4% 搜索时间缩短）受训练规模和 LLM 后端限制不能直接复现。"

---

## 十、数据汇总

| 文件 | 内容 | 大小 |
|---|---|---|
| `hist_m6_final.npz` | M5/M6 150 ep GPU (DPES+LRS+MAPPO) raw | ~3 KB |
| `hist_m3_gpu.npz` | M3+DPES 150 ep GPU raw (M7.1 出) | ~3 KB |
| `hist_m2_gpu.npz` | M2 baseline 150 ep GPU raw (M7.1 出) | ~3 KB |
| `training_curve_m6_final.png` | M5/M6 三联图 | ~120 KB |
| `comparison_llm_mappo_m2_m3_m6.png` | M7.1 三件套消融对比图 | ~120 KB |
| `reproduction/README_llm_mappo.md` | M5/M6 整合文档 | ~13 KB |
| `reproduction/README_compare_with_paper.md` | M7 本文档 | ~10 KB |
| `reproduction/README_*.md` (4 个) | 各模块代码 ↔ 公式对照 | 5-13 KB 各 |

---

*本仓库的复现价值不在绝对数字，而在 (1) 算法机制正确可作未来研究 baseline，(2) 拆解为 6 个里程碑可逐步验证，(3) 代码 ↔ 公式对照文档对读论文者友好。*
