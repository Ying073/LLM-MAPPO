# LLM-MAPPO 论文复现 — 设计文档

- **日期**：2026-08-31
- **目标论文**：Multi-UAV Trajectory Planning for Dynamic Target Search: An LLM-Enhanced Multi-Agent Reinforcement Learning Algorithm（IEEE TCCN 2026，DOI 10.1109/TCCN.2026.3710519）
- **仓库**：github.com/Ying073/LLM-MAPPO
- **周期**：2 周冲刺（代码主体 ~10 天 + 训练挂机）

## 1. 目标与约束

**目标**：完整复现 LLM-MAPPO，交付「完整正确可运行的代码 + 缩减规模端到端验证 + 全量训练启动脚本 + 简版复现报告」。

**约束**（来自用户确认）：
- 硬件：NVIDIA 独显 ≥8GB
- LLM：DeepSeek API（`deepseek-reasoner`）
- 基础：RL / PyTorch 基本未接触 → 边做边学，我写 80% 代码，用户运行 + 提问 + 改参数
- 节奏：越快越好，1–2 周上限

## 2. 算法拆解（实现范围）

### 2.1 仿真环境（env）
- 20×20 网格（2000m×2000m）、20 静态障碍物、15 动态目标（1 m/s 随机方向）、7 UAV
- 传感器模型 Eq.(4)(5)：感知半径/检测概率/虚警概率随高度变化（3 档高度 50/100/150m）
- LTPM/LEUM 贝叶斯更新 Eq.(6)(7)、GTPM/GEUM 融合 Eq.(8)–(10)
- 能耗模型 Eq.(1)–(3)（返航能量约束 Eq.12f）
- 动作空间：{北,东,南,西,升,降}；Gymnasium 风格接口
- **向量化 NumPy 实现**（7 UAV 批量推进），并做每秒 steps 性能基准

### 2.2 MAPPO
- 集中式 Critic + 分散式 Actor，2 层×64 全连接 ReLU，lr 0.0002，γ 0.95
- PPO 裁剪目标 Eq.(23)(24)、Critic 更新 Eq.(25)(26)、GAE 优势估计
- 动作掩码：过滤不安全动作（防撞）
- 先单智能体 PPO 在搜索环境 sanity check，再多智能体化

### 2.3 DPES 双模式信息素
- 网格分类 Eq.(13)：高价值/长期未访/已确认/其他
- 吸引信息素 Eq.(14)(15)（蒸发+扩散）、长期未访吸引 Eq.(16)、排斥信息素 Eq.(17)
- 参数（表 II）：重访阈值 200，蒸发/扩散系数 0.1，释放量 0.1/0.003/0.2
- 信息素 `DP_n(t)` 作为 Actor 输入

### 2.4 LRS 离线 LLM 奖励塑形
- DeepSeek API 接入，prompt 模板转录自论文图 11
- 初始化阶段 + 迭代优化阶段，K=5
- 免训练评估：固定初始观测，贪心 rollout（Eq.20），原始目标评分（Eq.21），反馈 prompt（Eq.22）
- 输出模板约束 + JSON 解析 + 失败重试

## 3. 里程碑（2 周）

| 阶段 | 天数 | 内容 | 交付物 |
|---|---|---|---|
| Week 1 | D1–2 | 环境搭建 + git + 仿真环境 | 可运行的 env + 单测 |
| | D3–4 | PPO → MAPPO + 动作掩码 | sanity check 收敛 |
| | D5 | DPES + LRS | 信息素单测 + LRS 闭环跑通 |
| Week 2 | D6–8 | 端到端 LLM-MAPPO（1 seed 缩减 episodes）+ 忠实级 baselines | pipeline 验证 |
| | D9–10 | 出图脚本 + 8 seeds 全量训练挂机启动 | 图 4/6/7/8/10 脚本 |
| | D11–12 | 复现报告 + 笔记 + 交接 | 报告框架 |

## 4. 代码结构

```
llm_mappo/
  env/  dpes/  mappo/  lrs/  baselines/  utils/
experiments/  scripts/  tests/  notebooks/  docs/
```

技术栈：Python 3.10+、PyTorch (CUDA)、NumPy、Gymnasium 接口、Matplotlib、openai 兼容 SDK（调 DeepSeek）。

## 5. 忠实度策略

- **忠实实现**：LLM-MAPPO、MAPPO（稀疏）、Handcraft、MDPS、MDPS-improved、LLM-MAPPO-OG/-OI
- **近似实现（可选后续）**：SAMARL[15]、AMAPPO[9]、EUREKA[23]
- **明确差异**：本地 DeepSeek-R1-7B → DeepSeek API `deepseek-reasoner`（README 已注明）

## 6. 学习机制（压缩版）

3 节 15 分钟快讲，穿插在运行代码间隙：
1. PyTorch 基础（张量/自动求导/nn.Module）
2. PPO 直觉（裁剪、GAE）
3. LLM-MAPPO 整体架构（env → DPES → LRS → MAPPO 数据流）

产出：复现报告 + RL 学习笔记（保研面试可用）。

## 7. 验收标准

1. 环境物理公式（传感器/贝叶斯更新/信息素）有单元测试且通过
2. LLM-MAPPO 端到端训练在缩减规模下收敛（搜索目标数上升、不确定度下降）
3. 忠实级 baseline 可跑，出图脚本生成图 4/6/7/8/10 风格曲线
4. 全量训练脚本可一键启动 8 seeds
5. 复现报告如实记录方法与论文的差异

## 8. 风险

1. **训练时长**：环境必须向量化；单 seed 目标 ≤ 数小时，8 seeds 并行一晚
2. **课业冲突（GPA 底线）**：里程碑独立可暂停，检查点可恢复
3. **LLM 输出不稳定**：输出模板约束 + 解析 + 重试
4. **用户零基础**：我写框架 + 快讲 + 理解检验，避免"抄代码不动脑"
