# LLM-MAPPO 复现

复现论文：**Multi-UAV Trajectory Planning for Dynamic Target Search: An LLM-Enhanced Multi-Agent Reinforcement Learning Algorithm**（IEEE TCCN, 2026）

> 目标：先跑通核心算法（**MAPPO + DPES 双模式信息素**），LLM 奖励塑形（LRS）部分暂时用手写稠密奖励替代。

## 里程碑与状态

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 环境准备（conda + GPU PyTorch + 目录） | 进行中 |
| M1 | 仿真环境 `env/search_env.py` | 待开始 |
| M2 | MAPPO 基线（集中 Critic + 分散 Actor + 动作掩码，手写奖励） | 未开始 |
| M3 | DPES 双模式信息素（公式 13–17） | 未开始 |
| M4 | LRS 离线 LLM 奖励塑形（可选） | 未开始 |

## 目录结构

```
reproduction/
├── README.md            # 本文件
├── requirements.txt     # 依赖
├── env/
│   └── search_env.py    # M1：仿真环境
├── algorithms/          # M2/M3：MAPPO、DPES
├── reward/              # M2：手写奖励（替代 LLM）
└── configs/             # 参数配置
```

## 环境安装

```bash
# 1. 建 conda 环境（Python 3.10，与你的 CUDA 匹配的 torch）
conda create -n llm_mappo python=3.10 -y
conda activate llm_mappo

# 2. 安装 GPU 版 PyTorch（到 pytorch.org 按你的 CUDA 版本选命令）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 其余依赖
pip install -r requirements.txt
```

M1 阶段其实只需要 `numpy` 和 `matplotlib`；`torch` 是 M2 训练时才用。

## M1 任务书：仿真环境

参考论文 §III（系统模型，公式 1–11）和表 I 的参数。核心是写一个「环境」，它提供两个方法：

- `reset()` → 随机摆好障碍物/目标/UAV，返回初始观测
- `step(actions)` → 执行 7 架 UAV 的动作，移动目标，更新感知地图，返回 `(obs, reward, done, info)`

数据约定：
- 网格 `(ix, iy)`，`ix, iy ∈ [0, 20)`，每格 100 m（2000 m / 20）
- 高度只有 3 档 `z ∈ {50, 100, 150}`，用下标 `h ∈ {0, 1, 2}` 表示
- 动作 `0北 1东 2南 3西 4升 5降`

要实现的公式（原文 + 中文说明见 `LLM-MAPPO_Markdown_Reader/equations.md`）：
- 公式 4：感知域 `Φ_n(t)`，半径 `S_n = z·tan(θ/2)/L`；表 I 给出结果值 1 / 5 / 9 个网格
- 公式 5：检测概率 `P^D`、虚警概率 `P^F`（表 I：高度 50/100/150 → 0.9/0.8/0.7 与 0.1/0.2/0.3）
- 公式 6：LTPM 贝叶斯更新（三种情况：检测到 / 未检测到 / 不在感知域）
- 公式 7：LEUM = 目标概率的信息熵
- 公式 8–10：全局地图融合（每个网格取"不确定度最低"的 UAV 的估计）
- 公式 9：区域平均不确定度

**验收标准**：`python env/search_env.py` 能跑出一个演示——UAV 在动、目标在动、障碍物挡住路、区域不确定度随时间下降。能画出类似论文图 1 的俯视图。
