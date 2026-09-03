# LLM-MAPPO 论文阅读与复现

复现论文：**Multi-UAV Trajectory Planning for Dynamic Target Search: An LLM-Enhanced Multi-Agent Reinforcement Learning Algorithm**（IEEE TCCN, 2026）

> 一句话：用大模型（LLM）设计奖励函数 + 多智能体 PPO（MAPPO）+ 双模式信息素（DPES），让 7 架无人机在 3D 环境里协同搜索移动目标，搜索时间相比手工设计奖励缩短 71.4%。

## 项目结构

| 路径 | 内容 |
|---|---|
| `LLM-MAPPO_Markdown_Reader/` | 论文中英对照精读（正文、公式索引、图表裁图） |
| `reproduction/` | 复现代码（仿真环境 + MAPPO + DPES + LLM 奖励塑形） |
| `Multi-UAV_..._Algorithm.pdf` | 论文原文 |

## 复现路线

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 环境准备（conda + GPU PyTorch） | 进行中 |
| M1 | 仿真环境 `reproduction/env/search_env.py` | 待开始 |
| M2 | MAPPO 基线（集中 Critic + 分散 Actor + 动作掩码，手写奖励） | 未开始 |
| M3 | DPES 双模式信息素（公式 13–17） | 未开始 |
| M4 | LRS 离线 LLM 奖励塑形（可选） | 未开始 |

详细的里程碑任务书见 [`reproduction/README.md`](reproduction/README.md)。

## 快速开始

```bash
cd reproduction
conda create -n llm_mappo python=3.10 -y
conda activate llm_mappo
pip install numpy matplotlib   # M1 阶段只需要这两个
```

> M1 阶段只需 `numpy` + `matplotlib`；`torch` 等到 M2 训练时再装（按你的 CUDA 版本到 pytorch.org 选命令）。

## 论文精读

论文的结构化精读材料在 `LLM-MAPPO_Markdown_Reader/` 目录：

- `paper.md` — 中英对照正文
- `equations.md` — 公式索引与中文说明
- 各图裁图（`*.png`）
