# LLM-MAPPO 论文阅读与复现

复现论文：**Multi-UAV Trajectory Planning for Dynamic Target Search: An LLM-Enhanced Multi-Agent Reinforcement Learning Algorithm**（IEEE TCCN, 2026）

> 一句话：用大模型（LLM）设计奖励函数 + 多智能体 PPO（MAPPO）+ 双模式信息素（DPES），让 7 架无人机在 3D 环境里协同搜索移动目标，搜索时间相比手工设计奖励缩短 71.4%。

## 项目结构

| 路径 | 内容 |
|---|---|
| `LLM-MAPPO_Markdown_Reader/` | 论文中英对照精读（正文、公式索引、图表裁图） |
| `reproduction/` | 复现代码（仿真环境 + MAPPO + DPES + LLM 奖励塑形） |
| `HOW_TO_RESUME.md` | **接力指南**（10 分钟上手 + 雷区清单 + 接力 TODO） |
| `reproduction/README_compare_with_paper.md` | **先读这个**（与论文的诚实差距分析） |
| `hist_*.npz` | 训练 raw history（rewards/searched/au/actor_loss/critic_loss）|
| `training_curve_*.png` / `comparison_*.png` | 9 张训练曲线 + 2 张消融对比图 |
| `Multi-UAV_..._Algorithm.pdf` | 论文原文 |

## 复现路线

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 环境准备（conda + GPU PyTorch） | ✅ |
| M1 | 仿真环境 `reproduction/env/search_env.py` | ✅ |
| M2 | MAPPO 基线 + 手写稠密奖励（v2 修复 credit assignment） | ✅ |
| M3 | DPES 双模式信息素（公式 13–17）+ A/B 训练 | ✅ |
| M4 | LRS 离线 LLM 奖励塑形（公式 20–22, 27–29） | ✅ K=5 闭环验证 η_k 单调 |
| M5 | LLM-MAPPO 端到端（DPES + LRS + MAPPO 联合 150 ep）| ✅ |
| M6 | GPU 化（conda + RTX 5060 + minibatch 256） | ✅ |
| M7 | 三件套消融对比（M2/M3/M5 同 seed 150 ep）+ 与论文差距分析 | ✅ |
| M8 | 接力指南（HOW_TO_RESUME.md）| ✅ |

详细的里程碑任务书见 [`reproduction/README.md`](reproduction/README.md)。

## 快速开始

```bash
# 1. 用已有 conda llm_mappo 环境（已装 torch+CUDA + numpy + matplotlib）
"C:/Users/lenovo/anaconda3/envs/llm_mappo/python.exe"

# 2. 跑通 M5/M6 端到端 (DPES + LRS + MAPPO, 150 ep, GPU, ~10 分钟)
cd "C:/Users/lenovo/AI/大创/LLM-MAPPO_论文阅读与复现"
python reproduction/train.py --total-episodes 150 --seed 42 \
    --use-dpes --use-lrs --lrs-K 5 \
    --device cuda --out-name run_check.png
```

> 第一次建环境？照 `reproduction/README_llm_mappo.md` §9.1 装 conda + torch+CUDA 130。
> 读懂代码？照 `HOW_TO_RESUME.md` §1.2 顺序读 6 个 README。

## 论文精读

论文的结构化精读材料在 `LLM-MAPPO_Markdown_Reader/` 目录：

- `paper.md` — 中英对照正文
- `equations.md` — 公式索引与中文说明
- 各图裁图（`*.png`）
