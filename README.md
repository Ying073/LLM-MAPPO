# LLM-MAPPO

**LLM 增强的多无人机动态目标搜索** —— 论文复现仓库。

用大语言模型（LLM）离线生成奖励函数，引导 MAPPO 多智能体强化学习，规划多无人机协同轨迹，在三维障碍环境中高效搜索动态目标。

## 📄 论文信息

| 项 | 内容 |
|---|---|
| 标题 | Multi-UAV Trajectory Planning for Dynamic Target Search: An LLM-Enhanced Multi-Agent Reinforcement Learning Algorithm |
| 作者 | Yifei Liu, Xiaoshuai Li, Xiaoping Jiang, Hui Liu, Junan Yang（国防科技大学） |
| 期刊 | IEEE Transactions on Cognitive Communications and Networking, 2026 |
| DOI | [10.1109/TCCN.2026.3710519](https://doi.org/10.1109/TCCN.2026.3710519) |
| IEEE Xplore | [document/11595831](https://ieeexplore.ieee.org/document/11595831) |

## 🎯 算法简介

LLM-MAPPO 由三部分组成：

1. **仿真环境** — 2000m×2000m 网格化搜索区域、静态障碍物、动态目标、7 架多旋翼 UAV（3 档高度，检测概率随高度变化）
2. **DPES 双模式信息素** — 吸引/排斥信息素引导 UAV 优先搜索高价值与长期未访问区域，避免冗余搜索
3. **LRS 离线 LLM 奖励塑形** — 用 LLM 迭代生成高质量奖励函数（免训练评估），缓解 MARL 的稀疏奖励问题，再喂给 MAPPO 训练

## ✅ 复现进度

| 模块 | 状态 |
|---|---|
| 论文精读（中英对照 Reader） | ✅ 完成（见 `LLM-MAPPO_Markdown_Reader/`） |
| 仿真环境 | ⏳ 待实现 |
| PPO / MAPPO | ⏳ 待实现 |
| DPES 信息素 | ⏳ 待实现 |
| LRS 奖励塑形 | ⏳ 待实现 |
| 实验与出图（图 4/6/7/8/10） | ⏳ 待实现 |
| 复现报告 | ⏳ 待实现 |

## 📁 目录结构

```
LLM-MAPPO/
├── llm_mappo/
│   ├── env/        # 仿真环境（传感器、贝叶斯地图、目标、能耗）
│   ├── dpes/       # 双模式信息素
│   ├── mappo/      # Actor / Critic / 训练器
│   ├── lrs/        # LLM 奖励塑形（prompt、API、rollout 评估）
│   ├── baselines/  # 对比算法
│   └── utils/      # 配置、随机种子、日志
├── experiments/    # 训练配置与脚本
├── scripts/        # 出图脚本
├── tests/          # 单元测试
├── notebooks/      # 学习笔记与迷你课
├── docs/           # 复现报告与设计文档
└── LLM-MAPPO_Markdown_Reader/  # 论文精读笔记
```

## 🚀 快速开始

> ⚠️ 代码搭建中，环境依赖与运行命令稍后补充。

```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. 安装依赖（待补充 requirements.txt）
pip install -r requirements.txt

# 3. 配置 DeepSeek API key（复制 .env.example 为 .env 并填入）
cp .env.example .env

# 4. 训练与出图（命令待补充）
```

## 🔍 与论文的差异说明

为保证学术诚信，此处如实列出复现与原文的差异：

- **LLM 接入**：论文使用本地 DeepSeek-R1-7B，本仓库使用 DeepSeek API（`deepseek-reasoner`）
- **近似基线**：SAMARL / AMAPPO / EUREKA 源自其他论文、无官方代码，按原文描述近似实现（可选后续）
- 其余差异随实现进度在此补充

## 📚 参考

- 原论文：见上方论文信息
- 论文精读笔记：[LLM-MAPPO_Markdown_Reader/paper.md](LLM-MAPPO_Markdown_Reader/paper.md)

## 📄 License

待定
