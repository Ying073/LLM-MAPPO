# LLM-MAPPO 论文阅读与复现

复现论文 *Multi-UAV Trajectory Planning for Dynamic Target Search: An LLM-Enhanced Multi-Agent Reinforcement Learning Algorithm*（IEEE TCCN, 2026）。

## 当前状态

代码已完成一次论文逐式审计和核心修正：动态目标以实体身份连续移动且数量保持 15；观测按 Eq. (18) 组织；7 架 UAV 使用独立 Actor 与集中式 Critic；非法动作在采样 logits 上屏蔽；DPES 按 Eq. (13)–(17) 更新；默认奖励采用附录 Eq. (34)；训练 transition 保存动作前的全局状态；测试阶段冻结策略并运行 8 个独立环境种子。

新版已在 AutoDL RTX 4090D 上通过 CUDA smoke 和断点续训 smoke。历史 pilot 后续审计发现批量环境的感知更新、联合碰撞处理和 DPES 访问时间与单环境不一致，因此其指标全部降级为诊断材料，不能作为复现结果。修复后必须先重跑 3,000-episode pilot，验证通过后再进行 28,000-episode 正式训练。

## 目录

- `LLM-MAPPO_Markdown_Reader/`：论文正文、公式索引与图表材料。
- `reproduction/`：环境、MAPPO、DPES、论文奖励、训练和固定策略评测。
- `tests/test_paper_alignment.py`：论文关键约束的回归测试。
- `cloud/cloud_run.sh`：AutoDL 上的 pilot/formal 训练与 8-seed 测试入口。
- `cloud/CLOUD_RUN.md`：服务器目录、协议和运行说明。
- `reproduction/README_compare_with_paper.md`：实现与论文差异清单。

## 运行约定

GPU 实验只在 SSH Host `autodl` 上运行，远程项目目录为 `/root/autodl-tmp/projects/llm-mappo`。本机只做静态检查和 CPU 单元测试，不运行 CUDA 训练。

```bash
ssh autodl
cd /root/autodl-tmp/projects/llm-mappo
bash cloud/cloud_run.sh
```

旧实验、旧 LRS 代码和旧图仍保留用于追溯，但正式结果必须来自 `reproduction/paper_aligned_runs/` 下的新运行目录。
