# LLM-MAPPO 论文阅读与复现

复现论文 *Multi-UAV Trajectory Planning for Dynamic Target Search: An LLM-Enhanced Multi-Agent Reinforcement Learning Algorithm*（IEEE TCCN, 2026）。

## 当前状态

代码已完成论文逐式审计和核心修正：动态目标以实体身份连续移动且数量保持 15；观测按 Eq. (18) 组织；7 架 UAV 使用独立 Actor 与集中式 Critic；非法动作在采样 logits 上屏蔽；DPES 按 Eq. (13)–(17) 更新；默认奖励采用附录 Eq. (34)；训练 transition 保存动作前的全局状态；测试阶段冻结参数并按 Algorithm 3 从策略分布采样，在 8 个独立随机种子上运行。

当前远程基线版本为提交 `e31d9f8`，已在 austlab 完成 3,000-episode pilot 和 28,000-episode formal。诊断确认策略在 outer 150 达到峰值后塌缩：同一验证 seeds 100–107 上，成功率从 98.33% 降至 outer 1400 的 90.83%，终局区域不确定度从 0.0920 升至 0.2297。当前代码正在加入显式熵正则与独立验证选模，完成后需重新运行短 pilot。

精读论文后已经确认：Algorithm 3 第 8 行使用 $a_n(t)\sim\pi_{\theta_n}$，执行阶段应从策略分布采样；论文未公开 entropy 系数和 checkpoint 选择规则；附录 Eq. (34) 中 $N^{sear}(t)$ 的累计定义及 success rate 的目标覆盖率口径均已正确实现。当前评测默认采用可复现的策略采样，`--deterministic` 仅用于 argmax 对照；远程实验显式使用 `entropy_coef=0.01`，并在隔离验证集上选模后才运行最终测试。

## 当前实验结果

austlab pilot（outer 150）在测试 seeds 300–307 上达到 14.625/15 个目标、97.50% success rate、终局区域不确定度 0.0996；28,000-episode formal 最终 checkpoint 在 seeds 400–407 上只有 13.75/15、91.67% 和 0.2565。两轮均无碰撞。该差异与同一验证集上的 checkpoint 横向结果一致，说明长训发生真实策略退化，而不是测试 seed 的偶然波动。

## 下一步实验

1. 对稳定性修订运行新的 3,000-episode pilot，唯一训练变量为 `entropy_coef=0.01`。
2. 每 50 outer 保存快照，用 seeds 100–107 验证选模，再用 seeds 300–307 测试。
3. 同时比较动作熵、success rate、终局区域不确定度与碰撞状态，确认 outer 150 前后不再迅速塌缩。
4. pilot 通过后才启动新的 28,000-episode formal；formal 的最终测试使用 seeds 400–407。

## 目录

- `LLM-MAPPO_Markdown_Reader/`：论文正文、公式索引与图表材料。
- `reproduction/`：环境、MAPPO、DPES、论文奖励、训练和固定策略评测。
- `tests/test_paper_alignment.py`：论文关键约束的回归测试。
- `cloud/cloud_run.sh`：austlab 上的 pilot/formal 训练、验证选模与 8-seed 测试入口。
- `cloud/CLOUD_RUN.md`：服务器目录、协议和运行说明。
- `reproduction/README_compare_with_paper.md`：实现与论文差异清单。

## 运行约定

GPU 实验只在 SSH Host `austlab` 上运行，远程项目目录为 `/home/lihaitao202413767/liuqiying2025313900/llm-mappo`。本机只做静态检查和 CPU 单元测试，不运行 CUDA 训练。

```bash
ssh austlab
cd /home/lihaitao202413767/liuqiying2025313900/llm-mappo
bash cloud/cloud_run.sh
```

旧实验、旧 LRS 代码和旧图仍保留用于追溯，但最终结果必须来自 `reproduction/paper_aligned_runs/` 下的新运行目录，并记录提交号、训练 seed、测试 seeds 和动作执行方式。
