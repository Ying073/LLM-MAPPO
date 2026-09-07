# LLM-MAPPO 论文阅读与复现

复现论文 *Multi-UAV Trajectory Planning for Dynamic Target Search: An LLM-Enhanced Multi-Agent Reinforcement Learning Algorithm*（IEEE TCCN, 2026）。

## 当前状态

代码已完成论文逐式审计和核心修正：动态目标以实体身份连续移动且数量保持 15；观测按 Eq. (18) 组织；7 架 UAV 使用独立 Actor 与集中式 Critic；非法动作在采样 logits 上屏蔽；DPES 按 Eq. (13)–(17) 更新；默认奖励采用附录 Eq. (34)；训练 transition 保存动作前的全局状态；测试阶段冻结参数并按 Algorithm 3 从策略分布采样，在 8 个独立随机种子上运行。

当前代码版本为提交 `9168253`，已同步至 GitHub 和 AutoDL，并在本地及 AutoDL 均通过 45 项论文对齐测试。首轮 28,000-episode 正式训练已经完成，但该模型训练时额外使用了论文 Eq. (23) 未报告的 `0.01` entropy bonus，原评测又采用了确定性 argmax 和中途 checkpoint 选择，因此这轮训练保留为重要诊断结果，尚不作为最终论文对齐结果。

精读论文后已经确认：Algorithm 3 第 8 行使用 $a_n(t)\sim\pi_{\theta_n}$，执行阶段应从策略分布采样；Eq. (23) 没有 entropy bonus；附录 Eq. (34) 中 $N^{sear}(t)$ 的累计定义及论文 success rate 的目标覆盖率口径均已正确实现。当前评测默认采用可复现的策略采样，`--deterministic` 仅用于 argmax 对照；正式脚本训练完成后直接评测最终 `policy.pt`，不再选择中途 checkpoint。

## 当前实验结果

首轮正式训练的最终 checkpoint 位于 AutoDL：

`reproduction/paper_aligned_runs/formal_seed0_20260907_125633/policy.pt`

同一最终 checkpoint 在 seeds 200–207 上的对照结果如下。两种执行方式均未发生碰撞。

| 动作执行方式 | 平均发现目标 | success rate | 平均搜索时间 | 终止区域不确定度 |
|---|---:|---:|---:|---:|
| 确定性 argmax（诊断） | 14.25 / 15 | 95.00% | 348.50 | 0.2551 |
| Algorithm 3 策略采样 | **14.50 / 15** | **96.67%** | 366.75 | **0.1031** |

修正后还完成了一个 40-episode、`entropy_coef=0` 的 GPU smoke，完整训练、保存和 8-seed 采样评测链均正常。其 91.67% success rate 只用于验证代码链，训练量过小，不能作为算法性能结论。

## 下一步实验

1. 从提交 `9168253` 启动新的 3,000-episode pilot，使用论文 Eq. (23) 的纯 clipped objective（`entropy_coef=0`）。
2. pilot 使用新的诊断 seeds 300–307；已经查看过的 seeds 200–207 不再用于模型选择或调参。
3. 检查 success rate、终止区域不确定度、搜索时间、碰撞情况，以及随机策略跨 seed 的方差。
4. pilot 通过后再启动新的 28,000-episode formal，并预留未参与诊断的 seeds 400–407 作为最终测试集。
5. 新 formal 与首轮模型必须在同一套策略采样协议下比较；40-episode smoke 和历史错误实验不进入论文结果表。

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

旧实验、旧 LRS 代码和旧图仍保留用于追溯，但最终结果必须来自 `reproduction/paper_aligned_runs/` 下的新运行目录，并记录提交号、训练 seed、测试 seeds 和动作执行方式。
