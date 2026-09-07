# AutoDL 运行说明（论文对齐版）

服务器约定：SSH Host 为 `autodl`，项目固定放在 `/root/autodl-tmp/projects/llm-mappo`。GPU 检查、环境安装、训练和日志读取均在该服务器完成；本机不运行 CUDA 训练。

## 当前协议

- 训练：一个 MAPPO 策略，默认先做 3,000 episode 的 pilot；正式复现为 28,000 episode。
- 场景：20×20 网格、7 架 UAV、20 个障碍、15 个动态目标、每 episode 最多 500 步。
- 网络与优化：每架 UAV 独立 Actor，集中式 Critic；两层 64 单元 ReLU；学习率 0.0002，折扣因子 0.95。
- 奖励：直接使用论文附录 Eq. (34) 的 `R_best`，不再把历史 LRS 缓存冒充正式奖励。
- DPES：参数采用表 II；Actor 只接收本机感知域中的 DP。
- 选模：每 50 个 outer iteration 保留一个冻结策略，用独立验证 seeds 100–103 选模；验证集优先保证无碰撞，再比较目标覆盖和区域不确定度。
- 测试：冻结验证集选出的策略，在完全独立的 seeds 0–7 上测试，不在测试阶段继续学习，也不使用测试集选模。

## 运行

```bash
ssh autodl
cd /root/autodl-tmp/projects/llm-mappo
source /root/mappo_venv/bin/activate

# 机制与趋势验证（默认 3,000 episode）
bash cloud/cloud_run.sh

# 论文训练规模（28,000 episode；启动前应先确认 pilot 正常）
RUN_KIND=formal bash cloud/cloud_run.sh
```

每次运行都会创建带时间戳的新目录：`reproduction/paper_aligned_runs/<类型>_seed<种子>_<时间>/`。脚本故意不复用已有目录，因此不会覆盖历史结果。产物包括训练曲线、原始历史、可恢复的最新训练状态、各阶段冻结策略、验证集选模记录、完整日志，以及 8-seed 固定策略测试 JSON。

## 重要边界

3,000 episode 只是 pilot，不能报告论文的最终量化结论。只有 28,000 episode 的正式训练和 8-seed 固定策略测试完成后，才具备与论文主实验比较的训练规模。论文没有公开所有实现细节；奖励中 `chi_th`、高空条件概率与安全距离的数值仍是明确标注的复现假设，不能表述为作者原始参数。

论文没有公开训练过程中的 checkpoint 选择规则。使用独立验证 seeds 100–103 选模是本复现为防止长训练奖励投机而增加的实验协议，不应表述为论文作者原始设置。

旧目录 `/root/LLM-MAPPO_论文阅读与复现` 以及 `m12_results` 属于历史实现，仅作追溯，不纳入新版结果。
