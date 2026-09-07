# HOW_TO_RESUME

## 一句话状态

论文对齐修订、AutoDL GPU/断点续训 smoke、历史 3,000-episode pilot 和首轮 28,000-episode 正式训练均已完成。最新精读确认 Algorithm 3 要求从策略分布采样动作，Eq. (23) 没有 entropy bonus。当前提交 `9168253` 已把默认评测改为逐 seed 可复现的策略采样、把默认 entropy 系数改为 0，并让正式脚本直接评测训练结束时的 `policy.pt`。首轮正式模型按新协议复测达到 96.67% success rate，但因训练时仍使用旧 entropy bonus，只作为诊断基线。

## 接手顺序

1. 先读 `reproduction/README_compare_with_paper.md`，确认已对齐项和仍需声明的假设。
2. 运行 `python -m unittest discover -s tests -p 'test_*.py' -v`。
3. 查看首轮正式运行 `reproduction/paper_aligned_runs/formal_seed0_20260907_125633/` 中的新采样/argmax 对照结果，以及 `smoke_entropy0_9168253/` 的短链路验证。
4. 下一次运行 3,000-episode pilot，使用诊断 seeds 300–307；通过后再运行 28,000-episode formal，并只在最终阶段使用预留 seeds 400–407。
5. 如中断，使用原 `RUN_DIR` 并令 `RESUME_FROM=${RUN_DIR}/training_state.pt` 继续；不要另起模型或覆盖其他实验目录。

## 已修正的高风险问题

- 目标移动时合并导致 15 个目标退化为约 3 个。
- 批量环境 reset 未清空全部 belief map，跨 episode 泄漏状态。
- rollout 把动作后的全局状态与动作前观测混存。
- 旧安全奖励会再次执行动作，改变环境两次。
- 单一共享 Actor、Tanh 激活、错误学习率与折扣因子。
- 缺少论文要求的硬动作屏蔽。
- DPES Eq. (14) 括号错误，长期未访问释放量被错误扩散。
- 曲线平滑人为补前导零，以及云端输出相对路径重复拼接。
- 把 8 个训练 seed 错当成论文的 8 个测试随机种子。
- Eq. (34) 的未截断间距项让远距离变成巨额正奖励；现按“惩罚间距不足”的文字含义逐机对截断为非负值。
- 下降动作掩码未检查当前水平位置的障碍物高度，导致 UAV 可以从障碍物上方下降进障碍物体积。
- 周期 checkpoint 只覆盖保存最新状态，无法诊断训练过程中策略变化；现额外保留冻结策略，但论文对齐正式结果直接使用训练结束时的最终策略，不再通过验证集选择中途 checkpoint。
- 评测曾强制使用确定性 argmax，与 Algorithm 3 的策略采样不一致；现默认采样并让环境及 PyTorch 随机流都由测试 seed 控制。
- Actor 更新曾额外加入 `0.01` entropy bonus，而论文 Eq. (23) 未包含该项；新训练默认使用 `entropy_coef=0`。

## 仍需诚实声明

论文没有给出 Eq. (34) 中 `chi_th`、高空条件概率和安全距离的全部数值；当前实现采用显式默认假设。`Rdis` 的非负截断也未写在公式中，是依据论文“惩罚间距不足”的文字语义作出的纠错性解释。论文的网格尺寸、1 s 时间步与 10 m/s UAV 速度同“每步移动到相邻 100 m 网格”的描述存在尺度张力，实现保留离散相邻动作并按论文速度计算能耗。DPES 使用融合后的全局目标概率构造信息素，再向各 UAV 提供局部 DP patch；这是根据算法流程作出的实现解释。论文没有公开 checkpoint 选模方法，因此正式复现采用 Algorithm 3 的自然终点：训练达到预设 episode 数后评测最终 Actor 网络。

## 服务器约定

- SSH Host：`autodl`
- 项目目录：`/root/autodl-tmp/projects/llm-mappo`
- Python：`/root/mappo_venv/bin/python`
- 新结果：`reproduction/paper_aligned_runs/`

任何删除、环境重装、覆盖 checkpoint 或覆盖历史结果的操作，都必须先征得用户同意。
