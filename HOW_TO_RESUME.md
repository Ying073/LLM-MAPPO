# HOW_TO_RESUME

## 一句话状态

论文对齐修订、AutoDL GPU/断点续训 smoke 和修正奖励后的 3,000-episode pilot 已完成；pilot 的 8-seed 成功率为 91.67%，全部无碰撞，当前准备 28,000-episode 正式训练。旧版 150-episode、m12 与首轮错误奖励 pilot 数据只作诊断记录，不能用于论文结论。

## 接手顺序

1. 先读 `reproduction/README_compare_with_paper.md`，确认已对齐项和仍需声明的假设。
2. 运行 `python -m unittest discover -s tests -p 'test_*.py' -v`。
3. 查看已通过的 `reproduction/paper_aligned_runs/pilot_seed0_rewardfix_20260906_221558/` 和 `resume_smoke_20260906_232711/`。
4. 使用 `RUN_KIND=formal bash cloud/cloud_run.sh` 完成 28,000-episode 正式训练。
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

## 仍需诚实声明

论文没有给出 Eq. (34) 中 `chi_th`、高空条件概率和安全距离的全部数值；当前实现采用显式默认假设。`Rdis` 的非负截断也未写在公式中，是依据论文“惩罚间距不足”的文字语义作出的纠错性解释。论文的网格尺寸、1 s 时间步与 10 m/s UAV 速度同“每步移动到相邻 100 m 网格”的描述存在尺度张力，实现保留离散相邻动作并按论文速度计算能耗。DPES 使用融合后的全局目标概率构造信息素，再向各 UAV 提供局部 DP patch；这是根据算法流程作出的实现解释。

## 服务器约定

- SSH Host：`autodl`
- 项目目录：`/root/autodl-tmp/projects/llm-mappo`
- Python：`/root/mappo_venv/bin/python`
- 新结果：`reproduction/paper_aligned_runs/`

任何删除、环境重装、覆盖 checkpoint 或覆盖历史结果的操作，都必须先征得用户同意。
