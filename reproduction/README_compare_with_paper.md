# 实现与论文对齐审计（2026-09-06）

## 结论

旧实验不能作为论文复现结果：旧环境会丢失动态目标实体，批量 reset 会泄漏 belief state，MAPPO 使用了共享 Actor 和错误超参数，奖励及测试 seed 的解释也与论文不一致。相关数据保留用于追溯，但新版报告不得引用其绝对指标。

新版已按论文正文、Algorithm 3、表 I/II 与附录 Eq. (34) 修正核心执行链，并通过 AutoDL CUDA smoke。后续审计发现历史 pilot 使用的批量环境存在感知 scatter 覆盖、随机流错位、联合碰撞处理和 DPES 访问时间错误，因此历史 pilot 的数值全部只作诊断，不再作为复现证据。当前应先重跑修复后的 3,000-episode pilot，再决定是否启动 28,000-episode 正式训练。

## 已对齐

| 项目 | 论文要求 | 当前实现 |
|---|---|---|
| 场景 | 20×20、7 UAV、20 障碍、15 动态目标、500 步 | 已对齐；目标按实体追踪，不再因移动合并而减少 |
| 目标运动 | 1 m/s、随机方向 | 连续米制位置、episode 内保持方向，边界/障碍处反射 |
| 观测 Eq. (18) | 所有 UAV 位置、局部不确定度、局部目标/障碍存在状态；Actor 另接 DP | Actor 71 维，无 DP；使用 DPES 时 96 维 |
| CTDE | 多个分散 Actor、一个集中 Critic | 7 个独立 Actor + 1 个 centralized Critic |
| 网络/超参 | 两层 64 ReLU、lr 0.0002、γ 0.95 | 已对齐 |
| 碰撞 | action mask 硬约束 | 采样与 PPO update 均屏蔽非法动作；环境另做冲突兜底 |
| DPES | Eq. (13)–(17)，表 II 参数 | 已修正括号、边界邻居数、局部 DP patch 与非扩散 G_lu |
| 奖励 | Appendix Eq. (34) | 默认 `paper-rbest`，基于同一 transition 计算团队奖励；`Rdis` 按论文“惩罚间距不足”的文字含义逐机对截断为非负值 |
| 训练 transition | `O(t), a(t), r(t), O(t+1)` | Critic 保存动作前全局状态，不再错配 post-step state |
| 测试 | Algorithm 3 以 `a_n(t) ~ pi_theta_n` 采样动作；固定模型，8 个独立随机 seed | `evaluate.py` 冻结参数、默认策略采样并同时固定环境与 PyTorch 随机流；`--deterministic` 仅供 argmax 诊断 |
| Actor 目标 Eq. (23) | 仅 PPO clipped surrogate objective | 默认 `entropy_coef=0`，不再额外加入论文未报告的 entropy bonus |
| 正式模型 | 训练达到 28,000 episodes 后输出 optimal Actor/Critic；未描述中途验证选模 | 正式脚本直接测试最终 `policy.pt`；周期快照仅用于断点恢复和诊断 |

## 论文实验参数

- 2,000 m × 2,000 m 区域划分为 20×20 网格。
- UAV 高度为 50/100/150 m；感知网格数为 1/5/9；检测概率为 0.9/0.8/0.7；虚警率为 0.1/0.2/0.3。
- UAV 速度 10 m/s，初始能量 `6×10^4 J`；目标速度 1 m/s；时间步 1 s。
- DPES：`D=200, E_s=0.1, G_s=0.1, d_hv=0.1, d_lu=0.003, d_cs=0.2`。
- 训练 28,000 episodes，每个 episode 最多 500 steps。
- 测试在相同设置下用 8 个独立随机 seeds；这不等于“训练 8 个模型”。

## 尚不能声称完全复现的部分

1. 已完成的 28,000-episode 结果是在旧的额外 entropy bonus 下训练，且曾用确定性 argmax 与中途 checkpoint 选择评测，不能作为最终论文对齐数字；可先按修正后的采样协议复测最终 checkpoint，再决定是否重训。
2. 论文未给出 Eq. (34) 中 `chi_th`、高空条件概率和安全距离的全部数值。当前默认值分别为 0.3、0.8 和 1 个网格，是公开、可改的复现假设。论文公式中的 `Rdis` 未写出截断，但正文将其描述为间距不足惩罚；直接使用原式会让远距离产生巨额正奖励，因此实现采用 `max(0, d_safe - distance + 1)`，报告中必须注明这是依据文字语义作出的纠错性解释。
3. 论文同时给出 100 m 网格、1 s 步长、10 m/s UAV 速度和相邻格动作，这些量存在尺度张力。当前轨迹仍使用离散相邻格动作，能耗按论文 10 m/s 推进模型计算。由 Eq. (1) 得到约 126.03 W，完整 500 秒 episode 约耗 63.0 kJ，超过表 I 的 60 kJ 初始能量；论文目标与公开奖励未说明耗尽终止/返航约束，当前仅记录并将归一化剩余能量截断到零，不擅自增加终止规则。
4. DPES 公式使用 `p_{n,i}`，而算法流程先融合 GTPM。当前解释为：用融合 GTPM 更新公共信息素场，再给每架 UAV 截取其感知域 DP；报告中应注明这一解释。
5. LRS/DeepSeek 代码仍作为研究模块保留，但正式对齐训练直接采用论文公开的 Eq. (34)，不把历史 canned/cache 输出当作重新生成的 LLM 结果。

## 新实验分级

- GPU smoke：只验证 CUDA、形状、反向更新、保存 checkpoint 与加载评测。
- Pilot：3,000 episodes；用于查数值稳定性和趋势，不用于论文最终数字。
- Formal：28,000 episodes；训练完成后冻结策略，在此前未用于诊断的 seeds 200–207 上测试并报告 mean±std。

正式训练每 50 个 outer iteration 原子保存一次最新完整训练状态，包括模型、优化器、环境、历史与 NumPy/PyTorch/CUDA 随机数流；同时把该时点的冻结策略保留在 `policy_checkpoints/`，不再只剩最终模型。可通过 `RESUME_FROM` 和原 `RUN_DIR` 继续，不必从头训练。

训练结束后使用验证 seeds 100–107 对阶段策略选模，优先级依次为：全部无碰撞、找到的不同目标数、终局区域不确定度、搜索时间、累计搜索指示。最终 seeds 200–207 只用于一次独立测试；原 seeds 0–7 已参与历史诊断，不再视为未见测试集。该 checkpoint 选择协议是本复现新增的防过拟合措施，论文没有公开对应细节。

所有新结果写入 `reproduction/paper_aligned_runs/` 的唯一时间戳目录，避免覆盖旧实验。
