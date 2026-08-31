OUTPUT_INTERFACE = """\
请输出一个 Python 函数，签名固定为：
def reward(obs, prev_obs, info) -> float
其中：
- obs 为字典：键 "obs" 是当前 UAV 的观测（numpy 数组，展平的网格序列，前 1/3 为目标概率、
  中间 1/3 为不确定度、后 1/3 为信息素）、键 "agent" 是 UAV 编号。
- info 为字典：贪心 rollout 评估时含 {"action": a, "next": (nx, ny, nalt)}，
  即候选动作编号与执行后所在网格，可据此对候选动作排序。
函数必须可被 exec 直接执行，只使用 numpy 与标准库，不引用任何外部变量。"""

TASK_DESCRIPTION = """\
你是一名多智能体强化学习的奖励函数设计者。场景：{n_uav} 架无人机在
{grid_size}x{grid_size} 网格的 2000mx2000m 区域搜索 {n_targets} 个动态地面目标，
区域内有静态障碍物。目标：最大化成功搜索的目标数，最小化搜索区域平均不确定度，
同时避免无人机之间及与障碍物的碰撞。动作空间：北/东/南/西/升/降。"""

REASONING_GUIDANCE = """\
设计奖励时请考虑：(1) 搜索到目标应给强正奖励；(2) 单步降低区域不确定度应给连续正奖励；
(3) 高度自适应：高空广域感知、低空精确检测；(4) UAV 间应保持分散以覆盖更多区域；
(5) 碰撞与越界应惩罚。避免只依赖稀疏的"找到目标"奖励。"""


def build_initial_prompt(n_uav, grid_size, n_targets, include_reasoning=True) -> str:
    parts = [TASK_DESCRIPTION.format(n_uav=n_uav, grid_size=grid_size, n_targets=n_targets)]
    if include_reasoning:
        parts.append(REASONING_GUIDANCE)
    parts.append(OUTPUT_INTERFACE)
    return "\n\n".join(parts)


def build_feedback_prompt(base_prompt, best_code, best_score, worst_codes) -> str:
    worst = "\n".join(f"```python\n{c}\n```\n(score={s})" for c, s in worst_codes)
    return "\n\n".join([
        base_prompt,
        f"当前最优奖励函数（score={best_score}）:\n```python\n{best_code}\n```",
        f"以下为表现较差的候选（避免类似设计）:\n{worst}",
        "请生成一个改进后的奖励函数，直接输出代码。",
    ])
