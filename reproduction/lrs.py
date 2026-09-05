"""
lrs.py —— 离线 LLM 奖励塑形 (LRS) 主循环 (论文 §IV-B, 算法 2, 公式 20–22, 27–29)

一句话：
    在 MAPPO 训练之前，用 LLM 迭代生成"奖励函数"，每个候选奖励函数不训练网络、
    只用"贪心跑一个 episode"来打分（用公式 12a 的原始任务目标 J），
    保留表现最好的那个作为训练用的 R^best。

论文流程（算法 2，K=5 次迭代）：
    初始化提示 P_1 (任务描述+系统模型+推理指导+奖励接口+输出约束)
        ↓
    k 从 1 到 K:
        R_k ~ LLM(P_k)                        生成候选奖励函数
        a_k^*(t) = argmax R_k(O(t), a(t))     公式 20: 贪心选动作 → 跑一个 episode → 轨迹 T_k
        J(T_k) = 公式 12a 打分                累计搜索目标数 − 终止区域平均不确定度
        B^R ← (R_k, J(T_k))                   存入候选缓冲
        R_k^best = argmax_{R∈B^R} J(T)        公式 21: 当前最优
        η_k = J(T_k^best)                     公式 27: 单调不增递
        P_k^feed = {R_k^best, J(T_k^best), {R_k', J(T_k')}_{k'<k}}   公式 22: 反馈提示
        P_{k+1} = P_1 + P_k^feed
        ↓
    R^best = R_K^best (训练/执行阶段固定)

为什么要"离线 + 不训练"：论文 §I-D 强调，逐候选重新训练 MARL 评估成本极高；
用"贪心轨迹的性能"替代"训练后的性能"可大幅降低成本，同时保持奖励质量。

LLM 后端抽象：
    这里把 LLM 实现成一个可插拔接口 `LLMBackend`。默认 `CannedLLM` 从一组
    预置的候选奖励函数里"生成"代码（对应于论文公布的 R1 / R3 / R^best 结构），
    这样在没有真实 LLM API 的环境里也能端到端跑通整个 LRS 闭环；
    换用真实 DeepSeek-R1-7B 时，只需实现一个子类返回真实生成的代码即可。
"""

import copy
import numpy as np

from .env.search_env import (
    LY, LX, N_UAV, MAX_STEPS,
    TARGET_CONFIRM_THRESHOLD,
)


# ============================================================
# 一、可插拔 LLM 后端
# ============================================================
class LLMBackend:
    """LLM 接口。子类需实现 generate(prompt) -> 候选奖励函数代码 (str)。"""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class CannedLLM(LLMBackend):
    """预置候选的"模拟 LLM"。

    对应论文 §附录 S062–S063 公布的候选奖励结构：
        R1   : 目标搜索 + 不确定度降低 + 高度自适应 + 分离
        R3   : 更强调目标确认 + 更明确惩罚间距不足
        Rbest: 提高搜索优先级 + 阈值式+连续式不确定度 + 促进协作

    每次 generate 按提示中的迭代次数 k 返回对应候选的代码。
    """

    def generate(self, prompt: str) -> str:
        # 提示里携带了当前迭代序号（在 build_prompt 里塞进去）
        if "iteration: 1" in prompt or "k=1" in prompt:
            return CAND_R1
        if "iteration: 3" in prompt or "k=3" in prompt:
            return CAND_R3
        # 其余迭代返回更强的 Rbest
        return CAND_RBEST


# ============================================================
# 二、候选奖励函数代码（LLM"生成"的可执行 Python 函数）
# ============================================================
# 统一接口: def reward(env, n, action, prev_au) -> float
#   env      : SearchEnv 快照 (拷贝出用于假想执行一步)
#   n        : 第 n 架 UAV
#   action   : 候选动作 0..5
#   prev_au  : 上一步 area uncertainty
# 返回: 该 UAV 采取该动作所对应的奖励分量值

CAND_R1 = '''
def reward(env, n, action, prev_au):
    """R1: 基础结构 — 目标搜索 + 不确定度降低 + 高度自适应 + 分离."""
    r = 0.0
    # (a) 目标搜索: 感知域内若有目标/高概率格 → 正奖励
    ix, iy, h = env.uav_pos[n]
    for (dx, dy) in ((0,-1),(1,0),(0,1),(-1,0),(0,0)):
        nx, ny = ix+dx, iy+dy
        if 0 <= nx < LX and 0 <= ny < LY and env.zeta[ny,nx] == 1:
            r += 2.0
    # (b) 不确定度降低: 本步 area uncertainty 下降
    r += 5.0 * (prev_au - env.area_uncertainty())
    # (c) 高度自适应: 上升/下降给小幅惩罚, 激励水平探索
    if action in (4, 5):
        r -= 0.1
    # (d) 分离: 远离其它 UAV 给正奖励
    for m in range(N_UAV):
        if m == n: continue
        dist = abs(env.uav_pos[m,0]-ix) + abs(env.uav_pos[m,1]-iy)
        r += 0.05 * min(dist, 5)
    return r
'''
# 注意: CAND_R1 里引用了 N_UAV / LX / LY，会被 exec 的 globals 提供。

CAND_R3 = '''
def reward(env, n, action, prev_au):
    """R3: 更强调目标确认 + 更明确惩罚间距不足."""
    r = 0.0
    # (a) 目标确认: 感知域内高置信目标 → 强正奖励
    ix, iy, h = env.uav_pos[n]
    confirm = 0
    for (dx, dy) in ((0,-1),(1,0),(0,1),(-1,0),(0,0)):
        nx, ny = ix+dx, iy+dy
        if 0 <= nx < LX and 0 <= ny < LY:
            if env.zeta[ny,nx] == 1 and env.gtpm[ny,nx] >= TARGET_CONFIRM_THRESHOLD:
                confirm += 1
    r += 3.0 * confirm
    # (b) 不确定度连续下降
    r += 8.0 * (prev_au - env.area_uncertainty())
    # (c) 高度自适应
    if action in (4, 5):
        r -= 0.2
    # (d) 分离: 明确惩罚间距不足 (<2)
    for m in range(N_UAV):
        if m == n: continue
        dist = abs(env.uav_pos[m,0]-ix) + abs(env.uav_pos[m,1]-iy)
        if dist < 2:
            r -= 1.0
        else:
            r += 0.05 * min(dist, 5)
    return r
'''

CAND_RBEST = '''
def reward(env, n, action, prev_au):
    """Rbest: 提高搜索优先级 + 阈值式+连续式不确定度 + 促进协作."""
    r = 0.0
    ix, iy, h = env.uav_pos[n]
    # (a) 目标搜索（最高优先级）
    for (dx, dy) in ((0,-1),(1,0),(0,1),(-1,0),(0,0)):
        nx, ny = ix+dx, iy+dy
        if 0 <= nx < LX and 0 <= ny < LY and env.zeta[ny,nx] == 1:
            r += 4.0
    # (b) 阈值式不确定度: 感知域内有格子的不确定度降到阈值以下 → 正奖励
    #     连续式不确定度: 整体 area uncertainty 下降 → 正奖励
    threshold_reduce = 0
    for (dx, dy) in ((0,-1),(1,0),(0,1),(-1,0),(0,0)):
        nx, ny = ix+dx, iy+dy
        if 0 <= nx < LX and 0 <= ny < LY and env.geum[ny,nx] < 0.3:
            threshold_reduce += 1
    r += 1.5 * threshold_reduce
    r += 10.0 * (prev_au - env.area_uncertainty())
    # (c) 高度自适应
    if action in (4, 5):
        r -= 0.1
    # (d) 协作分离: 保持适度间距
    close = 0
    for m in range(N_UAV):
        if m == n: continue
        dist = abs(env.uav_pos[m,0]-ix) + abs(env.uav_pos[m,1]-iy)
        if dist < 1:
            close += 1
        elif dist < 6:
            r += 0.1 * (6 - min(dist, 6))
    r -= 1.0 * close
    return r
'''


# ============================================================
# 三、候选奖励函数编译 + 贪心评估
# ============================================================
def compile_reward(code: str):
    """把 LLM 生成的代码字符串编译成可调用函数 reward(env,n,action,prev_au).

    exec 的命名空间提供候选函数用到的全局量 (N_UAV, LX, LY, TARGET_CONFIRM_THRESHOLD)。
    """
    ns = {
        "N_UAV": N_UAV, "LX": LX, "LY": LY,
        "TARGET_CONFIRM_THRESHOLD": TARGET_CONFIRM_THRESHOLD,
        "np": np,
    }
    exec(code, ns)
    return ns["reward"]


def greedy_step(env, reward_fn, prev_au: float) -> list[int]:
    """公式 20: 每架 UAV 贪心选使 R_k 最大的动作.

    对每架 UAV，枚举其 6 个候选动作，在 *深拷贝* 的 env 上假想执行一步，
    计算当前候选奖励函数的值，取 argmax。（深拷贝保证不污染真实环境状态）
    """
    actions = []
    for n in range(N_UAV):
        best_a, best_r = 0, -1e18
        for a in range(6):
            env_cp = copy.deepcopy(env)          # 快照，供假想执行
            env_cp.step([a if m == n else 0 for m in range(N_UAV)])   # 只让 n 动，其余原地
            r = reward_fn(env_cp, n, a, prev_au)
            if r > best_r:
                best_r, best_a = r, a
        actions.append(best_a)
    return actions


def evaluate_candidate(reward_fn, env, seed: int = 0) -> tuple[float, dict]:
    """公式 12a: 用候选奖励函数贪心跑一个完整 episode, 返回 J(T) 与轨迹指标.

    J(T) = Σ_t Σ_i 1[p_i(t)≥ξ, ζ_i(t)=1] − χ^area(T)
         = 本 episode 累计确认的目标网格数 − 终止时的区域平均不确定度

    返回 (J_score, metrics)
    """
    # 固定初始观测，保证跨迭代可比
    env.reset()
    total_searched = 0
    prev_au = env.area_uncertainty()
    for t in range(MAX_STEPS):
        actions = greedy_step(env, reward_fn, prev_au)
        _, _, done, info = env.step(actions)
        total_searched = info["searched_count"]
        prev_au = info["area_uncertainty"]
        if done:
            break
    J = total_searched - info["area_uncertainty"]
    metrics = {
        "J": float(J),
        "searched": int(total_searched),
        "area_unc": float(info["area_uncertainty"]),
    }
    return J, metrics


# ============================================================
# 四、LRS 主循环 (算法 2)
# ============================================================
class LRS:
    """离线 LLM 奖励塑形主循环. K 次迭代, 每次生成→贪心评估→选优→反馈."""

    def __init__(self, llm: LLMBackend = None, K: int = 5, seed: int = 0):
        self.llm = llm if llm is not None else CannedLLM()
        self.K = K
        self.seed = seed
        self.env = None      # 在 run() 里实例化（需要传入 SearchEnv）
        self.B_R = []        # 候选缓冲 B^R: list of (code, reward_fn, J, metrics)
        self.history = []    # 记录每次迭代: (k, J, searched, area_unc, eta_k)

    # ---- 初始化提示 P_1 (任务描述 + 系统模型 + 推理指导 + 接口 + 约束) ----
    def init_prompt(self) -> str:
        return (
            "You are a MARL reward-function designer for multi-UAV dynamic target search.\n"
            "TASK: 7 UAVs search 15 moving targets in a 20x20 grid with 20 obstacles.\n"
            "ACTION: 6 discrete actions {N,E,S,W,ascend,descend} per UAV.\n"
            "OBJECTIVE (Eq. 12a): maximize cumulative searched targets - terminal area uncertainty.\n"
            "REASONING: dense reward should reward search success, uncertainty reduction, "
            "altitude adaptation, and inter-UAV separation; penalize collisions.\n"
            "INTERFACE: define reward(env, n, action, prev_au) -> float.\n"
            f"iteration: k={self.cur_k}\n"
        )

    # ---- 反馈提示 P_k^feed (公式 22) ----
    def feedback_prompt(self) -> str:
        # 当前最优 (idx=0 候选代码, idx=2 J, idx=3 metrics；中间 idx=1 是 reward_fn)
        if not self.B_R:
            return "FEEDBACK: (no prior candidates yet)\n"
        best_code = self.B_R[self.best_idx][0]
        best_J    = self.B_R[self.best_idx][2]
        best_m    = self.B_R[self.best_idx][3]
        lines = ["FEEDBACK:",
                 f"BEST R (J={best_J:.3f}, searched={best_m.get('searched',0)}, "
                 f"area_unc={best_m.get('area_unc',0):.4f}):",
                 best_code]
        # 负面示例: 表现较差的候选
        lines.append("NEGATIVE EXAMPLES (worse candidates):")
        for (code, _, J, m) in self.B_R:
            if J < best_J - 1e-6:
                lines.append(f"  - J={J:.3f}: {code[:120]}...")
        return "\n".join(lines)

    # ---- 公式 21: 选出当前最优候选 ----
    @property
    def best_idx(self) -> int:
        return int(np.argmax([item[2] for item in self.B_R]))

    # ---- 公式 27: 当前最优性能指标 η_k（单调不增递）----
    @property
    def eta(self) -> float:
        return self.B_R[self.best_idx][2] if self.B_R else -1e18

    # ---- 主循环 ----
    def run(self, env, seed: int = 0):
        self.env = env
        self.B_R = []
        self.history = []

        for k in range(1, self.K + 1):
            self.cur_k = k
            prompt = self.init_prompt() + self.feedback_prompt()

            # (1) LLM 生成候选奖励函数代码 R_k
            code = self.llm.generate(prompt)

            # (2) 编译成可调用
            reward_fn = compile_reward(code)

            # (3) 公式 20 贪心跑一个 episode → 公式 12a 打分
            J, metrics = evaluate_candidate(reward_fn, self.env, seed)

            # (4) 存入候选缓冲 B^R
            self.B_R.append((code, reward_fn, J, metrics))

            # (5) 记录 η_k (公式 27)
            self.history.append((k, J, metrics["searched"], metrics["area_unc"], self.eta))
            print(f"[LRS k={k}/{self.K}] J={J:+.3f}  searched={metrics['searched']:4d}  "
                  f"area_unc={metrics['area_unc']:.4f}  |  best so far J={self.eta:+.3f} "
                  f"({'TARGET CONFIRMED' if metrics['searched']>=15 else ''})")

        # (6) 最终最优奖励函数 R^best
        self.best_code = self.B_R[self.best_idx][0]
        self.best_fn = self.B_R[self.best_idx][1]
        self.best_J = self.B_R[self.best_idx][2]
        self.best_metrics = self.B_R[self.best_idx][3]
        return self.best_fn, self.best_code, self.best_J, self.best_metrics
