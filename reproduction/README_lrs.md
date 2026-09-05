# M4 (LRS 离线 LLM 奖励塑形) 代码 ↔ 论文对照表

> 复现对象：论文 §IV-B (paper.md S038–S049, **算法 2**, 公式 **20 / 21 / 22 / 27 / 28 / 29**)。
>
> M4 在 M1+M2+M3 之上，把"奖励函数"也交由 LLM 设计。在 MAPPO 训练**之前**离线做 K 次迭代：
> LLM 生成候选奖励函数 → 每架 UAV 按公式 20 贪心跑一个 episode → 用公式 12a 的原始任务目标 J 打分
> → 保留表现最好的 `R^best`，后续训练固定用它。

---

## 一、LRS 是什么（一句话）

> 与其"训练 MARL 再评估每个候选奖励"，不如让 LLM 离线生成若干稠密奖励函数，
> 每个只跑"贪心轨迹 → 用公式 12a 的 J 打分"一次，挑最好的拿去做训练。
> LLM 的输出必须是 `reward(env, n, action, prev_au) -> float` 的 Python 代码。

这样既借了 LLM 的领域知识，又把评估成本从"每个候选训一次 MARL"压到"每个候选跑一次 episode"，对应论文 §I-D 强调的"降低评估成本"。

---

## 二、模块 ↔ 公式 / 算法

### 1. LLM 后端抽象 `class LLMBackend` ↔ 算法 2 第 3 行 `R_k ~ LLM(P_k)`

```python
class LLMBackend:
    """LLM 接口。子类需实现 generate(prompt) -> 候选奖励函数代码 (str)。"""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError
```

- 默认子类 **`CannedLLM`** 用"提示里的迭代序号 k"作为路由，从 `CAND_R1 / R3 / Rbest` 三个预置候选里挑一个返回。
  - 用预置候选的原因：本文在零联网环境下也能端到端跑通整个 LRS 闭环；
  - **接真实 DeepSeek-R1-7B 时**：写一个 `class DeepSeekR1(LLMBackend)`，在 `generate()` 里 POST 到 API，把返回的代码字符串原样 return 即可，下游完全不动。

### 2. 候选奖励函数 `CAND_R1 / R3 / Rbest` ↔ 论文 §附录 + 算法 2 第 3 行

每个候选都是一段符合接口 `reward(env, n, action, prev_au) -> float` 的 Python 字符串，统一由 `compile_reward(code)` 在沙箱里 `exec` 出可调用函数：

```python
def compile_reward(code: str):
    ns = {
        "N_UAV": N_UAV, "LX": LX, "LY": LY,
        "TARGET_CONFIRM_THRESHOLD": TARGET_CONFIRM_THRESHOLD,
        "np": np,
    }
    exec(code, ns)
    return ns["reward"]
```

对应论文：
- **R1**：基础结构（目标搜索 + 不确定度降低 + 高度自适应 + 分离）
- **R3**：更强调目标确认 + 更明确惩罚间距不足
- **R^best**：搜索优先级最高 + 阈值式+连续式不确定度 + 促进协作（论文最终保留的奖励函数）

### 3. 贪心选动作 `greedy_step()` ↔ **公式 20**

$$
a_k^*(t) \;=\; \arg\max_{a\in\mathcal{A}} \, R_k\!\left(\mathcal{O}(t), a\right)
$$

```python
def greedy_step(env, reward_fn, prev_au: float) -> list[int]:
    actions = []
    for n in range(N_UAV):
        best_a, best_r = 0, -1e18
        for a in range(6):
            env_cp = copy.deepcopy(env)                              # 快照, 假想执行
            env_cp.step([a if m == n else 0 for m in range(N_UAV)])  # 让 u_n 动，其余原地
            r = reward_fn(env_cp, n, a, prev_au)
            if r > best_r:
                best_r, best_a = r, a
        actions.append(best_a)
    return actions
```

要点：
- 每架 UAV 独立贪心（**联合动作** = 各 UAV 选自己局部最优动作的笛卡尔积）
- 对每个候选动作 `a ∈ {0..5}` 在**深拷贝**的环境上 `step()`，不污染真实轨迹
- "其余 UAV 原地不动"是为了让对比公平，只评估当前 UAV 的边际贡献

### 4. 评估候选 `evaluate_candidate()` ↔ **公式 12a**

$$
J(\mathcal{T}) \;=\; \sum_{t=1}^{T}\sum_{i=1}^{L_X L_Y} \mathbf{1}\!\left[p_i(t)\ge \xi,\;\zeta_i(t)=1\right] \;-\; \chi^{\text{area}}(T)
$$

```python
def evaluate_candidate(reward_fn, env, seed=0):
    env.reset()
    total_searched, prev_au = 0, env.area_uncertainty()
    for t in range(MAX_STEPS):
        actions = greedy_step(env, reward_fn, prev_au)
        _, _, done, info = env.step(actions)
        total_searched = info["searched_count"]          # 累计确认数 = Σ_t Σ_i 1[…]
        prev_au        = info["area_uncertainty"]       # χ^area(t)
        if done: break
    J = total_searched - info["area_uncertainty"]        # 终止时减一项
    metrics = {"J": float(J), "searched": int(total_searched),
               "area_unc": float(info["area_uncertainty"])}
    return J, metrics
```

返回 `(J_score, {searched, area_unc, J})`——`J` 同时是 metric 的 key 与返回值，刻意冗余方便读取。

### 5. 初始化提示 `LRS.init_prompt()` ↔ 算法 2 第 1 行 (P₁ 三件套)

论文 §IV-B 第 1 段说 P₁ 包含"任务描述 + 推理指导 + 输出模板"：

```python
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
```

任务描述（TASK/ACTION/OBJECTIVE） + 推理指导（REASONING） + 输出模板（INTERFACE + iteration tag）。
末尾的 `iteration: k=…` 供 `CannedLLM.generate` 做路由；真用 DeepSeek 时换成让 LLM 知道自己正在第几轮的提示。

### 6. 反馈提示 `LRS.feedback_prompt()` ↔ **公式 22**

$$
P_k^{\text{feed}} \;=\; \bigl\{\,R_k^{\text{best}},\;J(\mathcal{T}_k^{\text{best}}),\;\{(R_{k'},\;J(\mathcal{T}_{k'})\} \mid k'<k\,\bigr\}
$$

```python
def feedback_prompt(self) -> str:
    if not self.B_R:
        return "FEEDBACK: (no prior candidates yet)\n"
    best_code = self.B_R[self.best_idx][0]
    best_J    = self.B_R[self.best_idx][2]
    best_m    = self.B_R[self.best_idx][3]
    lines = ["FEEDBACK:",
             f"BEST R (J={best_J:.3f}, searched={best_m.get('searched',0)}, "
             f"area_unc={best_m.get('area_unc',0):.4f}):",
             best_code]
    lines.append("NEGATIVE EXAMPLES (worse candidates):")
    for (code, _, J, m) in self.B_R:
        if J < best_J - 1e-6:
            lines.append(f"  - J={J:.3f}: {code[:120]}...")
    return "\n".join(lines)
```

- 输入格式：**当前最优 R^best 全量代码 + 其 J**，外加**比它差的候选作负面示例**（论文 §IV-B 倒数第 2 段）
- 期望 LLM 收敛到"趋近 R^best、避开 R_worse"，但仍留出探索空间

### 7. 当前最优索引与 η_k ↔ **公式 21 + 公式 27**

$$
R_k^{\text{best}} \;=\; \arg\max_{R\in B^R}\! J(\mathcal{T})\qquad(21)
$$
$$
\eta_k \;=\; J\!\left(\mathcal{T}_k^{\text{best}}\right)\qquad(27)
$$

```python
@property
def best_idx(self) -> int:
    return int(np.argmax([item[2] for item in self.B_R]))   # item[2] = J

@property
def eta(self) -> float:
    return self.B_R[self.best_idx][2] if self.B_R else -1e18
```

注意：`item[1]` 是 `reward_fn` 函数对象（不能排大小），`item[2]` 才是 J。
（这里我栽过一次——`slice [:3]` 会把 `reward_fn` 当成 J 来 `:.3f` 格式化，立即 TypeError。）
实现里把"按 J 取 argmax"和"读 η"分开成两个 property，避免再碰 tuple 索引陷阱。

### 8. 主循环 `LRS.run()` ↔ **算法 2** 整体

```
Algorithm 2 (paper):
    输入: K, max_steps T, env seed
    初始化 B^R = ∅
    for k = 1..K:
        P_k ← P_1 + P_k^feed                    # prompt 组装
        R_k ~ LLM(P_k)                          # LLM 生成
        a_k*(t) = argmax R_k(O(t), a) for t∈T  # 公式 20 贪心跑一个 episode
        J_k ← J(T_k)                             # 公式 12a 打分
        B^R ← B^R ∪ {(R_k, J_k)}                # 入缓冲
        R_k^best ← argmax_{R∈B^R} J              # 公式 21
        η_k     ← J(T_k^best)                    # 公式 27
    return R_K^best, η_K
```

代码 (`LRS.run`) 一一对应：

| 步骤 | 代码行 | 公式 |
|---|---|---|
| prompt 组装 P_k = P_1 + P_k^feed | `prompt = self.init_prompt() + self.feedback_prompt()` | — |
| `R_k ← LLM(P_k)` | `code = self.llm.generate(prompt)` | 算法 2 第 3 行 |
| 编译 + 贪心评估 | `reward_fn = compile_reward(code)` ; `J, metrics = evaluate_candidate(...)` | 公式 20 + 12a |
| 入缓冲 `B^R` | `self.B_R.append((code, reward_fn, J, metrics))` | — |
| 公式 21 当最优 | `self.best_idx` property (argmax over J) | 公式 21 |
| 公式 27 η_k | `self.history.append(... self.eta)` | 公式 27 |
| `R_K^best` 输出 | `self.best_code / best_fn / best_J / best_metrics` | 算法 2 末尾 |

---

## 三、闭环运行验证 (K=5)

```bash
"C:/Users/lenovo/.workbuddy/binaries/python/envs/llm_mappo/Scripts/python.exe" -c "
import sys; sys.path.insert(0, 'C:/Users/lenovo/AI/大创/LLM-MAPPO_论文阅读与复现')
from reproduction.env.search_env import SearchEnv
from reproduction.lrs import LRS

env = SearchEnv(seed=0)
lrs = LRS(K=5, seed=0)
best_fn, best_code, best_J, best_metrics = lrs.run(env, seed=0)
"
```

输出日志（在本机实测）：

```
[LRS k=1/5] J=+65.865  searched=  66  area_unc=0.1352  |  best so far J=+65.865
[LRS k=2/5] J=+34.271  searched=  35  area_unc=0.7287  |  best so far J=+65.865
[LRS k=3/5] J=+71.982  searched=  72  area_unc=0.0177  |  best so far J=+71.982
[LRS k=4/5] J=+38.261  searched=  39  area_unc=0.7393  |  best so far J=+71.982
[LRS k=5/5] J=+47.316  searched=  48  area_unc=0.6837  |  best so far J=+71.982

=== LRS finished in 75.8s ===
Best J = +71.982  searched=72  area_unc=0.0177
eta_k = [65.865, 65.865, 71.982, 71.982, 71.982]   monotonic non-decreasing? True
```

验证到的事实：
1. **`η_k` 单调不降**（公式 27 + 28），从 `+65.87` 单调爬到 `+71.98`
2. 候选"差 → 优 → 重复"都触发缓冲保留最优逻辑（k=2/4/5 没破纪录时 η_k 维持上次最优）
3. 最终 `R^best` = R^best（k=3 入选），给出最优 `area_unc=0.018`

参数对应：`K=5`、每轮 500 步 × 7 UAV × 6 动作 = 21 000 次 `env.step`，
总开销约 75 秒（CPU 单进程）。

---

## 四、接真实 LLM 的最小改动

把 `CannedLLM` 换成下面这种子类即可，下游不需要任何变更：

```python
import requests

class DeepSeekR1(LLMBackend):
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": "deepseek-r1-7b",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.7},
            timeout=120,
        )
        text = r.json()["choices"][0]["message"]["content"]
        # 期望 LLM 输出 ```python ... ``` 包围的代码
        if "```python" in text:
            text = text.split("```python", 1)[1].split("```", 1)[0]
        return text.strip() + "\n"
```

要还原论文里"训练 R^best 给 MAPPO 用"：

```python
from reproduction.reward.manual_reward import compute_reward_v2  # 已有的稠密奖励基线
# 把 R^best 注入：在 compute_reward_v2 末尾追加 LRS 奖励项的缩放并求和
# 这里只暴露 best_fn 给 train.py；具体接入属于 M5 整合（在 README 末尾补充）
```

---

## 五、故意简化的地方

| 简化 | 现在 | 升级时 |
|---|---|---|
| LLM 后端 | `CannedLLM` 用预置候选伪装 LLM | 接 DeepSeek-R1-7B，复现论文 §V-A 的真实跑分 |
| 贪心选动作的"联合" | 各 UAV 独立取 argmax | 真正的联合贪心要遍历 `6^N_UAV`，可改为序列决策+蒙特卡洛采样 |
| 评估 Episode 数 | 每个候选只用 1 个 episode 的轨迹 | 论文评估范式相同（单 episode），但论文 §V-A 用了 8 个独立种子 |
| 反馈提示拼接 | 字符串直拼 | 加入 token 限长截断 / 改用对话历史结构 |
| 安全检查 | LLM 生成的代码直接 `exec` | 接入沙箱（RestrictedPython / docker）隔离 |
| 没接 MAPPO | 只输出 `R^best` 的 Python 字符串 | M5 注入到训练 loop 的 reward 项里 |

---

## 六、LRS 给 MAPPO 训练用时的位置（接入示意）

```python
# reproduction/train.py 的 reward 装配处
from reproduction.lrs import LRS

# 训练前: 跑一次 LRS 拿 R^best 的代码
lrs = LRS(K=5, seed=args.seed)
best_fn, *_ = lrs.run(SearchEnv(seed=args.seed))

def mappo_reward(env_info, n, action, prev_au):
    # R^best 项 + M2 手写稠密项（α 加权融合）
    return 0.7 * best_fn(env_info, n, action, prev_au) \
         + 0.3 * manual_shaping(env_info, n, action, prev_au)
```

完整 M5 整合（DPES patch + LRS reward + MAPPO 统一流水线）记在后续 PR。
