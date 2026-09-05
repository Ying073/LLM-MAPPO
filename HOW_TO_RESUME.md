# HOW_TO_RESUME · LLM-MAPPO 复现接力指南

> 写给**之后接手这个项目的人**（你自己/同学/导师），让你能在 10 分钟内搞清楚"项目状态"+"从这里开始能做什么"+"哪里有雷"。

---

## 0. 30 秒状态速览

| 维度 | 状态 |
|---|---|
| 论文 | Liu et al. 2026, IEEE TCCN, DOI 10.1109/TCCN.2026.3710519 |
| 复现完整度 | 算法机制 **M0-M7 全闭环 ✅**；量化数字（71.4% 搜索时间缩短）**不可报** |
| 训练规模 | 150 episode × 1 seed（论文 28 000 × 8 seed）|
| 训练时间 | 单次 150 ep GPU ~6-10 分钟 |
| GPU 加速 | 1.2-1.5×（小任务规模上限）|
| LLM 后端 | `CannedLLM` 占位（接真 DeepSeek-R1-7B 需 API key）|
| 最近 commit | `78d04e3` M0-M7 |
| remote | `https://github.com/Ying073/LLM-MAPPO.git` |

**一句话**：算法三件套（DPES + LRS + MAPPO）按论文实现跑通，但训练规模 + LLM 后端差 1-2 个数量级，**绝对数字不可比**。可报"机制对、规模小"。

---

## 1. 10 分钟上手

### 1.1 跑通

```bash
# 1. 激活环境（conda 不用 venv）
"C:/Users/lenovo/anaconda3/envs/llm_mappo/python.exe"

# 2. 跑 M5/M6 端到端 (DPES + LRS + MAPPO, 150 ep, GPU)
cd "C:/Users/lenovo/AI/大创/LLM-MAPPO_论文阅读与复现"
python reproduction/train.py --total-episodes 150 --seed 42 \
    --use-dpes --use-lrs --lrs-K 5 \
    --device cuda --out-name run_check.png
# 期望: ~10 分钟出图，reward +484 → +658
```

### 1.2 读懂

按这个顺序读 README：
1. `reproduction/README_compare_with_paper.md` — **先读这个**，了解"做了什么 + 距离论文差多少"
2. `reproduction/README_llm_mappo.md` — M5/M6 端到端整合
3. `reproduction/README_mappo.md` — M2 MAPPO 基线
4. `reproduction/README_dpes.md` — M3 DPES
5. `reproduction/README_lrs.md` — M4 LRS
6. `reproduction/env/README_search_env.md` — M1 仿真环境

每个 README 都有"代码 ↔ 公式"对照表，跟 `LLM-MAPPO_Markdown_Reader/paper.md` + `equations.md` 对应。

### 1.3 调参

| 想调 | 改哪里 |
|---|---|
| 训练 episode 数 | `train.py --total-episodes N` |
| 训练 seed | `train.py --seed N` |
| 关闭 DPES | 不加 `--use-dpes`（默认关）|
| 关闭 LRS | 不加 `--use-lrs`（默认关）|
| 关闭 GPU | 不加 `--device cuda`（默认 cpu）|
| PPO minibatch | `train.py --minibatch-size 256`（默认 256）|
| LRS 迭代次数 | `train.py --lrs-K 5`（默认 5）|
| 网络结构 | `algorithms/networks.py` Actor/Critic 类 |
| 奖励权重 | `reward/manual_reward.py` 常量 |
| DPES 参数 | `algorithms/dpes.py` 表 II 常量 |
| LLM 后端 | `lrs.py` 的 `CannedLLM` → 替换为 `DeepSeekR1` |

---

## 2. 代码地图

```
LLM-MAPPO_论文阅读与复现/
├── README.md                              # 项目入口
├── reproduction/                          # 复现代码包
│   ├── README_compare_with_paper.md       # ⭐ 先读这个
│   ├── README_llm_mappo.md                # M5/M6 整合
│   ├── README_*.md                        # M1-M4 各模块
│   ├── env/
│   │   ├── search_env.py                  # M1 仿真环境（公式 4-11）
│   │   ├── env_wrapper.py                 # MAPPO 接口 + DPES patch + LRS 注入
│   │   └── README_search_env.md           # M1 代码↔公式
│   ├── algorithms/
│   │   ├── networks.py                    # Actor/Critic MLP
│   │   ├── buffer.py                      # MAPPO rollout buffer + GAE
│   │   ├── mappo.py                       # PPO update (公式 23-26)
│   │   └── dpes.py                        # M3 DPES (公式 13-17)
│   ├── reward/
│   │   └── manual_reward.py               # M2/M3 手写稠密奖励
│   ├── lrs.py                             # M4 LRS (公式 20-22, 27-29)
│   ├── train.py                           # MAPPO 训练入口
│   ├── plot_dpes_ablation.py              # M3 A/B 对比图
│   └── plot_llm_mappo_comparison.py       # M2/M3/M5 对比图
├── LLM-MAPPO_Markdown_Reader/             # 论文阅读资产
│   ├── paper.md                           # 中英对照正文
│   ├── equations.md                       # 公式 1-29 索引
│   ├── source_map.json                    # 段落↔页码↔章节
│   ├── deconstruction.md                  # 深读笔记 + 自检 15 题
│   └── assets/fig*.png                    # 11 张图裁图
├── hist_*.npz                             # 训练 raw history（rewards/searched/au）
├── training_curve_*.png                   # 训练曲线（9 个）
└── comparison_*.png                       # 消融对比图
```

---

## 3. 已确认的雷区

### 3.1 matplotlib 雷区
- `conda llm_mappo` 装 torch 时会留 fonttools 双重 dist-info 残骸
- 触发 `Cannot uninstall: no RECORD file`
- 手动修复：wheel download → zipfile -e → rm + cp（详见 `README_llm_mappo.md` §9.5）

### 3.2 GPU 化小张量是反优化
- 7×9=63 个 Bayesian 更新在 torch tensor 上比 numpy 慢
- 原因：GPU launch overhead ~1ms > 计算本身 ~10μs
- **结论**：env 保持 numpy，GPU 只给 PPO update 用
- 已撤回 v2 torch 化版本，`search_env.py` 保持 v1 numpy 实现

### 3.3 GPU 加速上限 1.2-1.5×
- actor 64→6 / critic 595→1 小网络
- rollout 3500 样本 ÷ minibatch 256 = 14 步 PPO update/epoch × 4 epoch
- 想再榨 5-10× 需做 batched env（32 env 并行共享 batched tensor），1-2 小时 wrapper 改造

### 3.4 相对导入位置
- 包内 `reproduction/xxx.py` → `from .env.X import ...`（单点）
- `reproduction/sub/xxx.py` → `from ..env.X import ...`（两点）
- 混淆会 `ImportError: attempted relative import beyond top-level package`

### 3.5 tuple 里有函数对象别用 slice
- `B_R[idx] = (code, fn, J, metrics)`，要 `B_R[idx][2]` 取 J
- 不要 `B_R[idx][:3]` 赋给 `(a, b, c)`，否则 `b` 是函数对象，`{b:.3f}` 崩溃
- **显式索引比 slice 安全**

### 3.6 searched 数字口径
- 我们的 `searched = self.searched | newly_confirmed` 累计，但目标会游走
- 500 步累计"确认事件" 30+ ≠ 15 个目标实体
- 论文 §V-A "cumulative number of searched targets" 含义未明说，可能也是累计事件
- **不能拿 30 vs 15 做绝对值对比**

### 3.7 LRS 的 CannedLLM 不是真 LLM
- `CannedLLM` 用迭代号 k 路由到 R₁/R₃/R^best 三段 paper 已发表代码
- 论文用 DeepSeek-R1-7B 生成 + 推理指导
- **CannedLLM 跑通的 K=5 闭环验证的是"LRS 算法对"（公式 27 单调），不是"LLM 找到了论文式最优 R"**

---

## 4. 接力 TODO（按代价从小到大）

| 编号 | 任务 | 代价 | 价值 | 状态 |
|---|---|---|---|---|
| 8 | 接真 DeepSeek-R1-7B（user 提供 API key）| 30 min | 让 LRS 真用 LLM 推理 | 待 user 决定 |
| 9 | 跑 2000 ep × 3 seed（CPU ~3 小时）| 3 h | 3 seed mean ± std，可信度 ↑ | 待跑 |
| 10 | 跑 28 000 ep × 1 seed（CPU ~30 h）| 30 h | 规模上与论文齐平 | 待跑 |
| 11 | batched env 改造（wrapper）| 1-2 h | 让 GPU 加速到 5-10×，28000 ep 跑得起 | 待做 |
| 12 | 28 000 ep × 8 seed × DeepSeek（完整复现）| 30 h + API 费 | 论文 71.4% 数字可报 | follow-up |

**推荐路径**：8 → 11 → 9 → 12（先接 LLM + 做 batched 加速，再跑 3 seed）

---

## 5. 接 DeepSeek-R1-7B 的代码骨架

`lrs.py` 已经有 `LLMBackend` 抽象类，加一个新子类即可（伪代码）：

```python
# lrs.py 里加
import requests
class DeepSeekR1(LLMBackend):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.deepseek.com/v1/chat/completions"
    
    def generate(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {
            "model": "deepseek-reasoner",  # DeepSeek-R1
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
        }
        r = requests.post(self.url, json=data, headers=headers, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
```

然后改 `LRS.__init__`：
```python
# 原来
self.llm = CannedLLM()
# 改为
self.llm = DeepSeekR1(api_key=os.environ["DEEPSEEK_API_KEY"])
```

5 分钟接入，不用改任何其他代码。DeepSeek API 价格 ~¥0.001/1K token，5 次 LRS 迭代预计 < ¥1。

---

## 6. 接力的最快方法

1. **clone repo**：`git clone https://github.com/Ying073/LLM-MAPPO.git`
2. **建环境**：照 `reproduction/README_llm_mappo.md` §9.1 装 conda + torch+CUDA + matplotlib
3. **跑烟测**：上面 1.1 命令 10 分钟看是否出图
4. **读 README**：按 1.2 顺序，先 README_compare_with_paper.md
5. **接 LLM**：照 §5 加 DeepSeekR1 类
6. **跑更大规模**：照 §4 接力 TODO 选 8/9/10

---

## 7. 已知 Bug / TODO 代码注释

代码里搜 `# TODO` 看分散的待办点：
- `train.py` --use-lrs 启动时 LRS 评估与训练串行，可以并行加速
- `lrs.py` 没有 early-stop（η_k 已收敛就停）
- `dpes.py` 4-邻域扩散未做 cell boundary 处理
- `env_wrapper.py` action masking 用 -2/-1 惩罚，论文用 logit=-inf 更标准

---

*这份文档由 M7 commit `78d04e3` 一起落档。后续接力者请把变更记到 commit message 里，方便追踪。*
