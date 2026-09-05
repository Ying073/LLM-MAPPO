# LLM-MAPPO 复现

复现论文：**Multi-UAV Trajectory Planning for Dynamic Target Search: An LLM-Enhanced Multi-Agent Reinforcement Learning Algorithm**（IEEE TCCN, 2026）

> 目标：把论文三大组件（**MAPPO + DPES 双模式信息素 + LRS 离线 LLM 奖励塑形**）全部跑通，每个组件都有"代码 ↔ 公式"对照文档。
> 当前进展:**M0–M5 全部闭环**——从仿真到 LLM 生成奖励到端到端联合训练。

## 里程碑与状态

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 环境准备（隔离 Python 环境 + PyTorch + 目录） | ✅ |
| M1 | 仿真环境 `env/search_env.py`（公式 4–11） | ✅ |
| M2 | MAPPO 基线 + 手写稠密奖励 | ✅（reward v2 修复 credit assignment 后能学） |
| M3 | DPES 双模式信息素（公式 13–17）+ A/B 训练 | ✅ |
| M4 | LRS 离线 LLM 奖励塑形（公式 20–22, 27–29） | ✅ K=5 闭环验证 η_k 单调 |
| M5 | **LLM-MAPPO 端到端**:DPES 信息素 + LRS 奖励 R^best + MAPPO 联合训练 | ✅ 150ep 跑通 |

## 目录结构

```
reproduction/
├── README.md                    # 本文件（项目入口）
├── README_mappo.md              # M2:MAPPO + 手写奖励代码 ↔ 论文对照表
├── README_dpes.md               # M3:DPES 代码 ↔ 论文对照表
├── README_lrs.md                # M4:LRS 代码 ↔ 论文对照表
├── README_llm_mappo.md          # M5:LLM-MAPPO 端到端整合(架构/接入点/跑法/结果)
├── requirements.txt             # Python 依赖(pip 用)
├── env/
│   ├── search_env.py            # M1:仿真环境(完整实现)
│   ├── env_wrapper.py           # M2/M3/M5:MAPPO 接口 + DPES patch + LRS 注入点
│   └── README_search_env.md     # env ↔ 公式对照
├── algorithms/
│   ├── networks.py              # Actor / Critic MLP
│   ├── buffer.py                # MAPPO rollout buffer + GAE
│   ├── mappo.py                 # PPO update(公式 23–26)
│   └── dpes.py                  # M3:信息素场更新(公式 13–17)
├── reward/
│   └── manual_reward.py         # M2/M3:手写稠密奖励(v2 信用分配修复版)
├── lrs.py                       # M4:LLM 奖励塑形主循环(公式 20–22, 27–29)
├── train.py                     # 训练入口 --use-dpes / --use-lrs / --save-history
├── plot_dpes_ablation.py        # M3 DPES A/B 对比图
└── plot_llm_mappo_comparison.py # M5 M2/M3/M5 端到端曲线对比图
```

## 环境安装（两种方式，二选一）

### 方式 A：venv（推荐，最稳）

已建好，路径在：
```
C:\Users\lenovo\.workbuddy\binaries\python\envs\llm_mappo\
```

如果以后想重装：
```bash
"C:\Users\lenovo\.workbuddy\binaries\python\versions\3.13.12\python.exe" -m venv "C:\Users\lenovo\.workbuddy\binaries\python\envs\llm_mappo"
"C:\Users\lenovo\.workbuddy\binaries\python\envs\llm_mappo\Scripts\pip.exe" install -r requirements.txt
```

### 方式 B：conda（如果你坚持用 conda）

环境已创建在 `C:\Users\lenovo\anaconda3\envs\llm_mappo`，但自动装 matplotlib 时遇到 sandbox 拦截。你可以手动装，**绕过 conda transaction**（不走 `conda install`，直接 pip 装）：

```bash
conda activate llm_mappo
pip install --force-reinstall --no-deps "matplotlib<3.9"
pip install -r requirements.txt
```

> ⚠️ matplotlib 必须 ≤3.8.x：3.9+ 在 numpy 2.x 上有 circular import bug。

### PyTorch（M2 阶段才需要）

到 pytorch.org 选你 CUDA 对应的命令，例如 CUDA 12.1：

```bash
# 任一方式下都一致
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

CPU 用户：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 跑通验证（M1）

**用 venv（推荐）**：
```bash
"C:\Users\lenovo\.workbuddy\binaries\python\envs\llm_mappo\Scripts\python.exe" "C:\Users\lenovo\AI\大创\LLM-MAPPO_论文阅读与复现\reproduction\env\search_env.py"
```

**用 conda**：
```bash
conda activate llm_mappo
cd "C:\Users\lenovo\AI\大创\LLM-MAPPO_论文阅读与复现\reproduction\env"
python search_env.py
```

期望看到（最后会尝试弹一张图，没图形界面的环境会 warning 但不影响）：

```
[init] t=0, area_uncertainty=1.0000, obstacles=20, targets=15
[end ] t=500, area_uncertainty=0.16XX, searched=...
```

- `area_uncertainty`: 初始 1.0，500 步后应**明显下降**（随机策略下大概降到 0.10–0.20）
- `searched`: 应有不少 > 0 的目标网格

如果数值不对，说明贝叶斯更新或全局融合出 bug——把数字贴回来。

## M1 任务书（已实现）

参考论文 §III（系统模型，公式 1–11）和表 I 的参数。完整代码在 `env/search_env.py`，代码 ↔ 公式对应关系在 `env/README_search_env.md`。一眼看完就是这张表：

| 函数 | 公式 | 作用 |
|---|---|---|
| `reset()` | — | 随机摆好障碍物 / 目标 / UAV |
| `_sensing_domain(n)` | 公式 4 | UAV $n$ 高度档决定能看 1/5/9 网格 |
| `_detect(n,ix,iy)` | 公式 5 | 真有目标按 $P^D$，无目标按 $P^F$ 抽样 $D$ |
| `_update_maps()` | 公式 6 + 7 | $D=1$ 升 $p$ / $D=0$ 降 $p$ / 不感知不变；然后算熵 |
| `_fuse_maps()` | 公式 8 + 10 | `geum = min(leum)`，gtpm = 对应 UAV 的 $p$ |
| `area_uncertainty()` | 公式 9 | `geum` 的均值 |
| `step()` 末尾 | 公式 11 | $\zeta=1$ 且 $p \ge \xi$ 记为"已搜到" |

**验收：** `python env/search_env.py` 跑通，area uncertainty 从 1.0 单调下降到 < 0.5。

## 跑通 M4（LRS）

```bash
"C:/Users/lenovo/.workbuddy/binaries/python/envs/llm_mappo/Scripts/python.exe" -c "
import sys; sys.path.insert(0, 'C:/Users/lenovo/AI/大创/LLM-MAPPO_论文阅读与复现')
from reproduction.env.search_env import SearchEnv
from reproduction.lrs import LRS
env = SearchEnv(seed=0)
lrs = LRS(K=5, seed=0)
best_fn, best_code, best_J, best_metrics = lrs.run(env, seed=0)
print('Best J =', best_J, 'area_unc =', best_metrics['area_unc'])
"
```

期望：
- 5 次迭代打印 `[LRS k=k/5] J=... | best so far J=...`
- η_k 单调不减（验证了公式 27）
- 大约 1–2 分钟跑完（CPU 单进程）
- 最终 `R^best` = 在 K 次候选里 J 最高的奖励函数

## 跑通 M3（DPES + MAPPO）

```bash
"C:/Users/lenovo/.workbuddy/binaries/python/envs/llm_mappo/Scripts/python.exe" train.py --total-episodes 120 --seed 42 --use-dpes
```

期望：`training_curve.png` 与 `training_curve_dpes_ablation.png` 在 `reproduction/` 下生成。DPES 在前 ~30 episode 拉开差距（reward -163 vs -230），no-DPES 在 ~120 episode 追平（详见 `plot_dpes_ablation.py`）。

## 跑通 M2（MAPPO 基线）

```bash
"C:/Users/lenovo/.workbuddy/binaries/python/envs/llm_mappo/Scripts/python.exe" train.py --total-episodes 150 --seed 42
```

期望：reward 从 -260 升到 -50 左右，searched 10 → 20+（自动验证 `manual_reward.py` v2 修复生效）。

## 跑通 M5（LLM-MAPPO 端到端）

```bash
"C:/Users/lenovo/.workbuddy/binaries/python/envs/llm_mappo/Scripts/python.exe" \
    reproduction/train.py --total-episodes 150 --seed 42 \
    --use-dpes --use-lrs --lrs-K 5 \
    --out-name training_curve_llm_mappo.png
```

总时间 ~10 min（CPU）：
- LRS 离线评估 K=5 候选：~85 s（打印 `[LRS k=k/5]` 进度）
- MAPPO 训练 150 episode：~6–8 min
- 输出：`training_curve_llm_mappo.png`（reward / searched / area_unc 三联图）

**期望趋势**：
- M5 reward 起点就在 **+400 ~ +500**（R^best 本身给正信号），不是从负爬起
- `searched` 稳步上 25+
- `area_unc` 早期被 DPES+R^best 协同拖到 ~0.15，训练中可能浮动但不应恶化

需要 raw history 给对比图：在命令后加 `--save-history hist_m5.npz`。
多组对比用 `plot_llm_mappo_comparison.py`：

```bash
python reproduction/plot_llm_mappo_comparison.py \
    --inputs hist_m2.npz hist_m3.npz hist_m5.npz \
    --labels MAPPO MAPPO+DPES LLM-MAPPO \
    --out comparison_llm_mappo.png
```

---

## 跑通 M6（GPU 加速 + conda 环境）

环境位置：`C:\Users\lenovo\anaconda3\envs\llm_mappo`（**conda 不用 venv**）

```bash
# 1. conda llm_mappo 装 torch+CUDA
"C:/Users/lenovo/anaconda3/envs/llm_mappo/python.exe" -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu130

# 2. 验证 GPU
"C:/Users/lenovo/anaconda3/envs/llm_mappo/python.exe" -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望: True NVIDIA GeForce RTX 5060 Laptop GPU

# 3. 跑 M5 GPU 版
"C:/Users/lenovo/anaconda3/envs/llm_mappo/python.exe" \
    "C:/Users/lenovo/AI/大创/LLM-MAPPO_论文阅读与复现/reproduction/train.py" \
    --total-episodes 150 --seed 42 \
    --use-dpes --use-lrs --lrs-K 5 \
    --device cuda --out-name training_curve_m6_final.png --save-history hist_m6_final.npz
```

**M6 实测 (conda llm_mappo + RTX 5060)**：

| 阶段 | 耗时 | 备注 |
|---|---|---|
| LRS 离线 (K=5) | ~80 s | K=3 约 50 s，K=5 约 80 s |
| MAPPO 训练 150 ep | ~7 min | 含 GPU PPO update（minibatch 256 默认）|
| 30 ep GPU 烟测 | ~2.5 min | 估 CPU 同样跑 30 ep ~3-4 min |
| **加速比** | **~1.2-1.5×** | 受任务规模限制：actor 64→6, critic 595→1, rollout 3500 样本 |

**GPU 化经验教训**（M6 验证过的结论，写在这里避免重蹈）：

1. **小张量 (< 1万元素) GPU 反而更慢**：曾经把 search_env 的 `_update_maps` / `_step_targets` 改成 torch tensor 整张图一次算，实测 CUDA 3.12ms/step vs CPU 1.99ms/step。GPU launch overhead 1ms > 计算本身 10μs。**已撤回**，env 保持 v1 numpy 实现。
2. **GPU 真正的加速对象是 batched matmul**（PPO update），但本任务 actor 64→6、critic 595→1 都是小网络，rollout 3500 样本 ÷ minibatch 256 = 14 步/epoch × 4 epoch = 56 步 PPO update，加速上限就是 1.2-1.5×。
3. **想再榨 5-10×** 需要改 wrapper 让 env 也是 batched 跑（同时 32 个 env 共享一个 batched tensor），1-2 小时 wrapper 改造。
4. **minibatch_size 默认从 64 改 256**（M6），GPU 适合大 batch。
5. **conda llm_mappo 环境 matplotlib 雷区**：装 torch 时会留 fonttools 双重 dist-info 残骸，触发 `Cannot uninstall: no RECORD file`。手动修复见 memory 2026-09-05 M6 段（wheel download → zipfile extract → rm + cp）。

**何时该 GPU / 何时该 CPU**：
- ✅ GPU：训练时（PPO update）+ 大量并行 rollout 时
- ✅ CPU：env step（小张量 + 频繁 launch）+ LRS 候选评估（贪心 rollout 在 env 上）

---
