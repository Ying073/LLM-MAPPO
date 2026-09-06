# M12 云 GPU 运行手册（B 方案：8 seed × 3000 ep）

> 目标：把 M11（真 DeepSeek-R1 当 LRS 后端，但只有单 seed 150 ep）**升级成 8 seed 闭环**，
> 复现论文 §V 三件套（DPES + LRS + MAPPO）在多个独立训练 seed 下的 mean±std 曲线。
>
> **B 不是论文级复现**——论文是 28 000 ep × 8 seed，我们这里 **3000 ep × 8 seed**（约 1/9）。
> 定位是「**机制验证 + 规模化趋势**」：能看出 R^best 在更多 ep 下是否收敛、8 seed 方差多大、
> 以及 vs Canned 的真实差距。**不代表 71.4% / 100% 这两个 headline 数字。**

---

## 一、跑什么、跑多久、多少钱（请你先对齐再下单）

| 项 | 值 |
|---|---|
| 种子数 | 8 个独立训练 seed（0–7，各一颗独立网络） |
| 每 seed ep | 3000 |
| batch_envs | 16（并行 env，只加速采样，**不是** 8 个 seed） |
| 单 seed outer | 3000 / 16 ≈ **188 outer** |
| 单 seed 耗时 | 188 × ~8.7s（4090D 实测外推）≈ **27 min** |
| **全部 8 seed** | ≈ **3.6 h GPU** |
| LRS K=5 | **不重跑**（用 M11 的 R^best 缓存，无需 DeepSeek API） |
| 参考费用 | 4090D ¥1.88/h × 3.6h ≈ **¥7** |

关键点：云端跑的是 **LRS 已经定好的 R^best**，所以 **不需要 API key、不烧 LLM 调用费**。
代价是我们**看不到** LRS 迭代本身的动态（M11 已经看过）。

### 论文与我们的真实差距（务必诚实写进报告）

| 维度 | 论文 | 我们（B 方案） | 差距 |
|---|---|---|---|
| 单 seed ep | 28 000 | 3 000 | **9.3 ×** |
| seed 数 | 8 | 8 | 持平 ✅ |
| LLM 后端 | DeepSeek-R1-7B | DeepSeek-R1（真） | 持平 ✅ |
| headline 数字 | 71.4% / 100% | **不可报** | 需 30k + 全部找不到目标场景 |

---

## 二、云端安装依赖

云 GPU 通常是 Ubuntu + CUDA 镜像。先建 venv 装依赖（不会污染系统）：

```bash
# 1. 装 python venv (若无)
sudo apt update && sudo apt install -y python3-venv

# 2. 建 venv 并激活
python3 -m venv ~/mappo_venv
source ~/mappo_venv/bin/activate

# 3. 装依赖 (pip 最快; 云端无需 conda)
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib
```

> `--index-url cu121` 会装带 CUDA 的 torch；不想指定就 `pip install torch`（可能装 CPU 版，
> 跑不了 GPU）。**建议一定要指定 CUDA 版本。**

验证 GPU 可用：

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

---

## 三、上传代码 + 数据

用本地 rsync 只传跟 M12 有关的文件（`cloud/`、`reproduction/`、`README*.md`、`paper.md` 等）：

```bash
# 在【本地】项目根目录执行，把代码推到云主机
rsync -avz --exclude='.git' --exclude='.workbuddy' \
  ./ user@CLOUD_IP:~/mappo/
```

**重要：不要漏掉这两个文件**（cloud_run.sh 依赖它）：
- `reproduction/lrs_runs/R1_K5_seed42_Rbest_fig11.py` —— M11 的 R^best（LRS 缓存）
- `reproduction/batched_m5_8seed.npz` —— Canned 8-env 基线（可选，用来做对比图；没有也能跑，只出 8-seed 图）

登录云主机：

```bash
ssh user@CLOUD_IP
cd ~/mappo
```

---

## 四、跑脚本

```bash
source ~/mappo_venv/bin/activate
bash cloud/cloud_run.sh
```

脚本会自动：
1. 校验 R^best 缓存文件存在
2. 逐 seed（0–7）各训 3000 ep × 16 env，把每个 seed 的 history 存到 `reproduction/m12_results/batched_m12_seed{i}_3000.npz`
3. 全部跑完自动出 `reproduction/m12_results/comparison_m12_8seed_vs_canned.png`

可以自定义参数：

```bash
SEEDS="0 1 2 3" bash cloud/cloud_run.sh            # 只跑 4 个 seed
EPS_PER_SEED=6000 bash cloud/cloud_run.sh          # 每 seed 6000 ep
CANNED_NPZ="" bash cloud/cloud_run.sh              # 不画 Canned 对比
```

> 想更省电可后台跑：`nohup bash cloud/cloud_run.sh > m12_run.log 2>&1 &`
> 但云主机一般只按小时计费，后台跑跟前台跑费用一样，建议直接前台盯着 log。

---

## 五、下载结果

跑完把 `m12_results/` 拉回本地：

```bash
# 本地执行
rsync -avz user@CLOUD_IP:~/mappo/reproduction/m12_results/ \
  ./reproduction/m12_results/
```

拿到的东西：
- `batched_m12_seed{0..7}_3000.npz`（8 份原始训练历史）
- `comparison_m12_8seed_vs_canned.png`（8-seed mean±std vs Canned 对比图）

---

## 六、诚实边界（写报告前必读）

1. **B 方案跑不出论文的 71.4% / 100%**。那些数字是 28 000 ep + "全部找不到目标" 场景下的结果。
   我们 3000 ep 只能看**趋势**（reward/area_unc 是否随 ep 收敛、8 seed 方差是否够小）。
2. **8 seed 是独立训练 seed，不是 LRS seed**。论文协议是「LLM 离线生成一次 R^best，8 个训练 seed 共用」，
   跟 cloud_run.sh 一致。我们 LRS 那一次是在 seed=42 下跑的，8 个训练 seed 与它无关。
3. **Canned 基线那个 npz 是"1 个网络共享 8 个并行 env"，不是真 8 seed**。它当 trend baseline 可以，
   但不能跟真 8-seed 的 mean±std 直接做严格统计检验。真要严格对比，得让 Canned 也跑 8 个独立 seed。
4. **如果时间/预算允许，你想补论文级**，就把 `EPS_PER_SEED` 提到 28 000，并确保场景是"全部目标找不到"
   的困难配置——但那是 8 × 188 × 15 ≈ **80h GPU**（¥150+），需另行评估。

---

## 七、M12 文件清单

| 文件 | 作用 |
|---|---|
| `cloud/cloud_run.sh` | 一键跑 8 seed × 3000 ep（共用 R^best 缓存） |
| `reproduction/plot_8seed.py` | 8 seed mean±std vs Canned 出图 |
| `reproduction/train_batched.py` | 新增 `--lrs-cache` 选项（免重跑 LRS） |
| `reproduction/lrs_runs/R1_K5_seed42_Rbest_fig11.py` | M11 的 R^best（LRS 缓存，8 seed 共用） |
| `README_compare_with_paper.md` | 差距表同步更新（seed ✓ / LLM ✓ / ep 9.3×） |
