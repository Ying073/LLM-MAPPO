#!/usr/bin/env bash
# =============================================================================
# cloud_run.sh —— [M12] 在云 GPU 上一次跑完 8 seed × 3000 ep 的 LLM-MAPPO
# =============================================================================
# 目的: 复现论文 §V 的三件套 (DPES + LRS + MAPPO) 在 8 个独立训练 seed 下的
#       mean±std 闭环, 取代 M7/M10 的"单 seed 150 ep 冒烟"。
#
# 关键设计 (对比 M11):
#   * LRS K=5 只在本地 seed=42 跑过一次 → R^best = R1_K5_seed42_Rbest_fig11.py
#     这里 8 个训练 seed 全部 --lrs-cache 加载同一份 R^best (论文协议:
#     "LLM 离线生成一次 R^best, 然后 8 个 training seed 各训一遍, 报 mean±std"),
#     所以云端不消耗 DeepSeek API, 也不需要 DEEPSEEK_API_KEY。
#   * 8 seed × 3000 ep × batch_envs(=16), 每 seed 单独一颗独立网络, 各自存一份 npz。
#   * 跑完自动调 plot_8seed.py 出 mean±std 图。
#
# 用法 (在云端项目根目录):  bash cloud/cloud_run.sh
#   可选环境变量:
#     SEEDS       默认为 "0 1 2 3 4 5 6 7"
#     EPS_PER_SEED  默认为 3000
#     BATCH_ENVS    默认为 16
#     DEVICE        默认为 "cuda"
#     RESULTS_DIR   默认为 "reproduction/m12_results"
#
# 预计耗时: 8 seed each ~187 outer × ~8.7s (4090D) ≈ 27min/seed → ~3.6h 总计。
# 预计费用: ~3.6h × ¥1.88/h ≈ ¥7。
# =============================================================================
set -euo pipefail

# -- 解析脚本目录, 保证 cd 到哪都能跑 --
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# -- 可覆盖默认值 --
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
EPS_PER_SEED="${EPS_PER_SEED:-3000}"
BATCH_ENVS="${BATCH_ENVS:-16}"
DEVICE="${DEVICE:-cuda}"
RESULTS_DIR="${RESULTS_DIR:-reproduction/m12_results}"

# LRS 预跑好的 R^best (跳过 API, 8 seed 共用)
LRS_CACHE="${LRS_CACHE:-reproduction/lrs_runs/R1_K5_seed42_Rbest_fig11.py}"

echo "==================================================================="
echo "[M12] cloud_run.sh"
echo "  seeds       = ${SEEDS}"
echo "  eps/seed    = ${EPS_PER_SEED}"
echo "  batch_envs  = ${BATCH_ENVS}"
echo "  device      = ${DEVICE}"
echo "  results_dir = ${RESULTS_DIR}"
echo "  lrs_cache   = ${LRS_CACHE}"
echo "==================================================================="

# -- 校验 R^best 缓存存在 --
if [[ ! -f "${LRS_CACHE}" ]]; then
    echo "ERROR: LRS cache not found: ${LRS_CACHE}" >&2
    echo "  请确认已上传 reproduction/lrs_runs/R1_K5_seed42_Rbest_fig11.py" >&2
    exit 1
fi

# -- 找 python --
PY="$(command -v python || command -v python3)"
if [[ -z "${PY}" ]]; then
    echo "ERROR: python not found in PATH" >&2
    exit 1
fi
echo "Using python: ${PY} ($(${PY} --version 2>&1))"

# -- 校验依赖 (不静默失败) --
echo "[check] torch + numpy + matplotlib..."
${PY} - <<'PYEOF'
import importlib
for mob in ("torch", "numpy", "matplotlib"):
    try:
        importlib.import_module(mob)
        print(f"  OK  {mob}")
    except ImportError as e:
        print(f"  MISSING {mob}: {e}")
        raise SystemExit(1)
PYEOF

mkdir -p "${RESULTS_DIR}"
echo "[run] results will land in ${RESULTS_DIR}"

# -- 逐 seed 训练, 每 seed 独立 npz / png --
for SEED in ${SEEDS}; do
    echo ""
    echo "-------------------------------------------------------------------"
    echo "[seed ${SEED}] training ${EPS_PER_SEED} ep x ${BATCH_ENVS} envs (lrs-cache=${LRS_CACHE})"
    echo "-------------------------------------------------------------------"
    OUT_PNG="${RESULTS_DIR}/m12_curve_seed${SEED}_${EPS_PER_SEED}.png"
    OUT_NPZ="${RESULTS_DIR}/batched_m12_seed${SEED}_${EPS_PER_SEED}.npz"
    ${PY} reproduction/train_batched.py \
        --batch-envs "${BATCH_ENVS}" \
        --total-episodes "${EPS_PER_SEED}" \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --use-dpes \
        --lrs-cache "${LRS_CACHE}" \
        --out-name "${OUT_PNG}" \
        --save-history "${OUT_NPZ}" \
        --log-every 20
done

# -- 汇总出图 --
echo ""
echo "==================================================================="
echo "[M12] all seeds done. Plotting 8-seed mean±std vs Canned baseline..."
echo "==================================================================="
# Canned 基线 npz 可选: 找不到就只出 8-seed 图 (plot_8seed.py 会跳过 canned)
CANNED_NPZ="${CANNED_NPZ:-reproduction/batched_m5_8seed.npz}"
if [[ ! -f "${CANNED_NPZ}" ]]; then
    echo "[warn] Canned baseline not found at ${CANNED_NPZ}; plotting 8-seed only."
    CANNED_NPZ=""
fi
${PY} reproduction/plot_8seed.py \
    --results-dir "${RESULTS_DIR}" \
    --canned "${CANNED_NPZ}" \
    --out "${RESULTS_DIR}/comparison_m12_8seed_vs_canned.png"

echo ""
echo "[M12] DONE. Artifacts:"
echo "  ${RESULTS_DIR}/batched_m12_seed{0..7}_${EPS_PER_SEED}.npz  (8 份 raw)"
echo "  ${RESULTS_DIR}/comparison_m12_8seed_vs_canned.png           (mean±std 对比图)"
echo ""
echo "下一步: 把 ${RESULTS_DIR}/ 用 rsync/scp 下载回本地, 写报告."
