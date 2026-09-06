#!/usr/bin/env bash
# M12 download — 把云端跑出来的 m12_results 拉回本地
# 用法: 在本地 (PowerShell 或 WSL) 跑
#   bash cloud/download_from_cloud.sh <user@host:port> [project-name]
# 例:
#   bash cloud/download_from_cloud.sh root@connect.bj1.autodl.com:24022 llm_mappo

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "用法: $0 <user@host:port> [project-name]"
    echo "例:   $0 root@connect.bj1.autodl.com:24022 llm_mappo"
    exit 1
fi

REMOTE="$1"
PROJECT_NAME="${2:-llm_mappo}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST_DIR="${PROJECT_ROOT}/reproduction/m12_results"

echo "==================================================="
echo "  [download] remote : $REMOTE:/root/$PROJECT_NAME/reproduction/m12_results"
echo "  [download] local  : $DEST_DIR"
echo "==================================================="

mkdir -p "$DEST_DIR"
rsync -avz --progress \
    -e ssh -o "StrictHostKeyChecking=no" \
    "$REMOTE:/root/$PROJECT_NAME/reproduction/m12_results/" \
    "$DEST_DIR/"

echo ""
echo "[download] done. 拉回的内容 (在 $DEST_DIR/):"
ls -la "$DEST_DIR/"
echo ""
echo "[download] NEXT: 本地跑 plot:"
echo "  ${PY:-python} reproduction/plot_8seed.py \\"
echo "    --results-dir reproduction/m12_results \\"
echo "    --canned batched_m5_8seed.npz \\"
echo "    --out reproduction/m12_results/comparison_m12_8seed_vs_canned.png"
echo ""
echo "[download] 看图前先把 batched_m5_8seed.npz 复制到 reproduction/m12_results/ 或在 --canned 给绝对路径"
