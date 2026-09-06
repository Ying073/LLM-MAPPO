#!/usr/bin/env bash
# M12 upload — 从本地项目根目录同步到云端 GPU 机
# 用法: 在本地 (PowerShell 或 WSL) 跑
#   bash cloud/upload_to_cloud.sh <user@host:port> [project-name]
# 例:
#   bash cloud/upload_to_cloud.sh root@connect.bj1.autodl.com:24022 llm_mappo

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "用法: $0 <user@host:port> [project-name]"
    echo "例:   $0 root@connect.bj1.autodl.com:24022 llm_mappo"
    exit 1
fi

REMOTE="$1"
PROJECT_NAME="${2:-llm_mappo}"

# 本地项目根 (脚本所在目录的上一级)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==================================================="
echo "  [upload] local  : $PROJECT_ROOT"
echo "  [upload] remote : $REMOTE:/root/$PROJECT_NAME"
echo "==================================================="

# rsync 排除项:
#   .git                — 历史里有(云端不需要)
#   __pycache__         — Python 缓存(云端自己生成)
#   *.pyc               — 同上
#   .workbuddy          — WorkBuddy 沙盒数据(>50MB,云端用不到)
#   *.npz (项目根的)   — 我们本地很多 npz,云端只要 R^best 依赖链 + 可能要的 canned 基线
#   *.png (项目根的)  — 本地图
#   reproduction/lrs_runs/ 里的 cache  .py 文件 (M12 必须保留 — R^best 源)
#   tests/ / docs/     — 大创项目可能混杂, 先排除
rsync -avz --progress \
    -e ssh -o "StrictHostKeyChecking=no" \
    --exclude=".git" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    --exclude=".workbuddy" \
    --exclude=".vscode" \
    --exclude="*.tmp" \
    --include="reproduction/lrs_runs/R1_K5_seed42_Rbest_fig11.py" \
    --include="reproduction/lrs_runs/R1_K5_seed42_log_fig11.txt" \
    --include="reproduction/lrs_runs/" \
    --exclude="reproduction/lrs_runs/R1_K1_seed42_Rbest.py" \
    --exclude="reproduction/lrs_runs/*.npz" \
    --exclude="*.npz" \
    --exclude="*.png" \
    --exclude="*.pdf" \
    --exclude="*.md" \
    "$PROJECT_ROOT/" \
    "$REMOTE:/root/$PROJECT_NAME/"

echo ""
echo "[upload] done. 上传的内容:"
echo "  - 全部 reproduction/ 包代码 (env / algorithms / train_batched.py)"
echo "  - R^best 缓存: lrs_runs/R1_K5_seed42_Rbest_fig11.py (115 行真 R1 设计)"
echo "  - cloud/cloud_run.sh 一键脚本"
echo "  - cloud/CLOUD_RUN.md 运行手册"
echo ""
echo "[upload] NEXT: ssh 到云端 (见 CLOUD_RUN.md §二)"
