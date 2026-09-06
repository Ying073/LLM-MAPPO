#!/usr/bin/env bash
# Download one named paper-aligned run without overwriting an existing local copy.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <run-directory-name>" >&2
  exit 2
fi

REMOTE="${REMOTE:-autodl}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/projects/llm-mappo}"
RUN_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_ROOT="${PROJECT_ROOT}/reproduction/downloaded_runs"
DEST="${DEST_ROOT}/${RUN_NAME}"

if [[ -e "${DEST}" ]]; then
  echo "Local destination already exists; refusing to overwrite: ${DEST}" >&2
  exit 2
fi
mkdir -p "${DEST_ROOT}"
scp -r "${REMOTE}:${REMOTE_ROOT}/reproduction/paper_aligned_runs/${RUN_NAME}" "${DEST_ROOT}/"
echo "Downloaded to ${DEST}"
