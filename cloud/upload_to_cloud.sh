#!/usr/bin/env bash
# Upload source code to the configured austlab SSH host.
# Existing files may be replaced, so updating a non-empty target requires ALLOW_UPDATE=1.
set -euo pipefail

REMOTE="${REMOTE:-austlab}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/lihaitao202413767/liuqiying2025313900/llm-mappo}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NONEMPTY="$(ssh "${REMOTE}" "if [ -d '${REMOTE_ROOT}' ] && [ \"\$(find '${REMOTE_ROOT}' -mindepth 1 -maxdepth 1 -print -quit)\" ]; then echo yes; else echo no; fi")"
if [[ "${NONEMPTY}" == yes && "${ALLOW_UPDATE:-0}" != 1 ]]; then
  echo "Remote target is non-empty: ${REMOTE}:${REMOTE_ROOT}" >&2
  echo "Review the target first, then rerun with ALLOW_UPDATE=1 to update source files." >&2
  exit 2
fi

ssh "${REMOTE}" "mkdir -p '${REMOTE_ROOT}'"
scp -r \
  "${PROJECT_ROOT}/README.md" \
  "${PROJECT_ROOT}/HOW_TO_RESUME.md" \
  "${PROJECT_ROOT}/reproduction" \
  "${PROJECT_ROOT}/cloud" \
  "${PROJECT_ROOT}/tests" \
  "${REMOTE}:${REMOTE_ROOT}/"

echo "Uploaded to ${REMOTE}:${REMOTE_ROOT}"
