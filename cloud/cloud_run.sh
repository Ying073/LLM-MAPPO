#!/usr/bin/env bash
# Paper-aligned training + fixed-policy evaluation on AutoDL.
# Default is a 3,000-episode pilot. Set RUN_KIND=formal for 28,000 episodes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_KIND="${RUN_KIND:-pilot}"
TRAIN_SEED="${TRAIN_SEED:-0}"
BATCH_ENVS="${BATCH_ENVS:-20}"
DEVICE="${DEVICE:-cuda}"
PY="${PY:-/root/mappo_venv/bin/python}"

case "${RUN_KIND}" in
  pilot)  TRAIN_EPISODES="${TRAIN_EPISODES:-3000}" ;;
  formal) TRAIN_EPISODES="${TRAIN_EPISODES:-28000}" ;;
  *) echo "RUN_KIND must be pilot or formal" >&2; exit 2 ;;
esac

if (( TRAIN_EPISODES % BATCH_ENVS != 0 )); then
  echo "TRAIN_EPISODES must be divisible by BATCH_ENVS for an exact episode count." >&2
  exit 2
fi
if [[ ! -x "${PY}" ]]; then
  echo "Python environment not found: ${PY}" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-reproduction/paper_aligned_runs/${RUN_KIND}_seed${TRAIN_SEED}_${STAMP}}"
mkdir "${RUN_DIR}"

echo "Paper-aligned ${RUN_KIND} run"
echo "project=${PROJECT_ROOT}"
echo "episodes=${TRAIN_EPISODES}, batch_envs=${BATCH_ENVS}, train_seed=${TRAIN_SEED}"
echo "artifacts=${RUN_DIR}"

"${PY}" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable in the selected Python environment"
print("GPU:", torch.cuda.get_device_name(0))
PY

"${PY}" reproduction/train_batched.py \
  --batch-envs "${BATCH_ENVS}" \
  --total-episodes "${TRAIN_EPISODES}" \
  --rollout-len 500 \
  --seed "${TRAIN_SEED}" \
  --device "${DEVICE}" \
  --use-dpes \
  --reward-source paper-rbest \
  --out-name "${RUN_DIR}/training_curve.png" \
  --save-history "${RUN_DIR}/training_history.npz" \
  --checkpoint-out "${RUN_DIR}/policy.pt" \
  --log-every 10 \
  2>&1 | tee "${RUN_DIR}/training.log"

# Paper §V-A: freeze the learned policy, then test on eight independent seeds.
"${PY}" reproduction/evaluate.py \
  --checkpoint "${RUN_DIR}/policy.pt" \
  --seeds 0 1 2 3 4 5 6 7 \
  --device "${DEVICE}" \
  --max-steps 500 \
  --out "${RUN_DIR}/evaluation_8seeds.json" \
  2>&1 | tee "${RUN_DIR}/evaluation.log"

echo "Completed: ${RUN_DIR}"
