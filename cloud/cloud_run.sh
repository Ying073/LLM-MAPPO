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
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
RESUME_FROM="${RESUME_FROM:-}"
CODE_VERSION="${CODE_VERSION:-unknown}"

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
TEE_OPTION=""
if [[ -n "${RESUME_FROM}" ]]; then
  [[ -f "${RESUME_FROM}" ]] || { echo "Resume checkpoint not found: ${RESUME_FROM}" >&2; exit 1; }
  mkdir -p "${RUN_DIR}"
  TEE_OPTION="-a"
else
  mkdir "${RUN_DIR}"
fi

echo "Paper-aligned ${RUN_KIND} run"
echo "project=${PROJECT_ROOT}"
echo "episodes=${TRAIN_EPISODES}, batch_envs=${BATCH_ENVS}, train_seed=${TRAIN_SEED}"
echo "artifacts=${RUN_DIR}"
echo "checkpoint_every=${CHECKPOINT_EVERY}, resume_from=${RESUME_FROM:-none}"

{
  echo "code_version=${CODE_VERSION}"
  echo "run_kind=${RUN_KIND}"
  echo "episodes=${TRAIN_EPISODES}"
  echo "batch_envs=${BATCH_ENVS}"
  echo "train_seed=${TRAIN_SEED}"
  echo "device=${DEVICE}"
  sha256sum reproduction/train_batched.py reproduction/algorithms/mappo.py \
    reproduction/env/search_env.py reproduction/env/batched_env_wrapper.py \
    reproduction/reward/paper_reward.py reproduction/select_checkpoint.py cloud/cloud_run.sh
} > "${RUN_DIR}/run_manifest.txt"

"${PY}" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable in the selected Python environment"
print("GPU:", torch.cuda.get_device_name(0))
PY

TRAIN_ARGS=(
  --batch-envs "${BATCH_ENVS}"
  --total-episodes "${TRAIN_EPISODES}"
  --rollout-len 500
  --seed "${TRAIN_SEED}"
  --device "${DEVICE}"
  --use-dpes
  --reward-source paper-rbest
  --out-name "${RUN_DIR}/training_curve.png"
  --save-history "${RUN_DIR}/training_history.npz"
  --checkpoint-out "${RUN_DIR}/policy.pt"
  --training-checkpoint "${RUN_DIR}/training_state.pt"
  --policy-checkpoint-dir "${RUN_DIR}/policy_checkpoints"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --log-every 10
)
if [[ -n "${RESUME_FROM}" ]]; then
  TRAIN_ARGS+=(--resume-from "${RESUME_FROM}")
fi

"${PY}" reproduction/train_batched.py \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee ${TEE_OPTION} "${RUN_DIR}/training.log"

# Reproduction-only model selection: validation seeds are disjoint from final test seeds.
"${PY}" reproduction/select_checkpoint.py \
  --checkpoints "${RUN_DIR}"/policy_checkpoints/policy_outer_*.pt \
  --seeds 100 101 102 103 \
  --device "${DEVICE}" \
  --max-steps 500 \
  --out "${RUN_DIR}/checkpoint_selection.json" \
  2>&1 | tee ${TEE_OPTION} "${RUN_DIR}/checkpoint_selection.log"

SELECTED_POLICY="$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_checkpoint"])' "${RUN_DIR}/checkpoint_selection.json")"

# Paper §V-A: freeze the validation-selected policy, then test on eight independent seeds.
"${PY}" reproduction/evaluate.py \
  --checkpoint "${SELECTED_POLICY}" \
  --seeds 0 1 2 3 4 5 6 7 \
  --device "${DEVICE}" \
  --max-steps 500 \
  --out "${RUN_DIR}/evaluation_8seeds.json" \
  2>&1 | tee ${TEE_OPTION} "${RUN_DIR}/evaluation.log"

echo "Completed: ${RUN_DIR}"
