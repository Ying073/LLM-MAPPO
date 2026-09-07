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
  pilot)  EVAL_SEEDS_DEFAULT="300 301 302 303 304 305 306 307"; TRAIN_EPISODES="${TRAIN_EPISODES:-3000}" ;;
  formal) EVAL_SEEDS_DEFAULT="400 401 402 403 404 405 406 407"; TRAIN_EPISODES="${TRAIN_EPISODES:-28000}" ;;
  *) echo "RUN_KIND must be pilot or formal" >&2; exit 2 ;;
esac
EVAL_SEEDS="${EVAL_SEEDS:-${EVAL_SEEDS_DEFAULT}}"
read -r -a EVAL_SEED_ARRAY <<< "${EVAL_SEEDS}"

if (( TRAIN_EPISODES % BATCH_ENVS != 0 )); then
  echo "TRAIN_EPISODES must be divisible by BATCH_ENVS for an exact episode count." >&2
  exit 2
fi
if (( ${#EVAL_SEED_ARRAY[@]} != 8 )); then
  echo "EVAL_SEEDS must contain exactly 8 integer seeds." >&2
  exit 2
fi
if [[ ! -x "${PY}" ]]; then
  echo "Python environment not found: ${PY}" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-reproduction/paper_aligned_runs/${RUN_KIND}_seed${TRAIN_SEED}_${STAMP}}"
TEE_OPTION=""
RESUME_SOURCE=""
if [[ -n "${RESUME_FROM}" ]]; then
  [[ -f "${RESUME_FROM}" ]] || { echo "Resume checkpoint not found: ${RESUME_FROM}" >&2; exit 1; }
  RESUME_SOURCE="$(realpath "${RESUME_FROM}")"
  mkdir -p "${RUN_DIR}"
  TEE_OPTION="-a"
  MANIFEST_NAME="run_manifest_resume_${STAMP}.txt"
else
  mkdir "${RUN_DIR}"
  MANIFEST_NAME="run_manifest.txt"
fi

echo "Paper-aligned ${RUN_KIND} run"
echo "project=${PROJECT_ROOT}"
echo "episodes=${TRAIN_EPISODES}, batch_envs=${BATCH_ENVS}, train_seed=${TRAIN_SEED}"
echo "evaluation_seeds=${EVAL_SEEDS}"
echo "artifacts=${RUN_DIR}"
echo "checkpoint_every=${CHECKPOINT_EVERY}, resume_from=${RESUME_FROM:-none}"

{
  echo "code_version=${CODE_VERSION}"
  echo "run_kind=${RUN_KIND}"
  echo "episodes=${TRAIN_EPISODES}"
  echo "batch_envs=${BATCH_ENVS}"
  echo "train_seed=${TRAIN_SEED}"
  echo "evaluation_seeds=${EVAL_SEEDS}"
  echo "device=${DEVICE}"
  echo "resume_from=${RESUME_SOURCE:-none}"
  if [[ -n "${RESUME_SOURCE}" ]]; then
    sha256sum "${RESUME_SOURCE}"
  fi
  sha256sum reproduction/train_batched.py reproduction/evaluate.py \
    reproduction/algorithms/mappo.py reproduction/algorithms/networks.py \
    reproduction/algorithms/buffer.py reproduction/algorithms/dpes.py \
    reproduction/algorithms/batched_dpes.py reproduction/env/search_env.py \
    reproduction/env/batched_search_env.py reproduction/env/env_wrapper.py \
    reproduction/env/batched_env_wrapper.py reproduction/reward/paper_reward.py cloud/cloud_run.sh
} > "${RUN_DIR}/${MANIFEST_NAME}"

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
  TRAIN_ARGS+=(--resume-from "${RESUME_SOURCE}")
fi

"${PY}" reproduction/train_batched.py \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee ${TEE_OPTION} "${RUN_DIR}/training.log"

# Paper Algorithm 3 / §V-A: after all episodes, freeze and test the final
# stochastic policy on eight independent seeds. The paper does not specify
# validation-based selection of an intermediate checkpoint.
"${PY}" reproduction/evaluate.py \
  --checkpoint "${RUN_DIR}/policy.pt" \
  --seeds "${EVAL_SEED_ARRAY[@]}" \
  --device "${DEVICE}" \
  --max-steps 500 \
  --out "${RUN_DIR}/evaluation_8seeds.json" \
  2>&1 | tee ${TEE_OPTION} "${RUN_DIR}/evaluation.log"

echo "Completed: ${RUN_DIR}"
