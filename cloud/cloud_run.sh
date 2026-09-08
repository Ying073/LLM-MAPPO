#!/usr/bin/env bash
# Paper-aligned training + fixed-policy evaluation on a remote GPU server.
# Default is a 3,000-episode pilot. Set RUN_KIND=formal for 28,000 episodes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_KIND="${RUN_KIND:-pilot}"
TRAIN_SEED="${TRAIN_SEED:-0}"
BATCH_ENVS="${BATCH_ENVS:-20}"
DEVICE="${DEVICE:-cuda}"
PY="${PY:-${PROJECT_ROOT}/.venv/bin/python}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
SELECTION_EVERY="${SELECTION_EVERY:-100}"
ENTROPY_COEF="${ENTROPY_COEF:-0.01}"
RESUME_FROM="${RESUME_FROM:-}"
CODE_VERSION="${CODE_VERSION:-unknown}"

case "${RUN_KIND}" in
  pilot)  EVAL_SEEDS_DEFAULT="300 301 302 303 304 305 306 307"; TRAIN_EPISODES="${TRAIN_EPISODES:-3000}" ;;
  formal) EVAL_SEEDS_DEFAULT="400 401 402 403 404 405 406 407"; TRAIN_EPISODES="${TRAIN_EPISODES:-28000}" ;;
  *) echo "RUN_KIND must be pilot or formal" >&2; exit 2 ;;
esac
EVAL_SEEDS="${EVAL_SEEDS:-${EVAL_SEEDS_DEFAULT}}"
VALIDATION_SEEDS="${VALIDATION_SEEDS:-100 101 102 103 104 105 106 107}"
read -r -a EVAL_SEED_ARRAY <<< "${EVAL_SEEDS}"
read -r -a VALIDATION_SEED_ARRAY <<< "${VALIDATION_SEEDS}"

if (( TRAIN_EPISODES % BATCH_ENVS != 0 )); then
  echo "TRAIN_EPISODES must be divisible by BATCH_ENVS for an exact episode count." >&2
  exit 2
fi
if ! [[ "${SELECTION_EVERY}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SELECTION_EVERY must be a positive integer." >&2
  exit 2
fi
if (( ${#EVAL_SEED_ARRAY[@]} != 8 )); then
  echo "EVAL_SEEDS must contain exactly 8 integer seeds." >&2
  exit 2
fi
if (( ${#VALIDATION_SEED_ARRAY[@]} != 8 )); then
  echo "VALIDATION_SEEDS must contain exactly 8 integer seeds." >&2
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
echo "validation_seeds=${VALIDATION_SEEDS}"
echo "artifacts=${RUN_DIR}"
echo "checkpoint_every=${CHECKPOINT_EVERY}, resume_from=${RESUME_FROM:-none}"
echo "entropy_coef=${ENTROPY_COEF}, selection_every=${SELECTION_EVERY}"

{
  echo "code_version=${CODE_VERSION}"
  echo "run_kind=${RUN_KIND}"
  echo "episodes=${TRAIN_EPISODES}"
  echo "batch_envs=${BATCH_ENVS}"
  echo "train_seed=${TRAIN_SEED}"
  echo "evaluation_seeds=${EVAL_SEEDS}"
  echo "validation_seeds=${VALIDATION_SEEDS}"
  echo "entropy_coef=${ENTROPY_COEF}"
  echo "selection_every=${SELECTION_EVERY}"
  echo "device=${DEVICE}"
  echo "resume_from=${RESUME_SOURCE:-none}"
  if [[ -n "${RESUME_SOURCE}" ]]; then
    sha256sum "${RESUME_SOURCE}"
  fi
  sha256sum reproduction/train_batched.py reproduction/evaluate.py reproduction/select_checkpoint.py \
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
  --entropy-coef "${ENTROPY_COEF}"
  --log-every 10
)
if [[ -n "${RESUME_FROM}" ]]; then
  TRAIN_ARGS+=(--resume-from "${RESUME_SOURCE}")
fi

"${PY}" reproduction/train_batched.py \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee ${TEE_OPTION} "${RUN_DIR}/training.log"

# The paper does not publish a checkpoint-selection rule. Select a frozen
# policy on disjoint validation seeds so late PPO collapse cannot silently
# replace an earlier, stronger policy. Keep the final test seeds untouched.
CHECKPOINT_CANDIDATES=()
while IFS= read -r checkpoint; do
  checkpoint_name="$(basename "${checkpoint}" .pt)"
  checkpoint_outer="${checkpoint_name##*_}"
  if (( 10#${checkpoint_outer} % SELECTION_EVERY == 0 )); then
    CHECKPOINT_CANDIDATES+=("${checkpoint}")
  fi
done < <(find "${RUN_DIR}/policy_checkpoints" -maxdepth 1 -type f -name 'policy_outer_*.pt' | sort -V)
if [[ -n "${RESUME_SOURCE}" ]]; then
  CHECKPOINT_CANDIDATES+=("${RESUME_SOURCE}")
fi
CHECKPOINT_CANDIDATES+=("${RUN_DIR}/policy.pt")

"${PY}" reproduction/select_checkpoint.py \
  --checkpoints "${CHECKPOINT_CANDIDATES[@]}" \
  --seeds "${VALIDATION_SEED_ARRAY[@]}" \
  --device "${DEVICE}" \
  --max-steps 500 \
  --out "${RUN_DIR}/checkpoint_selection_validation.json" \
  2>&1 | tee ${TEE_OPTION} "${RUN_DIR}/checkpoint_selection.log"

SELECTED_CHECKPOINT="$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_checkpoint"])' "${RUN_DIR}/checkpoint_selection_validation.json")"
echo "selected_checkpoint=${SELECTED_CHECKPOINT}"

"${PY}" reproduction/evaluate.py \
  --checkpoint "${SELECTED_CHECKPOINT}" \
  --seeds "${EVAL_SEED_ARRAY[@]}" \
  --device "${DEVICE}" \
  --max-steps 500 \
  --out "${RUN_DIR}/evaluation_8seeds.json" \
  2>&1 | tee ${TEE_OPTION} "${RUN_DIR}/evaluation.log"

echo "Completed: ${RUN_DIR}"
