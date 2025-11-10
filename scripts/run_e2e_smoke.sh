#!/usr/bin/env bash
set -euo pipefail

DEVICE=${DEVICE:-cpu}
TRAIN_CONFIG=${TRAIN_CONFIG:-pilot_smoke}
DEFAULT_MODEL_CONFIG="configs/${TRAIN_CONFIG}.yaml"
if [[ ! -f "${DEFAULT_MODEL_CONFIG}" ]]; then
  DEFAULT_MODEL_CONFIG="configs/hope/pilot.yaml"
fi
MODEL_CONFIG=${MODEL_CONFIG:-${DEFAULT_MODEL_CONFIG}}
TOKENIZER_PATH=${TOKENIZER_PATH:-artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-artifacts/checkpoints/${TRAIN_CONFIG}}
LOG_PATH=${LOG_PATH:-logs/${TRAIN_CONFIG}_release.json}
EVAL_OUTPUT=${EVAL_OUTPUT:-eval/zeroshot_smoke.json}
TASKS=${TASKS:-piqa}
MAX_SAMPLES=${MAX_SAMPLES:-32}

step() {
  echo
  echo "[$(date +%H:%M:%S)] $1"
  echo "------------------------------------------------------------"
}

step "1/4: Syncing environment (uv sync --all-extras)"
uv sync --all-extras

step "2/4: Preparing filtered sample data"
uv run bash scripts/data/run_sample.sh

step "3/4: Running ${TRAIN_CONFIG} smoke training on device=${DEVICE}"
mkdir -p "$(dirname "${LOG_PATH}")"
uv run python train.py \
  --config-name "${TRAIN_CONFIG}" \
  train.device="${DEVICE}" \
  logging.enabled=true \
  logging.backend=json \
  logging.path="${LOG_PATH}" \
  train.checkpoint.enable=true \
  train.checkpoint.dir="${CHECKPOINT_DIR}" \
  train.checkpoint.save_interval=999999 \
  train.checkpoint.save_last=true

if ! ls "${CHECKPOINT_DIR}"/step_*.pt >/dev/null 2>&1; then
  echo "No checkpoints found in ${CHECKPOINT_DIR}. Training may have failed."
  exit 1
fi
LATEST_CKPT=$(ls -1 "${CHECKPOINT_DIR}"/step_*.pt | sort | tail -n 1)
echo "[Info] Using checkpoint ${LATEST_CKPT}"

step "4/4: Running zero-shot eval (${TASKS})"
mkdir -p "$(dirname "${EVAL_OUTPUT}")"
uv run python scripts/eval/zeroshot.py \
  --config "${MODEL_CONFIG}" \
  --checkpoint "${LATEST_CKPT}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --tasks "${TASKS}" \
  --max-samples "${MAX_SAMPLES}" \
  --output "${EVAL_OUTPUT}" \
  --device "${DEVICE}"

echo
echo "[Done] Logs -> ${LOG_PATH}"
echo "[Done] Checkpoint -> ${LATEST_CKPT}"
echo "[Done] Eval metrics -> ${EVAL_OUTPUT}"
