#!/usr/bin/env bash
#
# Convenience wrapper to run the Stage 2 evaluation suite (zero-shot, NIAH, continual)
# on the pilot HOPE checkpoint and optional TITAN baseline.
#
# Environment variables (override as needed):
#   HOPE_CONFIG          (default configs/pilot.yaml)
#   HOPE_CHECKPOINT      (default artifacts/checkpoints/pilot/step_latest.pt)
#   TITAN_CONFIG         (optional)
#   TITAN_CHECKPOINT     (optional)
#   TOKENIZER_PATH       (default artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model)
#   DEVICE               (default cuda:1)
#   MAX_SAMPLES          (default 256 for zero-shot)
#   NIAH_CONTEXTS        (space-separated list, default "2048 4096 8192 16384 32768 65536")
#   NIAH_SAMPLES         (default 8 per context)
#   CONT_BATCH           (default 4)
#   CONT_MAX_BATCHES     (default 20)

set -euo pipefail

HOPE_CONFIG=${HOPE_CONFIG:-configs/pilot.yaml}
TOKENIZER_PATH=${TOKENIZER_PATH:-artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model}
DEVICE=${DEVICE:-cuda:1}
MAX_SAMPLES=${MAX_SAMPLES:-256}
NIAH_CONTEXTS=${NIAH_CONTEXTS:-"2048 4096 8192 16384 32768 65536"}
NIAH_SAMPLES=${NIAH_SAMPLES:-8}
CONT_BATCH=${CONT_BATCH:-4}
CONT_MAX_BATCHES=${CONT_MAX_BATCHES:-20}
SEGMENTS_YAML=${SEGMENTS_YAML:-configs/data/continual_segments_sample.yaml}

resolve_checkpoint() {
  local path="$1"
  if [[ -n "${path}" ]]; then
    echo "${path}"
    return
  fi
  local latest
  latest=$(ls -1t artifacts/checkpoints/pilot/step_*.pt 2>/dev/null | head -n 1 || true)
  if [[ -z "${latest}" ]]; then
    echo ""
  else
    echo "${latest}"
  fi
}

HOPE_CHECKPOINT=${HOPE_CHECKPOINT:-$(resolve_checkpoint "")}
if [[ -z "${HOPE_CHECKPOINT}" ]]; then
  echo "[eval] No HOPE checkpoint supplied and none found under artifacts/checkpoints/pilot."
  exit 1
fi

mkdir -p eval

run_zero_shot() {
  local config=$1
  local ckpt=$2
  local tag=$3
  UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python scripts/eval/zeroshot.py \
    --config "${config}" \
    --checkpoint "${ckpt}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --tasks all \
    --max-samples "${MAX_SAMPLES}" \
    --device "${DEVICE}" \
    --output "eval/zeroshot_${tag}.json" \
    --memorize \
    --memorize-steps 2 \
    --memorize-use-correct-answer
}

run_niah() {
  local config=$1
  local ckpt=$2
  local tag=$3
  local args=()
  for ctx in ${NIAH_CONTEXTS}; do
    args+=(--context-lengths "${ctx}")
  done
  UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python scripts/eval/niah.py \
    --config "${config}" \
    --checkpoint "${ckpt}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    "${args[@]}" \
    --samples-per-length "${NIAH_SAMPLES}" \
    --device "${DEVICE}" \
    --output "eval/niah_${tag}.json" \
    --memorize \
    --memorize-steps 2 \
    --memorize-use-correct-answer
}

run_continual() {
  local config=$1
  local ckpt=$2
  local tag=$3
  UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python scripts/eval/continual.py \
    --config "${config}" \
    --checkpoints "${ckpt}" \
    --segments-yaml "${SEGMENTS_YAML}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --batch-size "${CONT_BATCH}" \
    --max-batches "${CONT_MAX_BATCHES}" \
    --device "${DEVICE}" \
    --output "eval/continual_${tag}.json" \
    --memorize \
    --memorize-steps 1
}

echo "[eval] Running suite for HOPE (${HOPE_CHECKPOINT})"
run_zero_shot "${HOPE_CONFIG}" "${HOPE_CHECKPOINT}" "pilot"
run_niah "${HOPE_CONFIG}" "${HOPE_CHECKPOINT}" "pilot"
run_continual "${HOPE_CONFIG}" "${HOPE_CHECKPOINT}" "pilot"

if [[ -n "${TITAN_CONFIG:-}" && -n "${TITAN_CHECKPOINT:-}" ]]; then
  echo "[eval] Running suite for TITAN baseline (${TITAN_CHECKPOINT})"
  run_zero_shot "${TITAN_CONFIG}" "${TITAN_CHECKPOINT}" "titan"
  run_niah "${TITAN_CONFIG}" "${TITAN_CHECKPOINT}" "titan"
  run_continual "${TITAN_CONFIG}" "${TITAN_CHECKPOINT}" "titan"
else
  echo "[eval] TITAN baseline skipped (set TITAN_CONFIG and TITAN_CHECKPOINT to enable)."
fi

echo "[eval] Pilot suite complete. Outputs saved under eval/."
