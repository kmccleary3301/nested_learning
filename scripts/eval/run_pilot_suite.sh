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
HOPE_CONT_CHECKPOINTS=${HOPE_CONT_CHECKPOINTS:-}
PASSKEY_SAMPLES=${PASSKEY_SAMPLES:-64}
PASSKEY_FILLER=${PASSKEY_FILLER:-256}
PG19_SAMPLES=${PG19_SAMPLES:-32}
CONT_PLOT_SEGMENT=${CONT_PLOT_SEGMENT:-refinedweb_2018}
MEMORIZE_PATHS=${MEMORIZE_PATHS:-titan,cms_fast}
MEMORIZE_SURPRISE_THRESHOLD=${MEMORIZE_SURPRISE_THRESHOLD:-0.02}

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
if [[ -z "${HOPE_CONT_CHECKPOINTS}" ]]; then
  HOPE_CONT_CHECKPOINTS="${HOPE_CHECKPOINT}"
fi

mkdir -p eval
IFS=' ' read -r -a HOPE_CONT_LIST <<< "${HOPE_CONT_CHECKPOINTS}"

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
    --memorize-use-correct-answer \
    --memorize-paths "${MEMORIZE_PATHS}" \
    --memorize-surprise-threshold "${MEMORIZE_SURPRISE_THRESHOLD}"
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
    --memorize-use-correct-answer \
    --memorize-paths "${MEMORIZE_PATHS}" \
    --memorize-surprise-threshold "${MEMORIZE_SURPRISE_THRESHOLD}"
}

run_continual() {
  local config=$1
  local tag=$2
  shift 2
  local ckpts=("$@")
  if [[ ${#ckpts[@]} -eq 0 ]]; then
    echo "[eval] No checkpoints provided for continual eval (${tag}); skipping."
    return
  fi
  UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python scripts/eval/continual.py \
    --config "${config}" \
    --checkpoints "${ckpts[@]}" \
    --segments-yaml "${SEGMENTS_YAML}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --batch-size "${CONT_BATCH}" \
    --max-batches "${CONT_MAX_BATCHES}" \
    --device "${DEVICE}" \
    --output "eval/continual_${tag}.json" \
    --memorize \
    --memorize-steps 1 \
    --memorize-paths "${MEMORIZE_PATHS}" \
    --memorize-surprise-threshold "${MEMORIZE_SURPRISE_THRESHOLD}"
  if [[ ${#ckpts[@]} -gt 1 ]]; then
    local plot_target="reports/plots/continual_${tag}_${CONT_PLOT_SEGMENT}.png"
    mkdir -p reports/plots
    UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python scripts/eval/plot_forgetting.py \
      --continual-json "eval/continual_${tag}.json" \
      --segment "${CONT_PLOT_SEGMENT}" \
      --output "${plot_target}"
    echo "[eval] Forgetting plot saved to ${plot_target}"
  fi
}

run_passkey() {
  local config=$1
  local ckpt=$2
  local tag=$3
  UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python scripts/eval/passkey.py \
    --config "${config}" \
    --checkpoint "${ckpt}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --samples "${PASSKEY_SAMPLES}" \
    --filler-sentences "${PASSKEY_FILLER}" \
    --device "${DEVICE}" \
    --output "eval/passkey_${tag}.json" \
    --memorize \
    --memorize-steps 2 \
    --memorize-paths "${MEMORIZE_PATHS}" \
    --memorize-surprise-threshold "${MEMORIZE_SURPRISE_THRESHOLD}"
}

run_pg19() {
  local config=$1
  local ckpt=$2
  local tag=$3
  UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python scripts/eval/pg19_perplexity.py \
    --config "${config}" \
    --checkpoint "${ckpt}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --max-samples "${PG19_SAMPLES}" \
    --device "${DEVICE}" \
    --output "eval/pg19_${tag}.json" \
    --memorize \
    --memorize-paths "${MEMORIZE_PATHS}" \
    --memorize-surprise-threshold "${MEMORIZE_SURPRISE_THRESHOLD}"
}

echo "[eval] Running suite for HOPE (${HOPE_CHECKPOINT})"
run_zero_shot "${HOPE_CONFIG}" "${HOPE_CHECKPOINT}" "pilot"
run_niah "${HOPE_CONFIG}" "${HOPE_CHECKPOINT}" "pilot"
run_continual "${HOPE_CONFIG}" "pilot" "${HOPE_CONT_LIST[@]}"
run_passkey "${HOPE_CONFIG}" "${HOPE_CHECKPOINT}" "pilot"
run_pg19 "${HOPE_CONFIG}" "${HOPE_CHECKPOINT}" "pilot"

if [[ -n "${TITAN_CONFIG:-}" && -n "${TITAN_CHECKPOINT:-}" ]]; then
  echo "[eval] Running suite for TITAN baseline (${TITAN_CHECKPOINT})"
  run_zero_shot "${TITAN_CONFIG}" "${TITAN_CHECKPOINT}" "titan"
  run_niah "${TITAN_CONFIG}" "${TITAN_CHECKPOINT}" "titan"
  IFS=' ' read -r -a TITAN_CONT_LIST <<< "${TITAN_CHECKPOINTS:-$TITAN_CHECKPOINT}"
  run_continual "${TITAN_CONFIG}" "titan" "${TITAN_CONT_LIST[@]}"
  run_passkey "${TITAN_CONFIG}" "${TITAN_CHECKPOINT}" "titan"
  run_pg19 "${TITAN_CONFIG}" "${TITAN_CHECKPOINT}" "titan"
else
  echo "[eval] TITAN baseline skipped (set TITAN_CONFIG and TITAN_CHECKPOINT to enable)."
fi

echo "[eval] Pilot suite complete. Outputs saved under eval/."
