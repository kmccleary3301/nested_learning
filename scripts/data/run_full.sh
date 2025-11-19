#!/usr/bin/env bash
set -euo pipefail

# General controls
TOKENIZER_MANIFEST=${TOKENIZER_MANIFEST:-configs/data/refinedweb_mixture.yaml}
TOKENIZER_OUTPUT_DIR=${TOKENIZER_OUTPUT_DIR:-artifacts/tokenizer/refinedweb_mix}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-${TOKENIZER_OUTPUT_DIR}/spm_32000_unigram.model}
VOCAB_SIZE=${VOCAB_SIZE:-32000}
TOKENIZER_LOG=${TOKENIZER_LOG:-data/mixtures/refinedweb_mix_tokenizer_full.json}
MIXTURE_CONFIG=${MIXTURE_CONFIG:-configs/data/refinedweb_mixture_full.yaml}
SHARD_LOG=${SHARD_LOG:-data/mixtures/refinedweb_mix_full_shards.json}
FORCE_FILTER=${FORCE_FILTER:-0}
RETRAIN_TOKENIZER=${RETRAIN_TOKENIZER:-0}

mkdir -p data/filtered data/shards artifacts/tokenizer data/mixtures

filter_dataset() {
  local name=$1
  local dataset=$2
  local subset=$3
  local split=$4
  local text_column=$5
  local limit=$6
  local output=$7
  local target_lang=${8:-en}
  local lang_threshold=${9:-0.85}
  local min_chars=${10:-200}
  local max_chars=${11:-12000}

  if [[ "${FORCE_FILTER}" != "1" && -f "${output}" ]]; then
    echo "[Data][${name}] Found existing ${output}, skipping filter step (set FORCE_FILTER=1 to rebuild)"
    return
  fi

  echo "[Data][${name}] Filtering ${dataset}${subset:+/${subset}} -> ${output}"
  cmd=(uv run python scripts/data/filter_corpus.py
    --dataset "${dataset}"
    --split "${split}"
    --text-column "${text_column}"
    --target-lang "${target_lang}"
    --lang-threshold "${lang_threshold}"
    --min-chars "${min_chars}"
    --max-chars "${max_chars}"
    --output-path "${output}"
    --force-exit)
  if [[ -n "${subset}" ]]; then
    cmd+=(--subset "${subset}")
  fi
  if [[ -n "${limit}" ]]; then
    cmd+=(--limit "${limit}")
  fi
  "${cmd[@]}"
}

echo "[Data] === Stage 1: Filtering corpora ==="
filter_dataset "refinedweb" \
  "${RW_DATASET:-HuggingFaceFW/fineweb}" \
  "${RW_SUBSET:-sample-10BT}" \
  "${RW_SPLIT:-train}" \
  "${RW_TEXT_COLUMN:-text}" \
  "${RW_LIMIT:-100000}" \
  "${RW_OUTPUT:-data/filtered/refinedweb_en_full.txt}" \
  "${RW_LANG:-en}" \
  "${RW_LANG_THRESHOLD:-0.85}" \
  "${RW_MIN_CHARS:-200}" \
  "${RW_MAX_CHARS:-8000}"

filter_dataset "wikipedia" \
  "${WIKI_DATASET:-wikimedia/wikipedia}" \
  "${WIKI_SUBSET:-20231101.en}" \
  "${WIKI_SPLIT:-train}" \
  "${WIKI_TEXT_COLUMN:-text}" \
  "${WIKI_LIMIT:-50000}" \
  "${WIKI_OUTPUT:-data/filtered/wikipedia_en_full.txt}" \
  "${WIKI_LANG:-en}" \
  "${WIKI_LANG_THRESHOLD:-0.85}" \
  "${WIKI_MIN_CHARS:-200}" \
  "${WIKI_MAX_CHARS:-8000}"

filter_dataset "c4" \
  "${C4_DATASET:-allenai/c4}" \
  "${C4_SUBSET:-en}" \
  "${C4_SPLIT:-train}" \
  "${C4_TEXT_COLUMN:-text}" \
  "${C4_LIMIT:-50000}" \
  "${C4_OUTPUT:-data/filtered/c4_en_full.txt}" \
  "${C4_LANG:-en}" \
  "${C4_LANG_THRESHOLD:-0.85}" \
  "${C4_MIN_CHARS:-200}" \
  "${C4_MAX_CHARS:-8000}"

filter_dataset "redpajama" \
  "${RPJ_DATASET:-cerebras/SlimPajama-627B}" \
  "${RPJ_SUBSET:-}" \
  "${RPJ_SPLIT:-train}" \
  "${RPJ_TEXT_COLUMN:-text}" \
  "${RPJ_LIMIT:-50000}" \
  "${RPJ_OUTPUT:-data/filtered/redpajama_en_full.txt}" \
  "${RPJ_LANG:-en}" \
  "${RPJ_LANG_THRESHOLD:-0.85}" \
  "${RPJ_MIN_CHARS:-200}" \
  "${RPJ_MAX_CHARS:-8000}"

filter_dataset "code" \
  "${CODE_DATASET:-codeparrot/codeparrot-clean-train}" \
  "${CODE_SUBSET:-}" \
  "${CODE_SPLIT:-train}" \
  "${CODE_TEXT_COLUMN:-content}" \
  "${CODE_LIMIT:-50000}" \
  "${CODE_OUTPUT:-data/filtered/code_en_full.txt}" \
  "${CODE_LANG:-en}" \
  "${CODE_LANG_THRESHOLD:-0.50}" \
  "${CODE_MIN_CHARS:-200}" \
  "${CODE_MAX_CHARS:-16000}"

echo "[Data] === Stage 2: Tokenizer training ==="
if [[ ! -f "${TOKENIZER_MODEL}" || "${RETRAIN_TOKENIZER}" == "1" ]]; then
  uv run python scripts/data/train_tokenizer.py \
    --manifest "${TOKENIZER_MANIFEST}" \
    --vocab-size "${VOCAB_SIZE}" \
    --output-dir "${TOKENIZER_OUTPUT_DIR}" \
    --log-file "${TOKENIZER_LOG}"
else
  echo "[Data] Tokenizer already exists at ${TOKENIZER_MODEL}; set RETRAIN_TOKENIZER=1 to rebuild."
fi

echo "[Data] === Stage 3: Sharding filtered corpora ==="
uv run python scripts/data/process_mixture.py \
  "${MIXTURE_CONFIG}" \
  --tokenizer-path "${TOKENIZER_MODEL}" \
  --log-file "${SHARD_LOG}"

echo "[Data] Full pipeline complete."
