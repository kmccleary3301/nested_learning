#!/usr/bin/env bash
set -euo pipefail

TOKENIZER_MODEL=${1:-artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model}

if [[ ! -f "data/filtered/refinedweb_en_sample.txt" ]]; then
  echo "[Data] Creating filtered RefinedWeb sample"
  uv run python scripts/data/filter_corpus.py \
    --dataset HuggingFaceFW/fineweb \
    "--subset=sample-10BT" \
    --split train \
    --text-column text \
    --target-lang en \
    --lang-threshold 0.85 \
    --min-chars 200 \
    --max-chars 8000 \
    --limit 2000 \
    --output-path data/filtered/refinedweb_en_sample.txt \
    --force-exit
fi

if [[ ! -f "data/filtered/wikipedia_en_sample.txt" ]]; then
  echo "[Data] Creating filtered Wikipedia sample"
  uv run python scripts/data/filter_corpus.py \
    --dataset wikimedia/wikipedia \
    "--subset=20231101.en" \
    --split train \
    --text-column text \
    --target-lang en \
    --lang-threshold 0.85 \
    --min-chars 200 \
    --max-chars 8000 \
    --limit 1000 \
    --output-path data/filtered/wikipedia_en_sample.txt \
    --force-exit
fi

if [[ ! -f "data/filtered/c4_en_sample.txt" ]]; then
  echo "[Data] Creating filtered C4 sample"
  uv run python scripts/data/filter_corpus.py \
    --dataset allenai/c4 --subset en --split train \
    --text-column text --target-lang en --lang-threshold 0.85 \
    --min-chars 200 --max-chars 8000 --limit 1000 \
    --output-path data/filtered/c4_en_sample.txt --force-exit
fi

if [[ ! -f "data/filtered/redpajama_en_sample.txt" ]]; then
  echo "[Data] Creating filtered SlimPajama sample"
  uv run python scripts/data/filter_corpus.py \
    "--dataset=cerebras/SlimPajama-627B" \
    --split train \
    --text-column text \
    --target-lang en \
    --lang-threshold 0.85 \
    --min-chars 200 \
    --max-chars 8000 \
    --limit 1000 \
    --output-path data/filtered/redpajama_en_sample.txt \
    --force-exit
fi

if [[ ! -f "data/filtered/code_en_sample.txt" ]]; then
  echo "[Data] Creating filtered code sample"
  uv run python scripts/data/filter_corpus.py \
    --dataset codeparrot/codeparrot-clean-train --split train \
    --text-column content --target-lang en --lang-threshold 0.5 \
    --min-chars 200 --max-chars 12000 --limit 1000 \
    --output-path data/filtered/code_en_sample.txt --force-exit
fi

# Train tokenizer if it doesn't exist 
if [[ ! -f "${TOKENIZER_MODEL}" ]]; then
  echo "[Data] Training tokenizer (vocab_size=32000, model_type=unigram)"
  uv run python scripts/data/train_tokenizer.py \
    --manifest configs/data/refinedweb_mixture_sample.yaml \
    --vocab-size 32000 \
    --output-dir artifacts/tokenizer/refinedweb_mix \
    --log-file data/mixtures/refinedweb_mix_tokenizer.json
else
  echo "[Data] Tokenizer already exists at ${TOKENIZER_MODEL}"
fi

echo "[Data] Sharding filtered samples"
uv run python scripts/data/process_mixture.py \
  configs/data/refinedweb_mixture_filtered.yaml \
  --tokenizer-path ${TOKENIZER_MODEL} \
  --log-file data/mixtures/refinedweb_mix_filtered_shards.json

echo "[Data] Sample pipeline complete"
