# Data Pipeline (Stage 2)

This document explains how to generate tokenizer artifacts and token shards for Stage 2 training.

## Prerequisites
- Ensure the `uv` environment is synced (`uv sync --all-extras`).
- Large storage mounted at `data/raw/` and `data/shards/`.
- HF datasets cache configured with valid credentials if accessing gated sets.

## 1. Train tokenizer (multi-corpus manifest)

```bash
uv run python scripts/data/train_tokenizer.py \
  --manifest configs/data/refinedweb_mixture.yaml \
  --vocab-size 32000 \
  --output-dir artifacts/tokenizer/refinedweb_mix \
  --log-file data/mixtures/refinedweb_mix_tokenizer.json
```

The manifest pulls small samples from FineWeb (RefinedWeb proxy), Wikimedia/Wikipedia, AllenAI C4, SlimPajama, and codeparrot code datasets. Outputs live in `artifacts/tokenizer/refinedweb_mix/`.

## 2. Shard mixture components

```bash
uv run python scripts/data/process_mixture.py \
  configs/data/refinedweb_mixture_filtered.yaml \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --log-file data/mixtures/refinedweb_mix_filtered_shards.json
```

This iterates over each dataset entry, streams up to the specified `max_records`, tokenizes at sequence length 512, and writes NumPy shards to `data/shards/<dataset>`. Stats (records, sequences, shards) are recorded in the log file. For the filtered run, each dataset was preprocessed via `scripts/data/filter_corpus.py` to produce cleaned text files under `data/filtered/`.

## 3. Legacy pilot data
- `data/shards/tinystories_train/` retains 1,718 shards for unit tests and smoke runs.

## 4. Filtering & deduplication
Before sharding full-scale corpora, run language filtering + dedup to keep only high-quality English segments:

```bash
uv run python scripts/data/filter_corpus.py \
  --dataset HuggingFaceFW/fineweb \
  --subset sample-10BT \
  --split train \
  --text-column text \
  --output-path data/filtered/fineweb_en.txt \
  --min-chars 200 \
  --max-chars 8000 \
  --lang-threshold 0.85
```

Adjust dataset/subset arguments per manifest entry. The script enforces language probabilities via `langdetect`, performs length screening, and deduplicates using a rolling hash window. Point `scripts/data/process_mixture.py` to these filtered files (or custom dataset definitions) for large-scale processing.

## 5. Artifacts & stats
- Tokenizer samples: `data/mixtures/refinedweb_mix_tokenizer.json`
- Shard stats (streamed sample): `data/mixtures/refinedweb_mix_shards.json`
- Shard stats (filtered local files): `data/mixtures/refinedweb_mix_filtered_shards.json`
- Tokenizer model: `artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model`

## 6. Next steps
- Swap manifest sample limits with full-scale counts once storage and bandwidth permit, then rerun the commands above.
- Implement dedup/language-ID filtering pre-sharding.
- Version mixture manifests and stats under `configs/data/` as recipes change.
