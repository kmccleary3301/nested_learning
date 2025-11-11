# Data Pipeline (Stage 2)

This document explains how to generate tokenizer artifacts and token shards for Stage 2 training.

## Prerequisites
- Ensure the `uv` environment is synced (`uv sync --all-extras`).
- Large storage mounted at `data/raw/` and `data/shards/`.
- HF datasets cache configured with valid credentials if accessing gated sets.

## Dataset acquisition & licensing
The Stage 2 mixture mimics RefinedWeb + supplements. Download each source into `data/raw/<source>/` and document provenance before filtering.

| Source | License / Terms | Acquisition Command(s) | Notes |
|--------|-----------------|------------------------|-------|
| RefinedWeb / FineWeb proxy | CC BY 4.0 (FineWeb) | `uv run python scripts/data/shard_corpus.py --dataset HuggingFaceFW/fineweb --subset sample-10BT --split train --output data/raw/refinedweb.ndjsonl --limit 20000000` | Keep a copy of the HF dataset card; respect scraping policies. |
| Wikipedia 2023-12 dump | CC BY-SA 3.0 | Download `https://huggingface.co/datasets/wikipedia/20220301.en` via HF CLI or mirror the XML dump. | Use HF `datasets load_dataset` inside the filtering script to avoid storing raw XML. |
| C4 (en) | ODC-By | `uv run python scripts/data/shard_corpus.py --dataset allenai/c4 --subset en --split train --output data/raw/c4_en.ndjsonl --limit 8000000` | Heavy dataset; ensure disk quota before streaming. |
| RedPajama CC subset | CC BY | Use `togethercomputer/RedPajama-Data-1T-Sample` or the CC subset tarballs. | Store gzipped JSONL files under `data/raw/redpajama/*.jsonl.gz`. |
| Code (Stack/Python mix) | Mostly MIT/Apache | Pull from `bigcode/starcoderdata` shards or permissively licensed repos. | Preserve LICENSE metadata per shard (`data/raw/code/LICENSES.md`). |

All raw pulls should include a short README describing the source URL, date retrieved, and any filters applied. Update `docs/data_pipeline.md` whenever the mix changes so downstream users know which corpora are safe to redistribute.

## 1. Train tokenizer (multi-corpus manifest)

```bash
uv run python scripts/data/train_tokenizer.py \
  --manifest configs/data/refinedweb_mixture.yaml \
  --vocab-size 32000 \
  --output-dir artifacts/tokenizer/refinedweb_mix \
  --log-file data/mixtures/refinedweb_mix_tokenizer.json
```

The manifest pulls small samples from FineWeb (RefinedWeb proxy), Wikimedia/Wikipedia, AllenAI C4, SlimPajama, and codeparrot code datasets. Outputs live in `artifacts/tokenizer/refinedweb_mix/`.

### Tokenizer checksum
Record the checksum of every published tokenizer so collaborators can verify integrity before launching runs.

```bash
uv run python scripts/data/check_tokenizer.py \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --expected-sha256 f8871517ca968839bf6b9595a6e7891e6b8c6a70fd4df788696bce35be62d6c2 \
  --metadata-json artifacts/tokenizer/refinedweb_mix/checksum.json
```

The command prints the SHA-256 digest and writes a JSON record (optional). Keep the expected hash in this doc so CI/scripts can assert integrity. Update the hash whenever the tokenizer is retrained.

## 2. Shard mixture components

```bash
uv run python scripts/data/process_mixture.py \
  configs/data/refinedweb_mixture_filtered.yaml \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --log-file data/mixtures/refinedweb_mix_filtered_shards.json
```

This iterates over each dataset entry (either streamed from HF or the filtered local files), tokenizes at sequence length 2048, and writes NumPy shards to `data/shards/<dataset>`. Stats (records, sequences, shards, total tokens) are recorded in `data/mixtures/refinedweb_mix_shards_full.json`.

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
- Shard stats (pilot stream): `data/mixtures/refinedweb_mix_shards.json`
- Shard stats (filtered sample run): `data/mixtures/refinedweb_mix_filtered_shards.json`
- Shard stats (full filtered run, seq_len=2048): `data/mixtures/refinedweb_mix_shards_full.json`
- Latest corpus verification log: `logs/data_inventory_2025-11-10.md` (matches `data/mixtures/refinedweb_mix_full_shards.json` with `verified_at_utc` timestamp).
- Tokenizer model: `artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model`
- Continual-learning sample segments: `configs/data/continual_segments_sample.yaml`

## 6. Next steps
- Integrate the full shards into the training configs (see `configs/hope/mid.yaml`, `configs/hope/target.yaml`).
- Automate periodic re-generation (e.g., weekly) if new data arrives.
- Version mixture manifests and stats under `configs/data/` as recipes evolve.
