# Phase 2 – HOPE-Attention vs Transformer Baseline

Phase 2 is “implementation-complete” when we can compare the **paper-defined HOPE-Attention** variant
(`Attention → CMS`) against a **standard Transformer** baseline (`Attention → MLP`) using the same
tokenizer, context lengths, and evaluation harness.

This does **not** require paper-scale training; it’s intended for correctness/ergonomics and
CPU-friendly smoke checks.

## 0) Prerequisites

- A SentencePiece tokenizer at `artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model`.
  - If missing, run `uv run bash scripts/data/run_sample.sh` (see `docs/guide.md`).

## 1) Smoke checkpoints (CPU)

Train two tiny smoke checkpoints from the same base config:

```bash
# HOPE-Attention smoke (paper-defined variant)
uv run python train.py --config-name pilot_smoke \
  model.block_variant=hope_attention \
  model.qk_l2_norm=true model.local_conv_window=4 \
  train.checkpoint.dir=artifacts/checkpoints/pilot_smoke_attention \
  logging.path=logs/pilot_smoke_attention.json

# Transformer baseline smoke
uv run python train.py --config-name pilot_smoke \
  model.block_variant=transformer \
  model.qk_l2_norm=true model.local_conv_window=4 \
  train.checkpoint.dir=artifacts/checkpoints/pilot_smoke_transformer \
  logging.path=logs/pilot_smoke_transformer.json
```

## 2) Long-context comparison (CPU)

Use the comparison runner (writes a single JSON with both results):

```bash
uv run python scripts/eval/compare_variants.py \
  --a-config configs/pilot_smoke.yaml \
  --a-checkpoint artifacts/checkpoints/pilot_smoke_attention/step_000010.pt \
  --b-config configs/pilot_smoke.yaml \
  --b-checkpoint artifacts/checkpoints/pilot_smoke_transformer/step_000010.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --device cpu \
  --smoke \
  --output eval/phase2_compare_smoke.json
```

For larger GPU-backed pilots, use the dedicated Hydra configs:
- `configs/hope/pilot_attention.yaml`
- `configs/hope/pilot_transformer.yaml`

and rerun the comparison script on the resulting checkpoints.

## 3) Adaptation sanity check (no training)

This repo also includes a deterministic unit-level smoke that demonstrates **in-context adaptation**
exists for `hope_attention` (via CMS fast-state updates) and is absent for `transformer`:

```bash
uv run pytest -q tests/test_phase2_memorization_delta.py
```

For a standalone JSON output (no tokenizer/checkpoints required):

```bash
uv run python scripts/eval/phase2_memorization_delta_smoke.py --device cpu
```
