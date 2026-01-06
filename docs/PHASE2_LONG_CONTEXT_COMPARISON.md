# Phase 2 – HOPE-Attention vs Transformer (Long-Context Sanity)

This repo includes a lightweight Phase‑2 sanity check that compares **HOPE-Attention** (Attention → CMS) against a **baseline Transformer** on synthetic long‑context retrieval prompts.

The goal is not to claim paper‑level results (that requires large‑scale training), but to provide a **reproducible, implementation-level signal** that:

- HOPE-Attention’s fast-state memorization path can **improve the margin/logprob** of the correct answer on long contexts.
- The baseline Transformer **cannot**, because it has no in‑context update path.

## What to run

This uses resolved, eval-friendly configs (no Hydra composition required):
- `configs/resolved/phase2_pilot_attention_eval.yaml`
- `configs/resolved/phase2_pilot_transformer_eval.yaml`

And uses the init checkpoints generated under `artifacts/checkpoints/phase2_init/` (gitignored).

Run (GPU recommended):

```bash
UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache \
uv run python scripts/eval/compare_variants.py \
  --a-config configs/resolved/phase2_pilot_attention_eval.yaml \
  --a-checkpoint artifacts/checkpoints/phase2_init/hope_attention_step000000.pt \
  --b-config configs/resolved/phase2_pilot_transformer_eval.yaml \
  --b-checkpoint artifacts/checkpoints/phase2_init/transformer_step000000.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --device cuda:1 \
  --output eval/phase2_compare_smoke_lastlayer_metrics.json \
  --seed 0 \
  --smoke \
  --memorize \
  --memorize-use-correct-answer \
  --memorize-layers last \
  --memorize-paths cms_fast
```

## What to look at

Open `eval/phase2_compare_smoke_lastlayer_metrics.json` and compare:

- **HOPE-Attention (A)**:
  - `a.passkey.mean_margin_delta` > 0
  - `a.niah.niah_256_mean_margin_delta` > 0
- **Transformer (B)**:
  - corresponding `*_mean_margin_delta` fields are exactly `0.0`

This demonstrates a concrete Phase‑2 differentiator at pilot scale: **test‑time learning updates move the model in a direction that improves long‑context answer margins**, and the baseline cannot.

