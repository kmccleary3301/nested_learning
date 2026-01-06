# Zero-shot Evaluation Guide

The script `scripts/eval/zeroshot.py` evaluates HOPE checkpoints on 
common reasoning benchmarks. Tasks currently supported:

- `piqa`
- `hellaswag`
- `winogrande`
- `arc_easy`
- `arc_challenge`
- `boolq`
- `siqa`
- `commonsenseqa`
- `openbookqa`
- Synthetic LongBench-style passkey (`scripts/eval/passkey.py`)
- PG-19 perplexity (`scripts/eval/pg19_perplexity.py`)

## Usage

```bash
uv run python scripts/eval/zeroshot.py \
  --config configs/hope/mid.yaml \
  --checkpoint checkpoints/mid/checkpoint_best.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --tasks all \
  --max-samples 500 \
  --output eval/zeroshot_mid.json
```

Set `--tasks` to a comma-separated list (e.g., `piqa,hellaswag`) or `all`.
Use `--list-tasks` to print available options.

Each task logs accuracy and sample count into the JSON file. Adjust
`--max-samples` (0 = evaluate entire validation set) based on runtime.

For reproducibility, record the checkpoint SHA, tokenizer version, 
and command invocation alongside the JSON results.

## Long-context diagnostics

### Needle-in-a-Haystack (NIAH)

Probe retrieval across increasing prompt lengths:

```bash
uv run python scripts/eval/niah.py \
  --config configs/hope/pilot.yaml \
  --checkpoint artifacts/checkpoints/pilot/step_230000.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --context-lengths 2048 --context-lengths 4096 --context-lengths 8192 \
  --samples-per-length 50 \
  --memorize --memorize-steps 2 \
  --memorize-paths titan,cms_fast \
  --memorize-surprise-threshold 0.02 \
  --output eval/niah_pilot.json
```

### RULER-ish NIAH Suite

Run multiple synthetic retrieval variants (single-needle, multi-needle, KV retrieval, and positioned needles):

```bash
uv run python scripts/eval/niah_suite.py \
  --config configs/hope/pilot.yaml \
  --checkpoint artifacts/checkpoints/pilot/step_230000.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --context-tokens 2048 --context-tokens 4096 \
  --samples-per-length 25 \
  --memorize --memorize-steps 2 \
  --memorize-paths titan,cms_fast \
  --memorize-surprise-threshold 0.02 \
  --output eval/niah_suite_pilot.json
```

Plot the suite as accuracy vs context length:

```bash
uv run python scripts/eval/plot_niah_suite.py \
  --niah-suite-json eval/niah_suite_pilot.json \
  --output reports/plots/niah_suite_pilot.png
```

### Passkey Retrieval
Generate synthetic passkey prompts to stress memorization at test time:

```bash
uv run python scripts/eval/passkey.py \
  --config configs/hope/pilot.yaml \
  --checkpoint artifacts/checkpoints/pilot/step_230000.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --samples 64 --filler-sentences 256 \
  --memorize --memorize-steps 2 \
  --memorize-paths titan,cms_fast \
  --memorize-surprise-threshold 0.02 \
  --output eval/passkey_pilot.json
```

The JSON reports baseline vs. memorize accuracy, Titan/CMS update stats, the active memory paths, and the surprise threshold. Use `--memorize-paths` to restrict updates to `titan`, `cms_fast`, or any comma-separated combination, and `--memorize-surprise-threshold` to match the paper’s surprise-gated updates.

Note: memorization runs against a per-example fast state by default (so checkpoint weights are not mutated). If you call `memorize_sequence()` programmatically with `MemorizeConfig.use_fast_state=true`, pass a `fast_state = model.init_fast_state()` object.

### PG-19 Perplexity

```bash
uv run python scripts/eval/pg19_perplexity.py \
  --config configs/hope/pilot.yaml \
  --checkpoint artifacts/checkpoints/pilot/step_230000.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --max-samples 64 \
  --output eval/pg19_pilot.json
```

This computes long-form perplexity (baseline vs. memorize) and records total tokens processed.
