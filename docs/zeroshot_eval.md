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
