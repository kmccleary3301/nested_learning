# Continual-Learning Evaluation Guide

Use `scripts/eval/continual.py` to quantify forgetting across streaming segments. Supply:

- `--config`: Hydra config for the HOPE model.
- `--checkpoints`: ordered list of checkpoint paths (chronological training steps).
- `--segments-yaml`: YAML describing segment names + shard directories (see `configs/data/continual_segments_sample.yaml`).
- `--batch-size`, `--max-batches`: evaluation throughput controls (0 = entire shard).

Example:
```bash
uv run python scripts/eval/continual.py \
  --config configs/hope/mid.yaml \
  --checkpoints checkpoints/mid/step_000050.pt checkpoints/mid/step_000100.pt \
  --segments-yaml configs/data/continual_segments_sample.yaml \
  --batch-size 4 --max-batches 20 \
  --memorize --memorize-steps 2 \
  --memorize-paths titan,cms_fast \
  --memorize-surprise-threshold 0.02 \
  --output eval/continual_mid.json
```

With memorization enabled the output includes baseline vs. memorize cross-entropy, Titan/CMS update stats per segment, the active memory paths, and the surprise threshold used. Adjust `--memorize-paths` (comma-separated) to restrict which levels update (e.g., `titan` only, or `titan,cms_fast`) and `--memorize-surprise-threshold` to replicate the paper’s surprise gating.

To visualize forgetting curves:

```bash
uv run python scripts/eval/plot_forgetting.py \
  --continual-json eval/continual_mid.json \
  --segment refinedweb_2018 \
  --output reports/plots/continual_mid_refinedweb.png
```

The plot overlays baseline vs. memorize loss across checkpoints for the chosen segment. For full-scale runs, replace the sample YAML with the production segment list (e.g., chronological Wikipedia shards, MAWI sequences, etc.) and archive both the JSON and plot in your checkpoint report.
