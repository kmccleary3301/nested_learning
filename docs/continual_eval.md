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
  --output eval/continual_mid.json
```

The script logs per-segment cross-entropy for each checkpoint. Plotting these curves reveals forgetting (loss increases on earlier segments) vs. stability. For full-scale runs, replace the sample YAML with the production segment list (e.g., chronological Wikipedia shards, MAWI sequences, etc.).
