# Stage 2 Smoke Artifact Summary

## Hardware
- 2× NVIDIA RTX 6000 Ada (49 GB VRAM each)
- PyTorch 2.9.0 (LTS), CUDA 12.4
- Python 3.12 via `uv`

## Data Prep
```bash
uv run bash scripts/data/run_sample.sh
# full pipeline (tmux recommended)
RW_LIMIT=20000 WIKI_LIMIT=10000 C4_LIMIT=8000 RPJ_LIMIT=8000 CODE_LIMIT=8000 \
  tmux new -s data_full 'uv run bash scripts/data/run_full.sh'
```

Key artifacts:
- Tokenizer: `artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model`
- Sample shards: `data/shards/*_filtered`
- Full shards: `data/shards/*_full`
- Stats: `data/mixtures/refinedweb_mix_filtered_shards.json`, `data/mixtures/refinedweb_mix_full_shards.json`

## Dual-GPU Smoke Run
```bash
uv run torchrun --nproc_per_node=2 train_dist.py --config-name mid_stage2_smoke
```
- Checkpoint: `artifacts/checkpoints/mid_stage2_smoke/step_000060.pt`
- Log: `logs/mid_stage2_smoke.json`

### Evaluations
```bash
uv run python scripts/eval/zeroshot.py --config configs/mid_stage2_smoke.yaml \
  --checkpoint artifacts/checkpoints/mid_stage2_smoke/step_000060.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --tasks piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,siqa \
  --max-samples 64 --device cuda:1

uv run python scripts/eval/niah.py --config configs/mid_stage2_smoke.yaml \
  --checkpoint artifacts/checkpoints/mid_stage2_smoke/step_000060.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --context-lengths 2048 --samples-per-length 5 --device cuda:1

uv run python scripts/eval/continual.py --config configs/mid_stage2_smoke.yaml \
  --checkpoints artifacts/checkpoints/mid_stage2_smoke/step_000060.pt \
  --segments-yaml configs/data/continual_segments_sample.yaml \
  --batch-size 4 --max-batches 5 --device cuda:1
```
- Zero-shot metrics: `eval/zeroshot_mid_stage2_smoke.json`
- NIAH: `eval/niah_mid_stage2_smoke.json`
- Continual: `eval/continual_mid_stage2_smoke.json`

## Mid-Scale Reference Run
```bash
tmux new -s mid_stage2_run "uv run torchrun --nproc_per_node=2 train_dist.py --config-name mid_stage2"
```
- Checkpoint: `artifacts/checkpoints/mid_stage2/step_000100.pt`
- Log: `logs/mid_stage2.json`
- Eval summaries: `eval/zeroshot_mid_stage2.json`, `eval/niah_mid_stage2.json`, `eval/continual_mid_stage2.json`

## Teach-scale Sweep (single GPU, batch=4)
Reference runs for teach_scale ∈ {0.05, 0.10, 0.20}:
```bash
uv run python train.py --config-name mid_stage2 \
  model.teach_scale=0.10 model.teach_clip=5.0 \
  data.batch_size=4 train.steps=40 train.device=cuda:1 \
  logging.path=logs/mid_stage2_single_ts10.json \
  train.checkpoint.dir=artifacts/checkpoints/mid_stage2_single_ts10
```
Logs and checkpoints are stored under `logs/mid_stage2_single_ts*.json`, `artifacts/checkpoints/mid_stage2_single_ts*/`.

## Extended Single-GPU Run (teach_scale=0.10)
```bash
tmux new -s mid_stage2_ts10_single "uv run python train.py --config-name mid_stage2 \
  model.teach_scale=0.10 model.teach_clip=4.0 \
  model.teach_schedule.warmup_steps=60 \
  model.teach_schedule.decay_start=140 \
  model.teach_schedule.decay_duration=80 \
  data.batch_size=4 optim.lr=1e-5 train.device=cuda:1 \
  train.steps=220 train.log_interval=20 \
  logging.path=logs/mid_stage2_ts10_single220_schedD.json \
  train.checkpoint.dir=artifacts/checkpoints/mid_stage2_ts10_single220_schedD"
```
- Checkpoint: `artifacts/checkpoints/mid_stage2_ts10_single220_schedD/step_000220.pt`
- Log: `logs/mid_stage2_ts10_single220_schedD.json`
- Evaluations: `eval/zeroshot_mid_stage2_ts10_single220_schedD.json`, `eval/niah_mid_stage2_ts10_single220_schedD.json`, `eval/continual_mid_stage2_ts10_single220_schedD.json`

## TITAN Baseline (single GPU)
```bash
uv run python train.py --config-name mid_titan_baseline
```
- Checkpoint: `artifacts/checkpoints/mid_titan_baseline/step_000200.pt`
- Log: `logs/mid_titan_baseline.json`
- Evaluations: `eval/zeroshot_mid_titan_baseline.json`, `eval/niah_mid_titan_baseline.json`, `eval/continual_mid_titan_baseline.json`

This document should accompany the release tag so others can reproduce the exact smoke workflow in a few commands.
