# Nested Learning Reproduction

![Python](https://img.shields.io/badge/python-3.12+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.9.0-red)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Status](https://img.shields.io/badge/tests-smoke--ready-lightgrey)

High-fidelity reproduction of Google's Nested Learning (HOPE) architecture, matching the quality bar set by lucidrains' TITAN reference while remaining fully open-source and `uv` managed.

## Quickstart
```bash
uv python install 3.12
uv sync --all-extras
uv run bash scripts/data/run_sample.sh
uv run bash scripts/run_smoke.sh pilot  # CPU-friendly HOPE block smoke test
uv run bash scripts/run_e2e_smoke.sh    # sync + sample data + smoke train + zeroshot eval
uv run python scripts/eval/zeroshot.py \
  --config configs/hope/pilot.yaml \
  --checkpoint artifacts/examples/pilot_dummy.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --tasks piqa --max-samples 32 --device cpu
```

## Requirements
- Python 3.12+
- `uv` package manager (https://github.com/astral-sh/uv)
- PyTorch 2.9.0 LTS + CUDA-capable GPUs for accelerated runs (CPU works for smoke tests)

## Setup
```bash
uv python install 3.12
uv sync --all-extras
```

Developer checks:
- `uv run ruff check .`
- `uv run mypy src`
- `uv run pytest`

## Data Pipeline
1. **Tokenizer training**
   ```bash
   uv run python scripts/data/train_tokenizer.py \
     --manifest configs/data/refinedweb_mixture.yaml \
     --vocab-size 32000 \
     --output-dir artifacts/tokenizer/refinedweb_mix \
     --log-file data/mixtures/refinedweb_mix_tokenizer.json
   ```
2. **Corpus filtering + sharding**
   ```bash
   uv run python scripts/data/process_mixture.py \
     configs/data/refinedweb_mixture_filtered.yaml \
     --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
     --log-file data/mixtures/refinedweb_mix_filtered_shards.json
   ```
3. **Sample pipeline** (downloads/licensed datasets, filters, shards, records stats)
   ```bash
   uv run bash scripts/data/run_sample.sh
   ```
4. **Full pipeline** (set env vars like `RW_LIMIT`, `WIKI_LIMIT`, etc. to scale ingestion)
   ```bash
   uv run bash scripts/data/run_full.sh  # default ~50k docs per corpus; increase limits as needed
   ```

## Training
- Single GPU / CPU:
  ```bash
  uv run python train.py --config-name pilot_smoke
  ```
- DDP (torchrun):
  ```bash
  torchrun --nproc_per_node=2 train_dist.py --config-name mid
  ```
- CPU-only DDP smoke (verifies `gloo` backend and deterministic seeding):
  ```bash
  uv run bash scripts/run_cpu_ddp_smoke.sh
  ```
- FSDP:
  ```bash
  torchrun --nproc_per_node=2 train_fsdp.py --config-name mid
  ```
- DeepSpeed (requires `deepspeed` installed separately):
  ```bash
  deepspeed --num_gpus=2 train_deepspeed.py --config-name target \
    deepspeed.config=configs/deepspeed/zero3.json
  ```

### Pilot (3 B tokens) workflow
1. Ensure TMUX session:
   ```bash
   tmux new -s pilot_train
   ```
2. Launch the long run on `cuda:1` (≈52 h wall clock):
   ```bash
   set -a && source git.env && set +a
   export UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy
   uv run python train.py --config-name pilot \
     logging.enabled=true logging.backend=wandb \
     logging.project=nested-learning logging.run_name=pilot-main-$(date +%Y%m%d%H%M%S) \
     train.device=cuda:1
   ```
3. Checkpoints appear in `artifacts/checkpoints/pilot/step_*.pt` every 1 000 steps; the accompanying W&B run captures full telemetry.
4. Copy the final checkpoint, config, logs, and eval JSON/CSV into `artifacts/pilot_release/` for distribution.

## Logging
Set `logging.enabled=true` in Hydra configs (or override via CLI) to send metrics to W&B (default). For local JSON logs, use `logging.backend=json logging.path=logs/run.json`. Sample outputs reside in `logs/` and `artifacts/examples/`.

## Evaluation
- Zero-shot:
  ```bash
  uv run python scripts/eval/zeroshot.py \
  --config configs/hope/mid.yaml \
  --checkpoint checkpoints/mid/step_000100.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --tasks all --max-samples 200 --device cuda:0
  ```
  Use `uv run python scripts/eval/zeroshot.py --list-tasks` to display the full benchmark roster (PIQA, HellaSwag, WinoGrande, ARC-E/C, BoolQ, SIQA, CommonsenseQA, OpenBookQA). See `docs/zeroshot_eval.md` for details.
- Needle-in-a-Haystack:
  ```bash
  uv run python scripts/eval/niah.py \
    --config configs/hope/mid.yaml \
    --checkpoint checkpoints/mid/step_000100.pt \
    --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
    --context-lengths 2048 4096 8192 --samples-per-length 20
  ```
- Continual-learning forgetting:
  ```bash
  uv run python scripts/eval/continual.py \
    --config configs/hope/mid.yaml \
    --checkpoints checkpoints/mid/step_000050.pt checkpoints/mid/step_000100.pt \
    --segments-yaml configs/data/continual_segments_sample.yaml \
    --batch-size 4 --max-batches 10
  ```

Evaluation summaries are written to `eval/` alongside per-task JSON metrics.

### Test-time memorization toggles
Every evaluator supports TITAN-style memorization so you can reproduce test-time adaptation:
```bash
uv run python scripts/eval/zeroshot.py \
  ... \
  --memorize \
  --memorize-steps 2 \
  --memorize-use-correct-answer \
  --memorize-no-reset  # optional: retain updates across samples
```
- `--memorize` turns on the learner with one LMS step per example by default.
- `--memorize-steps` controls the number of adaptation passes per prompt.
- `--memorize-use-correct-answer` injects ground-truth text during memorization for ablations.
- `--memorize-no-reset` carries memories across samples; omit it to reset every question.

Memorization metrics (baseline vs adaptive) are emitted alongside task accuracy for easy comparisons.

## Performance & optimizer options
- **Mixed precision:** enable bf16 autocast via `train.mixed_precision.enabled=true train.mixed_precision.dtype=bf16`.
- **`torch.compile`:** accelerate attention/core loops by toggling `train.compile.enable=true train.compile.mode=max-autotune`.
- **Fused AdamW:** default `optim.type=adamw optim.fused=auto` enables fused CUDA kernels when available.
- **Muon optimizer:** route matrix weights through `torch.optim.Muon` while keeping embeddings/biases on AdamW:
  ```bash
  uv run python train.py --config-name pilot \
    optim.type=muon optim.lr=2.5e-4 optim.weight_decay=0.01 \
    train.mixed_precision.enabled=true train.compile.enable=true
  ```

All Hydra knobs can be overridden from the CLI or composed via config groups (`configs/hope/*.yaml`). Use these flags in tandem with `scripts/run_e2e_smoke.sh` (automation) or `scripts/run_cpu_ddp_smoke.sh` (CPU-only determinism check) to validate releases quickly.

## Documentation & References
- `docs/guide.md` – full onboarding (setup → data → training → eval).
- `docs/release_plan.md` – release readiness checklist.
- `docs/data_pipeline.md` – large-scale sharding/tokenizer workflow.
- `docs/scaling_guidance.md` – roadmap for expanding data + compute footprints.
- `docs/stage1_plan.md`, `docs/stage2_plan.md` – architecture + experiment roadmaps.
- `docs/stage2_progress.md` – latest dual-GPU training/eval status and commands.
- `docs/experiments_report.md` – draft paper covering completed experiments.
- `docs/stability_journal.md` – chronological notes on NaN fixes & teach-scale tuning.
- `docs/future_directions.md` – prioritized roadmap after the initial release.
- `reports/stage2_smoke.md` – exact commands/artifacts for the release-ready smoke workflow.
- `google_papers/` – PDFs/markdown of Nested Learning & TITAN papers.
- `CHANGELOG.md` – user-facing changes per release.

## Contributing
1. Run formatting/tests (`uv run ruff check .`, `uv run pytest`).
2. Document new configs or scripts in `docs/guide.md` and update `CHANGELOG.md`.
3. Open a PR referencing the relevant NL/TITAN spec sections or planner transcript snippets.
