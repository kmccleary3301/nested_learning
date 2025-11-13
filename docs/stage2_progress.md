# Stage 2 Progress Report (Nov 9, 2025)

This note captures the current state of Stage 2 (results reproduction) so collaborators can pick up the dual-GPU workflow immediately.

## 1. Data Pipeline
- `scripts/data/run_full.sh` now orchestrates filtering + sharding for the full RefinedWeb mixture (see `configs/data/refinedweb_mixture_full.yaml`).
- Latest tmux run (`data_full`) completed successfully under the limits: `RW_LIMIT=20000`, `WIKI_LIMIT=10000`, `C4_LIMIT=8000`, `RPJ_LIMIT=8000`, `CODE_LIMIT=8000`.
- Stats logged to `data/mixtures/refinedweb_mix_full_shards.json` (≈20k RefinedWeb docs → 10 M tokens; similar counts for other corpora). Filtered text artifacts live in `data/filtered/*_full.txt`, shards under `data/shards/*_full/`.
- Mid-scale configs accept overrides such as `NL_SHARD_DIR_REFINEDWEB`, etc., so you can fall back to the filtered sample shards when running on machines with limited storage.
- For quick smoke validation, `scripts/data/run_sample.sh` remains the default command referenced in the README/guide.

## 2. Training Runs
### 2.1 Mid-scale DDP (2× RTX 6000 Ada)
- Config: `configs/mid_stage2.yaml` (18 layers, dim = 768, heads = 12) now points at the `_full` shards by default. Teach-signal damping is handled via `model.teach_scale` (currently `0.05`) plus clipping (`model.teach_clip=5.0`) so we can keep self-modifiers active without divergent gradients.
- Command (launched via tmux `mid_stage2_run` to avoid CLI timeouts):
  ```bash
  tmux new -s mid_stage2_run "cd /mnt/drive_4/research/nested_learning && \
    uv run torchrun --nproc_per_node=2 train_dist.py --config-name mid_stage2"
  ```
- Outcome: 100 steps on mixture data with stable loss (final logged ppl ≈3.0e3). Checkpoint `artifacts/checkpoints/mid_stage2/step_000100.pt` (≈7.0 GB) plus metrics `logs/mid_stage2.json`. Evaluation summaries:
  - Zero-shot (`eval/zeroshot_mid_stage2.json`): PIQA 0.50, Winogrande 0.625 (16 samples), HellaSwag near-random (expected at this stage) with teach-signal clipping enabled.
  - NIAH (`eval/niah_mid_stage2.json`): accuracy 0.33 at 2k/4k contexts.
  - Continual (`eval/continual_mid_stage2.json`): losses ≈8.0 on RefinedWeb/Wikipedia/RedPajama; no measured forgetting with single checkpoint.
- Next steps: re-enable self-mod (teach_scale > 0) once gradient stabilization (clipping, optimizer gating) is implemented.

- **Teach-scale sweep (DDP, batch size 4, 80 steps)**
| teach_scale | clip | batch | final log path | checkpoint | PIQA | Winogrande | Notes |
|-------------|------|-------|----------------|------------|------|------------|-------|
| 0.05 (baseline) | 5.0 | 8 | `logs/mid_stage2.json` | `artifacts/checkpoints/mid_stage2/step_000100.pt` | 0.50 | 0.625 | Stable through 100 steps |
| 0.10 (single GPU, lr=1.5e-5 + grad clip) | 5.0→0 | **4** | `logs/mid_stage2_ts10_single220_schedD.json` | `artifacts/checkpoints/mid_stage2_ts10_single220_schedD/step_000220.pt` | 0.469 | 0.594 | 220-step run on `cuda:1`; warmup/decay schedule keeps training finite |
| 0.10 (DDP) | 5.0 | **4** | `logs/mid_stage2_ts10.json` | `artifacts/checkpoints/mid_stage2_ts10/step_000080.pt` | 0.469 | 0.594 | Needed per-GPU batch drop to avoid OOM; diverged past step 80 |
| 0.20 | 8.0 | **4** | `logs/mid_stage2_ts20.json` | `artifacts/checkpoints/mid_stage2_ts20/step_000080.pt` | 0.469 | 0.594 | Similar behavior; NIAH/continual still unstable at this depth |

  NIAH accuracy for these runs remains near chance (latest `eval/niah_mid_stage2_ts10_single220_schedD.json` reports 0 at 2k/4k tokens). Continual metrics are now finite for the 220-step checkpoint, but still noisy because the run covers <1k tokens of data.

#### TITAN-only baseline (single GPU, batch=4)
- Config: `configs/mid_titan_baseline.yaml` (`type: titan`, same teach schedule and optimizer as the HOPE run).
- Command:
  ```bash
  uv run python train.py --config-name mid_titan_baseline
  ```
- Checkpoint: `artifacts/checkpoints/mid_titan_baseline/step_000200.pt`, Log: `logs/mid_titan_baseline.json`.
- Evaluations:
  - `eval/zeroshot_mid_titan_baseline.json` (PIQA 0.469, Winogrande 0.594 on 128 samples).
  - `eval/niah_mid_titan_baseline.json` (accuracy 0 at 2k/4k contexts).
  - `eval/continual_mid_titan_baseline.json` (finite losses similar to HOPE).

**Comparison snapshot (200–220 steps, same data/batch/teach schedule)**
| Model | Steps | PIQA | Winogrande | Notes |
|-------|-------|------|------------|-------|
| HOPE (teach_scale 0.10) | 220 | 0.469 | 0.594 | `eval/zeroshot_mid_stage2_ts10_single220_schedD.json` |
| TITAN baseline | 200 | 0.469 | 0.594 | `eval/zeroshot_mid_titan_baseline.json` |

At this early stage both models perform similarly on the short zero-shot probe, and neither shows meaningful NIAH gains. Longer runs will be needed to observe the paper’s reported HOPE vs. TITAN differences.

### 2.2 Dual-GPU Smoke (configs/mid_stage2_smoke.yaml)
- Smaller 12-layer, dim = 512 model for rapid integration tests. Uses `teach_scale=0.2` with `teach_clip=2.0` to keep self-mod active while remaining stable.
- Run command identical to above with `--config-name mid_stage2_smoke`.
- Outputs:
  - Checkpoint `artifacts/checkpoints/mid_stage2_smoke/step_000060.pt`
  - Log `logs/mid_stage2_smoke.json`
  - Evaluations: `eval/zeroshot_mid_stage2_smoke.json`, `eval/niah_mid_stage2_smoke.json`, `eval/continual_mid_stage2_smoke.json`
- These artifacts prove the distributed training/eval wiring and should accompany PRs before moving to the heavier config.

### 2.3 Pilot-scale run (3 B tokens, single GPU)
- Config: `configs/pilot.yaml` (dim 512, 12 layers, teach_scale 0.10, CMS fast/mid/slow/ultra). Batch 6 × seq 2048 → ≈3.03 B tokens at 246 667 steps.
- **Short-run snapshot:** A 9 000-step job (W&B `pilot-short-20251111184315`) produced checkpoints every 500 steps and a release bundle at `artifacts/pilot_release/` (includes PIQA/NIAH/continual JSONs). Loss dropped from 93 → 18 by step 600; PIQA accuracy at 128 samples is 0.5625.
- **Full run plan:** Resume the long tmux job once TITAN baseline finishes:
  ```bash
  tmux new -s pilot_full "cd /mnt/drive_4/research/nested_learning && \
    set -a && source git.env && set +a && \
    export UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy && \
    uv run python train.py --config-name pilot \
      logging.enabled=true logging.backend=wandb \
      +logging.project=nested-learning +logging.run_name=pilot-main-$(date +%Y%m%d%H%M%S) \
      train.device=cuda:1 train.steps=246667 train.checkpoint.save_interval=1000"
  ```
- Eval automation: `scripts/eval/run_pilot_suite.sh` now stitches zero-shot/NIAH/continual runs so we can refresh metrics whenever a new checkpoint is packaged.
- Next: keep `scripts/package_pilot_release.sh` in the tmux workflow (every 25k steps) and mirror the workflow for the TITAN baseline to establish direct comparisons.

## 3. Recommended Workflow for Contributors
1. **Environment** – `uv sync --all-extras && uv run bash scripts/data/run_sample.sh`
2. **Distributed smoke** – `uv run torchrun --nproc_per_node=2 train_dist.py --config-name mid_stage2_smoke`
3. **Eval suite** – Run the three scripts above pointing to `mid_stage2_smoke` checkpoint.
4. **Scaling** – Attach tmux sessions for long jobs:
   - Data: `tmux new -s data_full '... run_full.sh'`
   - Mid-scale: `tmux new -s mid_stage2_run '... torchrun ... mid_stage2'`
5. **Artifacts** – Drop new checkpoints/logs in `artifacts/checkpoints/mid_stage2*`, `logs/`, and store eval JSON under `eval/` with descriptive names. For long pilot runs, copy the resulting checkpoint + config + eval outputs into `artifacts/pilot_release/` so users can download a single bundle.

Documenting these steps here keeps everyone aligned while we chase full Stage 2 parity.

## 4. Release Checklist (current)
*(Assumes only `cuda:1` is available—adjust `train.device` overrides accordingly.)*
1. `uv sync --all-extras`
2. `uv run bash scripts/data/run_sample.sh` *(for quick validation; swap in `run_full.sh` when storage allows).*
3. `uv run bash scripts/run_smoke.sh pilot`
4. `uv run bash scripts/run_cpu_ddp_smoke.sh` *(ensures gloo backend determinism for contributors without GPUs).*
5. `uv run bash scripts/run_e2e_smoke.sh` *(sync → sample data → pilot smoke → PIQA eval).*
6. `uv run torchrun --nproc_per_node=2 train_dist.py --config-name mid_stage2_smoke`
7. `uv run python scripts/eval/zeroshot.py --config configs/mid_stage2_smoke.yaml --checkpoint artifacts/checkpoints/mid_stage2_smoke/step_000060.pt --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model --tasks piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,siqa --max-samples 64 --device cuda:1 --memorize --memorize-steps 2 --memorize-use-correct-answer`
8. `uv run python scripts/eval/niah.py --config configs/mid_stage2_smoke.yaml --checkpoint artifacts/checkpoints/mid_stage2_smoke/step_000060.pt --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model --context-lengths 2048 --context-lengths 4096 --context-lengths 8192 --samples-per-length 5 --device cuda:1`
9. `uv run python scripts/eval/continual.py --config configs/mid_stage2_smoke.yaml --checkpoints artifacts/checkpoints/mid_stage2_smoke/step_000060.pt --segments-yaml configs/data/continual_segments_sample.yaml --batch-size 4 --max-batches 5 --device cuda:1 --memorize --memorize-steps 1`
10. (Optional) Run `tmux new -s mid_stage2_run '... mid_stage2'` to produce the 100-step mid checkpoint + evals cited above, then start the long pilot run via `tmux new -s pilot_train ...`.
- **Teach-scale sweep (single GPU, batch 4, 40 steps)**  
  | teach_scale | clip | final loss | checkpoint | log |
  |-------------|------|------------|------------|-----|
  | 0.05 | 5.0 | 9.81 | `artifacts/checkpoints/mid_stage2_single_ts05/step_000040.pt` | `logs/mid_stage2_single_ts05.json` |
  | 0.10 | 5.0 | 9.77 | `artifacts/checkpoints/mid_stage2_single_ts10/step_000040.pt` | `logs/mid_stage2_single_ts10.json` |
  | 0.20 | 8.0 | 9.76 | `artifacts/checkpoints/mid_stage2_single_ts20/step_000040.pt` | `logs/mid_stage2_single_ts20.json` |
  Even at 0.2 the run stays stable, suggesting we can raise teach_scale beyond 0.05 once longer DDP runs are secured.
