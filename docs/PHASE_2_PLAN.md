# Phase 2 Plan – Execution & Results Packaging

## 1. Training Runs
1. **Pilot (160M / 3B tokens)**
   - Objective: confirm stability, log teach-scale findings, generate base checkpoints for eval harnesses.
   - Actions: run `configs/hope/pilot.yaml` with the full shard mixture; log to W&B and artifacts/.
2. **Mid-scale (760M / 30B tokens)**
   - Objective: produce the headline zero-shot/NIAH results.
   - Actions: run `configs/hope/mid.yaml` (FSDP or DeepSpeed), capture checkpoints every ~50k steps.
3. **Target (1.3B / 100B tokens)**
   - Objective: long-context + continual-learning showcase.
   - Actions: integrate 8k context curriculum, run with DeepSpeed ZeRO-3, checkpoint frequently.

## 2. Evaluation Campaign
1. **Zero-shot pack** – Use `scripts/eval/zeroshot.py --tasks all` on pilot/mid/target checkpoints; store JSON in `eval/zeroshot_*.json` and plot aggregated table in `docs/experiments_report.md`.
2. **NIAH curves** – Run `scripts/eval/niah.py` (2048→512k) for each major checkpoint and plot accuracy vs. context length.
3. **Continual-learning** – Run `scripts/eval/continual.py` across chronological segments; generate forgetting plots and correlate with level clocks.

## 3. Baseline Comparisons
- Reproduce lighter TITAN/Transformer baselines (reuse refs or simple adaptations) to evaluate on the same data/eval tasks.
- Log results alongside HOPE for direct comparison in `reports/ablations.md` and W&B dashboards.

## 4. Ablations
1. Self-modifier on/off.
2. CMS depth variations (1 vs. 3 vs. 5 levels).
3. Deep optimizer variants per level.
4. Attention swap (full vs. sliding-window/DeltaNet).
Record commands + metrics in `reports/ablations.md`.

## 5. Documentation & Release
1. Update `docs/experiments_report.md` with tables/plots.
2. Record stability tricks and teach-scale notes in `docs/stability_journal.md`.
3. Prepare a blog/paper draft summarizing architecture, training setup, and results.
4. Tag a release (`v0.2-stage2-prep`) with checkpoints, configs, eval JSONs.

## 6. Outreach & Community
- Share follow-up results posts (link to W&B dashboards, zero-shot tables, long-context plots).
- Invite collaborators for continual-learning and scaling experiments via README/Issues/Discussions.

## 7. Tracking
- Keep `TODO.md` updated per milestone.
- Use W&B projects for each run (pilot/mid/target) and link them in `docs/stage2_progress.md`.
