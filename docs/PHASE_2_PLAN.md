# Phase 2 Plan – Execution & Results Packaging

## Immediate Remediation Tasks (from EX_PHASE_1_CRITIQUE)

Before resuming large-scale runs, we must land the following **P0 faithfulness fixes** plus high-priority engineering upgrades. Each item lists the concrete code touchpoints, validation criteria, and downstream dependencies.

### 1. Tie LM head weights + correct teach signal
- **Scope**: `src/nested_learning/model.py`, `src/nested_learning/titan/model.py`, `src/nested_learning/training.py`, unit tests under `tests/`.
- **Actions**:
  1. Tie `lm_head.weight` to `embed.weight` for HOPE + TITAN models.
  2. Update `compute_teach_signal` to:
     - Use `model.lm_head.weight.detach()` instead of embeddings.
     - Shift logits/targets to align with CE loss (`logits[:, :-1]` vs `tokens[:, 1:]`).
     - Pad the teacher signal to maintain sequence length.
  3. Add `tests/test_teach_signal.py` performing a finite-difference gradient check.
- **Acceptance**: Unit test passing; manual verification on pilot smoke run logs (teach-signal norms logged).

### 2. Implement CMS chunk accumulation (Eq. 31)
- **Scope**: `src/nested_learning/cms.py` (or equivalent), `src/nested_learning/levels.py`, new telemetry structs, tests.
- **Actions**:
  1. Add per-level ring buffers sized to `update_period`.
  2. Accumulate gradients/error proxies each step; only trigger optimizer update when buffer is full, then clear.
  3. Emit `UpdateEvent` metrics (count, L2 norm) per level.
  4. Unit test verifying exactly one update per `update_period` ticks.
- **Acceptance**: Tests pass; pilot smoke shows stepped CMS updates in logs.

### 3. Add L2-regression inner update (Eq. 27–29)
- **Scope**: `src/nested_learning/optim/deep_momentum.py`, model forward hooks to pass `x_t`, tests.
- **Actions**:
  1. Introduce `variant="nl_l2_precond"` that computes the rank-1 projector from input activations.
  2. Route the relevant activations into the optimizer context.
  3. Config flag in `configs/hope/*.yaml` to enable this variant.
  4. Toy test: optimization reduces regression objective.
- **Acceptance**: Unit test + pilot smoke run with `variant` enabled (log preconditioner statistics).

### 4. Enable test-time memorization
- **Scope**: `scripts/eval/zeroshot.py`, `scripts/eval/niah.py`, `scripts/eval/continual.py`, model eval hooks.
- **Actions**:
  1. Add flags (`--memorize`, `--memorize-steps`, `--memory-lr`, `--surprise-threshold`).
  2. Implement TITAN memory updates (and optional CMS fast level) when `memorize=True`.
  3. Add synthetic integration test ensuring memorization improves accuracy on a constructed needle task.
- **Acceptance**: Tests pass; eval scripts produce separate `*_memorize.json` outputs with metrics > baseline on synthetic task.

### 5. PyTorch performance upgrades
- **Scope**: `src/nested_learning/*.py` (attention, training loop), optim factory.
- **Actions**:
  1. Replace `nn.MultiheadAttention` with manual QKV + `torch.nn.functional.scaled_dot_product_attention`, enabling FlashAttention where supported.
  2. Wrap training step in `torch.autocast(device_type, dtype=torch.bfloat16)`; add config switch.
  3. Add `torch.compile` (guarded) to model init.
  4. Use fused AdamW (`fused=True`) for outer optimizer.
- **Acceptance**: Pilot smoke runtime improves or stays stable; fallback path works on CPU.

### 6. Muon integration
- **Scope**: `src/nested_learning/optim/factory.py`, configs.
- **Actions**:
  1. Detect availability of `torch.optim.Muon`.
  2. Split param groups: matrices → Muon, embeddings/biases/LayerNorm → AdamW.
  3. Config knob `optim.outer.type = mixed_muon_adamw`.
  4. Benchmark vs AdamW and log results.
- **Acceptance**: Pilot smoke runs succeed with Muon; documentation updated.

### 7. Seeding & backend robustness
- **Scope**: training entrypoints (`train*.py`), `nested_learning/training.py`.
- **Actions**:
  1. Add `--seed` (Hydra config) and set Python/NumPy/Torch seeds + DataLoader worker init.
  2. Auto-select DDP backend (`nccl` for CUDA, `gloo` otherwise); expose override.
  3. Add CPU DDP smoke job in CI.
- **Acceptance**: Seed reproducibility test (two runs same seed → identical loss trace); CI job green.

### 8. Documentation & licensing polish
- **Scope**: `pyproject.toml`, README, release docs.
- **Actions**:
  1. Align license declaration with `LICENSE` (Apache-2.0).
  2. Ensure all referenced scripts are shipped; add `scripts/run_e2e_smoke.sh`.
  3. Update README with memorization instructions and Muon requirements.
- **Acceptance**: Lint job confirms license metadata; README diff reviewed.

These items are **blocking** for Stage 2 long runs. Only after P0 checklist completion do we resume the training/eval roadmap below.

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
