# Stability Journal – Nested Learning Reproduction

_Chronological notes on debugging and stabilizing the HOPE/TITAN implementation. Useful for future contributors digging into NaN fixes or regression hunting._

---

## Day 1–2: Initial Smoke Wiring
- Implemented HOPE block, TITAN memory, CMS chain based on planner transcript.
- `scripts/run_smoke.sh` verified pilot/mid configs on CPU; produced JSON logs (`logs/pilot_smoke.json`, `logs/mid_smoke.json`).
- Data sample pipeline (`scripts/data/run_sample.sh`) exercised each dataset filter/shard command.

## Day 3: DDP Bring-up (mid configs)
- First DDP attempts (`torchrun --nproc_per_node=2 train_dist.py --config-name mid`) failed due to missing Hydra overrides and `model.teach_scale=0`.
- Added tuneable teach scale, JSON logging, and checkpointing; DDP still OOM’d at batch = 16 on dual RTX 6000.
- Dropped per-GPU batch to 8 and later 4; DDP reached 80 steps but NaN’d immediately afterward.

## Day 4: Teach-Scale Issues
- Observed NaNs exactly when `teach_scale` exceeded 0.05.
- Added global teach clip and runtime schedule; partial alleviation but still unstable beyond 80 steps.
- Documented this in `docs/stage2_progress.md` and `TODO.md`.

## Day 5: Single-GPU Focus + tmux
- Due to limited GPU availability, shifted to single-GPU (`cuda:1`) runs.
- Created tmux workflows (`mid_ts10_short`, `mid_stage2_run`) so long jobs could continue while debugging.
- Teach-scale sweeps (0.05/0.10/0.20) confirmed per-layer residual clipping was necessary.

## Day 6: Gradient Clipping Inside Modules
- Added clipping to CMSBlock and TitanMemory forward paths.
- Introduced runtime teach schedule (warmup + decay) applied before each forward.
- With lr=1.5e-5 and batch = 4, HOPE runs finally passed 120 steps without NaNs.

## Day 7: Extended Run & DDP Regression
- Attempted 220-step single-GPU run using more aggressive decay (start 140, duration 80). Achieved stable checkpoint `step_000220.pt`.
- DDP runs with dual GPUs still diverge around step 80 unless batch reduced to 2 per GPU; further optimization deferred.

## Day 8: TITAN Baseline
- Added `model.type=titan` path so HOPE vs TITAN comparisons share code.
- Ran TITAN baseline to 200 steps; zero-shot metrics matched HOPE at this scale, confirming parity of setup.

## Lessons Learned
1. Teach-scale management is critical; unbounded signals quickly blow up CMS/TITAN memories.
2. Residual clipping inside memory blocks avoids relying solely on optimizer gradient clipping.
3. tmux is essential for long jobs in the CLI environment; all commands in this journal reference their tmux usage where applicable.
4. Keeping `logs/*.json` for every run made it easy to track regressions; these logs should be preserved (even if not committed) during future experiments.

## Next Stability Tasks
- Integrate gradient checkpointing or FSDP to revisit dual-GPU runs without slashing batch size.
- Experiment with per-level teach scales (fast CMS vs slow) to reduce interference.
- Add automated anomaly detection (e.g., torch autograd detect_anomaly) around self-mod updates for easier debugging.
