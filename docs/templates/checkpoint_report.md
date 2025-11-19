# Checkpoint Report Template

Copy this template into `reports/checkpoints/<run>.md` (or similar) for every published checkpoint.

## 1. Run Summary
- **Model / Config:** (e.g., HOPE pilot, `configs/pilot.yaml`)
- **Checkpoint path:** `artifacts/checkpoints/...`
- **Hydra overrides:** `...`
- **Tokens seen / steps:** (e.g., 3 B tokens / 230 k steps)
- **Outer optimizer:** (Muon/AdamW + settings)
- **Inner optimizer variant:** (`nl_l2_precond`, etc.)
- **Teach schedule:** warmup/decay parameters

## 2. Environment
- Git commit SHA
- PyTorch / CUDA / cuDNN versions
- `uv.lock` hash
- Tokenizer path + SHA256

## 3. Training Metrics
- Plot or table for loss/ppl vs step (include teach-signal norm)
- Gradient norms (global + per-level if available)
- Notable events (OOM retries, restarts)

## 4. Memory-System Telemetry
- Average `layer*.titan.*.grad_norm` and projector norms
- CMS chunk stats (`chunk_samples`, updates per 1k tokens)
- Surprise/memorization triggers (counts, thresholds)

## 5. Evaluation
- Zero-shot table (baseline vs memorize accuracy)
- NIAH accuracies by context length + memorize deltas
- Continual CE per segment (baseline vs memorize)
- Additional diagnostics (LongBench, PG-19, etc.)

## 6. Reproduction Commands
```
# train
uv run python train.py --config-name ...
# eval
uv run python scripts/eval/zeroshot.py ...
```

## 7. Risks / Notes
- Known deviations from the paper
- TODOs before scaling this checkpoint (e.g., data quirks, missing ablations)

---

Use this structure to keep every release auditable and to make comparisons across HOPE/TITAN checkpoints straightforward.
