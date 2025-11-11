# Experiments Report – Nested Learning Reproduction

_Draft covering work completed through 9 Nov 2025. This document is meant to accompany the initial public release so contributors understand what has been reproduced and what remains._

---

## 1. Overview
- **Goal:** Reproduce key aspects of Google's Nested Learning (HOPE) architecture using public tooling (`uv`, PyTorch 2.9.0) and release a community-ready codebase.
- **Hardware:** Dual RTX 6000 Ada (49 GB each). All long-running experiments in this report use a single GPU (`cuda:1`) to accommodate other projects on the host.
- **Data:** Filtered RefinedWeb mixture (FineWeb, Wikipedia, C4, SlimPajama, CodeParrot). Sample pipeline (`scripts/data/run_sample.sh`) for smoke tests; full pipeline (`scripts/data/run_full.sh`) for larger runs. Tokenizer: SentencePiece unigram 32k.

---

## 2. Experimental Setup
| Component | Details |
|-----------|---------|
| Framework | PyTorch 2.9.0 (LTS), CUDA 12.4 |
| Dependency Mgmt | `uv` with `pyproject.toml` + `uv.lock` |
| Logging | JSON logs under `logs/` (W&B optional but disabled for release) |
| Training Driver | `train.py` (single GPU), `train_dist.py` (torchrun) |
| Evaluation | `scripts/eval/zeroshot.py`, `scripts/eval/niah.py`, `scripts/eval/continual.py` |
| Teach Signal | Outer teach signal derived from logits residual; scale/clip adjustable per config with runtime scheduling |

### Key Configurations
1. **HOPE Mid (single GPU)**
   - Config: `configs/mid_stage2.yaml`
   - Dim = 768, 18 layers, 12 heads, TITAN-level + CMS levels (fast/mid/slow/ultra)
   - Teach schedule: warmup 60 steps, decay start 140, duration 80 (for 220-step run)
   - Gradient clipping applied inside TITAN and CMS blocks

2. **TITAN Baseline**
   - Config: `configs/mid_titan_baseline.yaml` (`model.type=titan`)
   - Same backbone (attention + TITAN memory) but no CMS/self-mod update path
   - Teach schedule mirrors HOPE run to enable apples-to-apples comparison

---

## 3. Experiments

### 3.1 Data Pipeline Validation
| Command | Purpose |
|---------|---------|
| `uv run bash scripts/data/run_sample.sh` | Smoke-friendly filtering + sharding (RefinedWeb/Wiki/C4/SlimPajama/Code) |
| `RW_LIMIT=20000 ... uv run bash scripts/data/run_full.sh` | Full pipeline (run in tmux `data_full`) to produce `_full` shards |
| `uv run python scripts/data/process_mixture.py configs/data/refinedweb_mixture_full.yaml ...` | Re-sharding with SentencePiece tokenizer |

Artifacts: `data/filtered/*_full.txt`, `data/shards/*_full`, stats in `data/mixtures/refinedweb_mix_full_shards.json`.

### 3.2 HOPE vs TITAN (single GPU, 220 steps)
All runs below use batch size 4, optimizer LR 1e‑5, teach_scale 0.10, teach_clip 4.0, runtime schedule (warmup 60, decay 140→220). Commands launched via tmux to keep the CLI free.

| Model | Checkpoint | PIQA (128) | Winogrande (128) | Notes |
|-------|------------|------------|------------------|-------|
| HOPE | `artifacts/checkpoints/mid_stage2_ts10_single220_schedD/step_000220.pt` | 0.469 | 0.594 | Loss drops from 10.55 → 8.55; NIAH still ~0 |
| TITAN | `artifacts/checkpoints/mid_titan_baseline/step_000200.pt` | 0.469 | 0.594 | Loss similar; continuous memory absent |

NIAH results (`eval/niah_mid_stage2_ts10_single220_schedD.json`, `eval/niah_mid_titan_baseline.json`) remain near random at 2k/4k tokens for both models. Continual-learning logs are finite but noisy (short runs). A longer training window is needed to expose the advantages cited in the paper (e.g., HOPE surpassing TITAN on long-context recall).

### 3.3 Teach-Scale Sweep (short runs)
| teach_scale | Configuration | Checkpoint | Final loss (step 40) |
|-------------|---------------|------------|----------------------|
| 0.05 | `logs/mid_stage2_single_ts05.json` | `artifacts/checkpoints/mid_stage2_single_ts05/step_000040.pt` | 9.81 |
| 0.10 | `logs/mid_stage2_single_ts10.json` | `artifacts/checkpoints/mid_stage2_single_ts10/step_000040.pt` | 9.77 |
| 0.20 | `logs/mid_stage2_single_ts20.json` | `artifacts/checkpoints/mid_stage2_single_ts20/step_000040.pt` | 9.76 |

Even at 0.20, residual clipping kept the run stable, indicating headroom for larger teach scales once the data window grows.

### 3.4 Dual-GPU Smoke (HOPE)
| Command | Output |
|---------|--------|
| `uv run torchrun --nproc_per_node=2 train_dist.py --config-name mid_stage2_smoke` | `artifacts/checkpoints/mid_stage2_smoke/step_000060.pt`, `logs/mid_stage2_smoke.json` |
| `uv run python scripts/eval/zeroshot.py ...` | `eval/zeroshot_mid_stage2_smoke.json` |
| `uv run python scripts/eval/niah.py ...` | `eval/niah_mid_stage2_smoke.json` |
| `uv run python scripts/eval/continual.py ...` | `eval/continual_mid_stage2_smoke.json` |

These runs validate the distributed training/eval path and are the recommended “smoke” workflows for contributors.

### 3.5 Test-Time Memorization Harness
HOPE/TITAN models now support TITAN-style test-time learning via shared CLI flags:

```
uv run python scripts/eval/zeroshot.py \
  --config configs/mid_stage2_smoke.yaml \
  --checkpoint artifacts/checkpoints/mid_stage2_smoke/step_000060.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --tasks piqa \
  --max-samples 32 \
  --output eval/zeroshot_mid_stage2_smoke_piqa_mem.json \
  --device cuda:1 \
  --memorize \
  --memorize-steps 2 \
  --memorize-use-correct-answer
```

NIAH and continual harnesses expose analogous options (`--memorize`, `--memorize-steps`, `--memorize-no-reset`, `--memorize-use-correct-answer`). The memorization loop replays the prompt (optionally augmented with the correct answer) through the teach-signal pathway before each eval query, letting us probe TITAN-style “learning at test time”.

Pilot PIQA example (32-sample subset, single GPU):

| Mode | Command / Output | Accuracy |
|------|------------------|----------|
| Baseline | `eval/zeroshot_mid_stage2_smoke_piqa_baseline.json` | 0.5625 |
| Memorize (prompt + answer, 2 steps) | `eval/zeroshot_mid_stage2_smoke_piqa_mem.json` | 0.5625 |

At this scale, memorization neither helps nor hurts, but the infrastructure is in place to replicate the substantial gains reported in HOPE/TITAN once longer contexts and richer checkpoints are available.

### 3.6 Pilot (3 B tokens) – in progress
- **Config:** `configs/pilot.yaml` (dim = 512, 12 layers, TITAN + CMS fast/mid/slow/ultra, teach_schedule warmup 2k → decay 120k→140k).
- **Run target:** 246 667 steps (batch 6 × seq 2048 ≈ 3.03 B tokens) on `cuda:1`, lr = 2.5e‑4 AdamW + Muon option disabled for parity with Stage 1.
- **Logging:** W&B project `nested-learning`, run `pilot-main-20251111161514` (`https://wandb.ai/lsu-kmccleary-0/nested-learning/runs/oy73f2m6`). Local console mirrors under `wandb/run-*/files/output.log`.
- **Status:** Step 150 reached with loss dropping from 93.4 → 40.2. Estimated wall-clock ≈52 hours for full 3 B tokens. Checkpoints saved every 1 000 steps to `artifacts/checkpoints/pilot/`.
- **Packaging:** `artifacts/pilot_release/` now contains a manifest stub (`metadata.json`, README) so the eventual checkpoint, config, logs, and eval outputs can be bundled for contributors without rerunning the long job.
- **Eval prep:** Zero-shot, NIAH (up to 64 k contexts), and continual-learning scripts were dry-run against the sample `artifacts/examples/pilot_dummy.pt` checkpoint to confirm the new memorization/compatibility flows before the real pilot weights land.

---

## 4. Observations & Lessons Learned
1. **NaNs past 80 steps:** Early runs blew up after 80 steps once teach_scale exceeded 0.05. Introducing runtime scaling + residual clipping inside TITAN/CMS eliminated the NaNs and allowed 220-step runs on a single GPU.
2. **Batch-size constraints:** With only one GPU, we reduced per-GPU batch to 4 to stay within 49 GB VRAM. DDP runs will need gradient checkpointing or FSDP to scale further.
3. **NIAH is data hungry:** Every HOPE/TITAN run so far shows near-random recall at 2k/4k tokens; longer contexts and more tokens are required to differentiate architectures.
4. **Teach signal scheduling:** A linear warmup (60 steps) followed by linear decay (start 140) kept the 220-step run stable. Future runs should explore cosine or per-level schedules.

---

## 5. Limitations
- Training horizon limited to ~220 steps / ≈150k tokens due to hardware and time; no large-scale reproduction yet.
- No TITAN vs HOPE comparison on long-context or continual benchmarks beyond short runs.
- DDP/TITAN runs still rely on JSON logging; integration with structured logging (e.g., W&B) is deferred to future contributors.
- Pipeline uses filtered RefinedWeb proxies; exact data parity with Google’s internal corpora is not guaranteed.

---

## 6. Next Steps
1. **Longer Runs:** Extend both HOPE and TITAN baselines to millions of tokens using FSDP/DeepSpeed (target ≥760 M parameter config).
2. **Eval Coverage:** Integrate full RAFT/ARC suite plus additional long-context datasets (Needle-in-a-Haystack 32k, PassKey tasks).
3. **HPO:** Once stable runs exist, sweep teach_scale/clip, CMS depth, and self-mod learning rates to quantify HOPE vs TITAN gains.
4. **Automation:** Add CI for data sampling + dual-GPU smoke to catch regressions, and consider nightly tmux scripts for longer training jobs.

---

## 7. References
- `docs/stage2_progress.md` – running log of all Stage 2 work.
- `docs/stability_journal.md` – chronological notes on NaN fixes, teach-scale tuning, tmux jobs.
- `reports/stage2_smoke.md` – command cheat sheet for reproducing the smoke runs referenced here.

This report will be updated as we push beyond short runs and start reproducing the full metrics from Google's Nested Learning paper.
