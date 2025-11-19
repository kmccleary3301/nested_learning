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
- Config: `configs/pilot.yaml` (dim 512, 12 layers, teach_scale 0.10, CMS fast/mid/slow/ultra). Batch 6 × seq 2048 for a 3.03 B-token target at 246 667 steps; runs on `cuda:1` with Muon optimizer + bf16 autocast/SDPA/`torch.compile`.
- **Long-run status (13 Nov):** Main tmux job (`pilot_full`) advanced to step 246 667 before pausing; checkpoints live under `artifacts/checkpoints/pilot/step_*.pt`. We standardize on step 230 000 (`artifacts/checkpoints/pilot/step_230000.pt`) for the release drop because its metrics stabilized and it precedes the LR cooldown.
- **Release packaging:** `scripts/package_pilot_release.sh artifacts/checkpoints/pilot/step_230000.pt` now copies the checkpoint to `artifacts/pilot_release/checkpoint.pt`, syncs `config.yaml`, and refreshes `MANIFEST.txt`/`metadata.json`. The bundle also carries tmux logs plus TITAN baselines so downstream users can download a single directory.
- **Eval refresh:** Re-ran the suite with memorization enabled (device `cuda:1`) which writes:
  - `eval/zeroshot_pilot_step230000.json` (PIQA 0.496 @256, HellaSwag 0.297, Winogrande 0.473, ARC-E 0.285, ARC-C 0.234, BoolQ 0.367, SIQA 0.316, CSQA 0.180, OpenBookQA 0.113).
  - `eval/niah_pilot_step230000.json` (0.625 @2k → 0.50 @65k context lengths).
  - `eval/continual_pilot_step230000.json` (CE ≈8.06/7.79/7.68/7.95 on RefinedWeb/Wiki/C4/RedPajama segments).
  Copies of these JSONs now sit inside `artifacts/pilot_release/` next to the historical step 22k dumps for comparison.
- **Automation:** `scripts/eval/run_pilot_suite.sh` will pick up the latest checkpoint automatically; override `HOPE_CHECKPOINT=artifacts/checkpoints/pilot/step_230000.pt` to re-run exactly this snapshot.
- **Next:** Keep checkpoints under version control (ignored by git) for ablations (teach-scale, CMS chunking, optimizer swaps) while TITAN catches up to a comparable step count.

### 2.4 TITAN baseline (short + long runs)
- Config: `configs/mid_titan_baseline.yaml` (TITAN-only stack, same teach schedule/optimizer as pilot).
- **Short-run snapshot:** 9 000 steps on `cuda:1` with checkpoints every 500 steps (`artifacts/checkpoints/mid_titan_baseline/step_009000.pt`). PIQA = 0.4922, NIAH (2k/4k/8k) = 0.5, continual CE ≈ 12–14.
- Bundle: copied into `artifacts/pilot_release/` alongside the HOPE checkpoint so both models share the same manifest/eval files.
- **Long run complete (13 Nov, GPU 0):** After fixing `/tmp` exhaustion via `TMPDIR=/mnt/drive_4/tmp_titan`, the 25 k-step job finished cleanly (W&B `titan-long-20251113192738`). Final checkpoint: `artifacts/checkpoints/mid_titan_baseline/step_025000.pt` (also copied to `artifacts/pilot_release/titan_step_025000.pt`).
- **Eval suite:** `eval/zeroshot_titan_step25000.json` (PIQA 0.484, HellaSwag 0.293, Winogrande 0.480, ARC-E 0.281, ARC-C 0.250, BoolQ 0.398, SIQA 0.293, CSQA 0.188, OBQA 0.145), `eval/niah_titan_step25000.json` (0.50/0.625/0.125/0.75/0.50/0.125 across 2 k→65 k contexts), and `eval/continual_titan_step25000.json` (CE ≈8.36/8.12/7.85/8.11). Copies live in `artifacts/pilot_release/` for download.
- Next: Use the Titan step 25 k metrics as the baseline in `docs/experiments_report.md`/`reports/ablations.md` and start the planned ablations (teach-scale, CMS chunking, optimizer swaps) against the HOPE checkpoint tree.

### 2.5 Teach-scale ablations (pilot)
- **teach_scale=0.05 (GPU 0, 2 k steps):** Checkpoints at `artifacts/checkpoints/pilot_teach05/step_{001000,002000}.pt`, log `logs/pilot-teach05-20251114010549.json`. Evals show PIQA 0.453 / HellaSwag 0.273 / Winogrande 0.508, NIAH 0.50→1.00 across 2k→65k, and continual CE ≈37–33.
- **teach_scale=0.05 long (GPU 1, 25 k steps):** Checkpoints under `artifacts/checkpoints/pilot_teach05_long/`, log `logs/pilot-teach05-long-20251114155521.json`. Evals: `zeroshot_pilot_teach05_long_step25000.json` (PIQA 0.508, ARC-E 0.320, etc.), `niah_pilot_teach05_long_step25000.json` (0.25 / 0.50 / 0.375 / 0.75 / 0.75 / 0.75), `continual_pilot_teach05_long_step25000.json` (CE ≈52 / 49 / 49 / 51).
- **teach_scale=0.15 (GPU 1, 2 k steps):** Checkpoints at `artifacts/checkpoints/pilot_teach15/step_{001000,002000}.pt`, log `logs/pilot-teach15-20251114012109.json`. Evals: PIQA 0.484 / HellaSwag 0.258 / Winogrande 0.461, NIAH scores 0.75/0.75/0.75/0.50/0.25/0.50, continual CE ≈69–66 (expected because the run only saw 2 k steps).
- **teach_scale=0.15 long (GPU 1, 25 k steps):** Checkpoints under `artifacts/checkpoints/pilot_teach15_long/`, log `logs/pilot-teach15-long-20251114185448.json`. Evals: `zeroshot_pilot_teach15_long_step25000.json` (PIQA 0.496, HellaSwag 0.305, Winogrande 0.500), `niah_pilot_teach15_long_step25000.json` (0.75 / 0.625 / 0.375 / 0.75 / 0.50 / 0.75), `continual_pilot_teach15_long_step25000.json` (CE ≈7.9 / 7.6 / 7.6 / 7.8).
- Takeaway: 0.05 offers modest long-context gains but hurts continual even after a long run, while 0.15 regains solid continual metrics and competitive zero-shot scores, so the mid/high teach scales remain the most promising defaults.

### 2.6 CMS chunk ablation (update_period=1)
- **Run:** `pilot-cms-nochunk-20251114232720` on GPU 1 (5 k steps) with all CMS levels forced to `update_period=1` (no chunk accumulation).
- **Artifacts:** Checkpoints `artifacts/checkpoints/pilot_cms_nochunk/step_*.pt`, JSON log `logs/pilot-cms-nochunk-20251114232720.json`.
- **Evals:** `zeroshot_pilot_cms_nochunk_step5000.json` (PIQA 0.520, HellaSwag 0.277, Winogrande 0.473, BoolQ 0.633, etc.), `niah_pilot_cms_nochunk_step5000.json` (0.75 / 0.25 / 0.25 / 0.25 / 0.75 / 0.50), `continual_pilot_cms_nochunk_step5000.json` (CE ≈46.6 / 47.8 / 49.7 / 52.1).
- Takeaway: removing chunk accumulation boosts some zero-shot scores (e.g., BoolQ) but causes significant continual-learning degradation, reinforcing the need for chunked CMS updates.

### 2.7 Self-modifier ablation (self_mod_lr=0)
- **Run:** `pilot-selfmod-off-20251115132848` on GPU 1 (5 k steps) with `model.self_mod_lr=0`.
- **Artifacts:** Checkpoints `artifacts/checkpoints/pilot_selfmod_off/step_*.pt`, JSON log `logs/pilot-selfmod-off-20251115132848.json`.
- **Evals:** `zeroshot_pilot_selfmod_off_step5000.json` (PIQA 0.516, BoolQ 0.633, etc.), `niah_pilot_selfmod_off_step5000.json` (0.75 / 0.75 / 0.50 / 0.75 / 0.25 / 0.75), `continual_pilot_selfmod_off_step5000.json` (CE ≈45.7 / 44.9 / 44.4 / 45.5).
- Takeaway: turning off the self-modifier leaves zero-shot performance roughly flat but dramatically worsens continual losses, matching the paper’s claim that test-time modulation is necessary for HOPE’s continual-learning behaviour.

### 2.8 CMS sparse-chunk ablation (periods 8→512)
- **Goal:** Stress-test Eq. 31’s chunk accumulation at extreme update periods without blowing GPU memory.
- **Config:** `configs/ablations/cms_sparse.yaml` (dim 384, 8 layers, seq 1024, batch 2, CMS hidden multiplier = 2, update periods 8/32/128/512). Exported resolved copy for evals at `configs/resolved/cms_sparse_eval.yaml`.
- **Run:** `pilot-cms-sparse-20251115165307` (5 k steps, GPU 1) with `PYTORCH_ALLOC_CONF=expandable_segments:True`.
- **Artifacts:** `artifacts/checkpoints/pilot_cms_sparse/step_005000.pt`, logs `logs/pilot_cms_sparse_metrics_{run}.json`.
- **Evals:** `zeroshot_pilot_cms_sparse_step5000.json` (PIQA 0.516, BoolQ 0.367, HellaSwag 0.258, Winogrande 0.500), `niah_pilot_cms_sparse_step5000.json` (0.75 / 0.50 / 0.625 / 0.625 / 0.50 / 0.375), `continual_pilot_cms_sparse_step5000.json` (CE ≈25.0 across the four segments).
- **Takeaway:** Increasing chunk size to 512 preserves zero-shot accuracy similar to the default chunked run but slashes continual loss relative to the no-chunk variant (~25 CE vs. high-40s), reinforcing that chunk accumulation + sparse updates are vital to HOPE’s continual-learning claims. Memory pressure is manageable (<42 GB) with the downsized configuration.

### 2.9 Optimizer ablation (fused AdamW vs Muon hybrid)
- **Goal:** Compare PyTorch 2.9’s native Muon optimizer (matrix-only) versus the fused AdamW default on the pilot config.
- **Setup:** Both runs use `configs/pilot.yaml` with `train.steps=5000`, batch = 6, seq = 2048, GPU 1 only. Mixed-precision bf16 + SDPA/compile remain enabled.
- **AdamW run:** `pilot-opt-adamw-20251115173858` → checkpoint `artifacts/checkpoints/pilot-opt-adamw-20251115173858/step_005000.pt`, log `logs/pilot-opt-adamw-20251115173858.log`. Metrics:
  * Zero-shot (`eval/zeroshot_pilot_opt_adamw_step5000.json`): PIQA 0.559, HellaSwag 0.273, Winogrande 0.500, BoolQ 0.367.
  * NIAH (`eval/niah_pilot_opt_adamw_step5000.json`): accuracies {0.75, 1.00, 0.50, 0.75, 0.50, 0.25}.
  * Continual (`eval/continual_pilot_opt_adamw_step5000.json`): CE ≈50.1 / 43.3 / 39.3 / 38.7.
- **Muon hybrid run:** `pilot-opt-muon-20251115180139` (Muons on ≥2D tensors, AdamW elsewhere) → checkpoint `artifacts/checkpoints/pilot-opt-muon-20251115180139/step_005000.pt`.
  * Zero-shot (`eval/zeroshot_pilot_opt_muon_step5000.json`): PIQA 0.531, HellaSwag 0.313, Winogrande 0.484, BoolQ 0.570.
  * NIAH (`eval/niah_pilot_opt_muon_step5000.json`): {0.50, 0.50, 0.25, 0.75, 0.75, 0.75}.
  * Continual (`eval/continual_pilot_opt_muon_step5000.json`): CE ≈11.3 / 11.3 / 11.2 / 10.8.
- **Takeaway:** Muon aggressively reduces continual loss (~4×) and boosts BoolQ/NIAH long-context retention at the expense of a slight PIQA dip. We’ll standardize on Muon for the resumed HOPE long run (step > 230 k) and keep the AdamW checkpoints as baseline references. Next action: re-launch the paused HOPE `pilot_full` tmux job with `optim.type=muon` and teach_scale=0.10 after the TITAN baseline frees GPU 0.

### 2.10 Long-context diagnostics + forgetting plots
- **Passkey eval:** `scripts/eval/passkey.py` now runs as part of the suite. Pilot step 230 k yields `eval/passkey_pilot_step230000.json` (64 prompts, filler = 256) with baseline vs memorize accuracy 0.484 and Titan updates ≈2.13. TITAN baseline outputs live in `eval/passkey_titan_step25000.json`.
- **PG‑19 perplexity:** Streaming PG‑19 excerpts truncated to 2 048 tokens, memorization optional. Pilot checkpoint: `eval/pg19_pilot_step230000.json` (ppl ≈ 2.5 k). TITAN baseline: `eval/pg19_titan_step25000.json` (ppl ≈ 3.1 k).
- **Continual forgetting plot:** multi-checkpoint run (`eval/continual_pilot_multi.json` covering steps 5 k/10 k/230 k) visualized via `scripts/eval/plot_forgetting.py` → `reports/plots/continual_pilot_refinedweb.png`. Use `HOPE_CONT_CHECKPOINTS="ckpt1 ckpt2 ..."` in `scripts/eval/run_pilot_suite.sh` to auto-generate plots for future checkpoints.

## 3. Recommended Workflow for Contributors
1. **Environment** – `uv sync --all-extras && uv run bash scripts/data/run_sample.sh`
2. **Distributed smoke** – `uv run torchrun --nproc_per_node=2 train_dist.py --config-name mid_stage2_smoke`
3. **Eval suite** – Run the three scripts above pointing to `mid_stage2_smoke` checkpoint.
4. **Scaling** – Attach tmux sessions for long jobs:
   - Data: `tmux new -s data_full '... run_full.sh'`
   - Mid-scale: `tmux new -s mid_stage2_run '... torchrun ... mid_stage2'`
5. **Artifacts** – Drop new checkpoints/logs in `artifacts/checkpoints/mid_stage2*`, `logs/`, and store eval JSON under `eval/` with descriptive names. For long pilot runs, copy the resulting checkpoint + config + eval outputs into `artifacts/pilot_release/` so users can download a single bundle.

Documenting these steps here keeps everyone aligned while we chase full Stage 2 parity.

### 2.9 Coverage guard + reporting automation (Nov 16)
- Added `scripts/checks/tokenizer_coverage_guard.py`, which recomputes coverage on `data/filtered/refinedweb_en_sample.txt` and compares against the baseline JSON (`data/mixtures/refinedweb_mix_tokenizer_coverage.json`). The guard is referenced from `docs/data_pipeline.md` / `docs/release_checklist.md` and the latest run wrote `data/mixtures/refinedweb_mix_tokenizer_coverage_latest.json` (guard passed; stats unchanged).
- Extended checkpoint reports beyond the core trio. New files cover `pilot_teach15_long`, `pilot_cms_nochunk`, `pilot_cms_sparse`, `pilot_selfmod_off`, `pilot_opt_muon`, and `pilot_opt_adamw`, each citing the relevant log + eval JSON so release bundles stay uniform.

### 2.10 Checkpoint integrity sidecars (Nov 16)
- All training entry points now emit `.meta.json`, `.sha256`, and `.yaml` files next to every checkpoint plus RNG state captures. Metadata includes checkpoint/config SHA256 digests, tokenizer hash, and pickled RNG states (Python, NumPy, torch CPU/CUDA) for reproducibility.
- `scripts/checkpoint/verify.py` calls the shared verifier (`nested_learning.training.verify_checkpoint_integrity`) to recompute hashes and ensure RNG fields exist; `docs/release_checklist.md` now mandates running it before publishing artifacts.

### 2.11 CI coverage for planner asks (Nov 16)
- `.github/workflows/ci.yml` now has two additional jobs: `cpu-ddp-smoke` (runs `scripts/run_cpu_ddp_smoke.sh` to validate the gloo backend determinism path) and `passkey-smoke` (runs `scripts/tests/run_passkey_smoke.sh` which trains `pilot_smoke`, executes `scripts/eval/passkey.py` against the new `tests/data/tiny_tokenizer.model`, and asserts a positive memorization delta).
- The helper script + tiny tokenizer assets live under `scripts/tests/` and `tests/data/`, so contributors can run `bash scripts/tests/run_passkey_smoke.sh` locally before sending PRs touching memorization code.

### 2.12 Surprise gating + memorize-path controls (Nov 16)
- Added global surprise-threshold wiring to HOPE/TITAN models (`ModelConfig.surprise_threshold`) and level filters so fast memories only update when teach-signal norms exceed the configured gate. Update stats now include `gate_hit` and the observed surprise value, and the verifier ensures checkpoints capture RNG/config hashes as before.
- All memorization CLIs (`zeroshot`, `niah`, `continual`, `passkey`) gained `--memorize-paths` and `--memorize-surprise-threshold`. These feed into `MemorizeConfig` which now restricts updates to selected paths (e.g., Titan-only vs Titan+CMS fast) and temporarily overrides the surprise gate during eval. JSON outputs record both the active paths and surprise thresholds so downstream analysis can trace which memory systems were engaged.

### 2.13 Pilot & TITAN relaunch with surprise gating (Nov 17)
- Relaunched the HOPE pilot job on `cuda:1` using `nohup uv run python train.py --config-name pilot train.step_offset=231000 ...` so the Muon + surprise-gated configuration produces fresh checkpoints beyond the previous 246 k ceiling. PIDs live in `logs/pilot_relaunch_surprise.pid`, streaming output in `logs/pilot_relaunch_surprise.log`, and checkpoints continue under `artifacts/checkpoints/pilot_relaunch/step_*.pt`. Once the next snapshot (e.g., step 247 k+) lands we’ll repackage the pilot release and rerun the eval suite with the updated memorize metadata.
- Relaunched the TITAN long baseline on `cuda:0` (same surprise threshold, Muon outer) via `nohup uv run python train.py --config-name mid_titan_baseline train.step_offset=7000 train.steps=25000 ...`. Outputs/ PIDs are captured in `logs/titan_relaunch_surprise.{log,pid}` and checkpoints land in `artifacts/checkpoints/mid_titan_long/step_*.pt`. Matching eval suites (zero-shot, NIAH, continual, passkey, PG‑19) will run as soon as the 10 k/25 k checkpoints arrive.
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
