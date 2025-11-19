# Project TODOs

## Stage 2 – Results Reproduction
- [ ] **Data Engineering**
  - [ ] Acquire RefinedWeb + supplement corpora under `data/raw/`.
  - [x] Implement filtering/dedup scripts (language ID, length bounds).
  - [x] Run `scripts/data/train_tokenizer.py` on combined corpus and store tokenizer artifacts.
  - [x] Shard each corpus component with `scripts/data/process_mixture.py`; log mixture stats.
  - [x] Automate `sample` and `full` pipelines via `scripts/data/run_sample.sh` / `scripts/data/run_full.sh`.
- [ ] **Infrastructure & Configs**
  - [x] Build Hydra config tree (`configs/hope/`) for pilot/mid/target, including optimizer + level schedules.
  - [x] Integrate logging (W&B/MLflow) hooks into training loop and configs.
  - [x] Provide DeepSpeed + FSDP launcher scripts with resume support.
  - [x] Add CI workflow (`.github/workflows/ci.yml`) for lint/type/tests via `uv`.
- [ ] **Scaling Training**
  - [x] Run pilot (160 M, 3 B tokens) to validate pipeline + self-mod updates. *(Step 230 k packaged 13 Nov; resume after TITAN baseline catches up.)*
  - [ ] Scale to 760 M / 30 B tokens; capture checkpoints + metrics. *(100-step mid run stable; longer runs waiting on teach-scale tuning + compute.)*
  - [ ] Execute 1.3 B / 100 B training with long-context curriculum.
- [ ] **Evaluation Harness**
  - [x] Implement `scripts/eval/zeroshot.py` scaffolding (PIQA baseline).
  - [x] Extend zero-shot harness to cover PIQA/HellaSwag/WinoGrande/ARC-E/C/BoolQ/SIQA/CommonsenseQA/OpenBookQA and document usage.
  - [x] Build NIAH long-context scaffolding script (`scripts/eval/niah.py`).
  - [x] Add continual-learning scripts measuring forgetting over streaming domains.
  - [x] Capture Stage 2 eval packs (zeroshot/NIAH/continual) from pilot checkpoints once stable (step 230 k release).
- [ ] **Ablations & Analysis**
  - [x] Run teach-scale sweep (0.05/0.10/0.15) on pilot checkpoints. *(0.05 & 0.15 short + 25 k long runs logged; see `logs/pilot-teach05-20251114010549.json` and `logs/pilot-teach15-long-20251114185448.json`.)*
  - [x] Run self-modifier off/on comparison at pilot scale.
  - [ ] Test CMS depth variations and optimizer variants.
  - [ ] Compare attention backbones (full vs. sliding vs. DeltaNet).
- [ ] **Baseline Monitoring**
  - [x] Finish TITAN long run (25 k steps, `cuda:0`, TMPDIR `/mnt/drive_4/tmp_titan`) and mirror HOPE packaging/eval workflow.
- [ ] **Documentation & Release**
  - [ ] Maintain experiment logs under `reports/`.
  - [ ] Publish data pipeline instructions + provenance for each corpus.
  - [ ] Summarize final metrics vs. baselines in Stage 2 report.

## Immediate Sprint Focus (Nov 15)
- [x] Design CMS sparse-chunk ablation config that stays within 49 GB (dim 384, seq 1024, batch 2, update periods 8/32/128/512).
- [x] Run CMS sparse-chunk experiment, package checkpoint (`artifacts/checkpoints/pilot_cms_sparse/step_005000.pt`), and produce evals (`eval/*_pilot_cms_sparse_step5000.json`).
- [x] Launch optimizer ablation comparing Muon hybrid vs fused AdamW on pilot-scale smoke (5–10 k steps) and archive eval metrics.
- [x] Roll the new CMS + optimizer findings into `reports/ablations.md`, `docs/stage2_progress.md`, and outline the resulting Stage 2 training plan updates.

## Planner Follow-up (P2)
- [x] Manifest validation report (`scripts/data/validate_mixture.py`) + token overlap stats.
- [x] Tokenizer coverage JSON via `scripts/data/check_tokenizer_coverage.py` + regression guard (`scripts/checks/tokenizer_coverage_guard.py`).
- [x] Extend eval suite with passkey, PG‑19, and continual forgetting plots (see `scripts/eval/run_pilot_suite.sh` + `reports/plots/` output).
- [x] Generate long-context/continual eval artifacts for pilot & TITAN checkpoints (`eval/passkey_*`, `eval/pg19_*`, `eval/continual_*`).
- [x] Fill checkpoint reports (`reports/checkpoints/pilot_step230000.md`, `.../titan_step25000.md`, `.../pilot_teach05_long.md`, CMS variants, optimizer ablations, self-mod off).
- [x] Run the same reporting workflow for future checkpoints (teach15 long, CMS sparse/no chunk, optimizer ablations) before publishing.

## Planner Follow-up (P1)
- [x] Make Muon the default outer optimizer (pilot/mid/target configs), log Muon vs AdamW param counts, and confirm bf16/SDPA/compile flags in training logs.
- [x] Finalize FSDP/ZeRO configs for 760 M / 1.3 B (with grad checkpointing + VRAM notes) and document usage.
- [x] Implement atomic checkpoint sidecars (SHA256, RNG state, tokenizer hash) plus a strict `scripts/checkpoint/verify.py`.
- [x] Extend CI with CPU DDP determinism smoke + synthetic passkey memorization test.

## Stage 2 – Execution Sprint (Nov 17)
- [x] Relaunch HOPE pilot run on `cuda:1` (Muon + surprise gating) and produce fresh checkpoints/logs.
  - Status (Nov 18): run completed/terminated by user request to free GPUs. Last logged train step ≈246 150 (global_step ≈477 000); latest checkpoint on disk `artifacts/checkpoints/pilot_relaunch/step_477000.pt` (to be verified/packaged later).
- [ ] Package the new pilot checkpoint via `scripts/package_pilot_release.sh` and rerun the full eval suite (zeroshot/NIAH/continual/passkey/PG19) with memorize path/threshold metadata.
  - Ready: checkpoint report stub at `reports/checkpoints/pilot_relaunch_step245000.md` (will be updated to final step); eval command to use `MEMORIZE_PATHS=titan,cms_fast`, `MEMORIZE_SURPRISE_THRESHOLD=0.02`.
- [x] Restart TITAN long baseline, mirror the eval suite, and record surprise gating stats.
  - Status (Nov 18): long run finished earlier (PID exited). Latest checkpoint: `artifacts/checkpoints/mid_titan_long/step_032000.pt` (global_step 32 000, to be verified/packaged).
- [ ] Run the mid-scale FSDP config (`configs/hope/mid_fsdp.yaml`), monitor VRAM, and archive checkpoints/logs.
- [ ] Update `reports/checkpoints/` + `reports/ablations.md` with the new HOPE/TITAN results (include memorize paths/surprise thresholds).
- [ ] Refresh `docs/stage2_progress.md`, `docs/experiments_report.md`, and `docs/stage2_plan.md` with the latest execution status and next scaling steps.
