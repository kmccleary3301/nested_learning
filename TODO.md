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
