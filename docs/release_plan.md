# Release Readiness Checklist (v0.1)

Goal: Package the Nested Learning reproduction so others can run data prep, pilot training, and evaluation out of the box using modest hardware (dual RTX 6000 Ada). Larger-scale configs remain documented for future scaling/community contributions.

---

## 1. Data Pipeline Deliverables
- [x] Filtering script (`scripts/data/filter_corpus.py`) with language/length/dedup enforcement and `--force-exit`.
- [x] Sample filtered outputs under `data/filtered/` for each corpus component (TinyStories + RefinedWeb/Wiki/C4/SlimPajama/Code).
- [x] Manifest for filtered sharding (`configs/data/refinedweb_mixture_filtered.yaml`) and sample stats (`data/mixtures/refinedweb_mix_filtered_shards.json`).
- [x] Sample-mode convenience script (`scripts/data/run_sample.sh`) chaining `filter_corpus.py` + `process_mixture.py`.
- [x] Full-mode convenience script (`scripts/data/run_full.sh`) mirroring the sample workflow with environment variable overrides for storage paths + shard counts (`configs/data/refinedweb_mixture_full.yaml`).
- [x] README section with a single command to reproduce sample data (Quickstart block).

## 2. Training Entry Points
- [x] Hydra configs for pilot/mid/target (`configs/hope/*.yaml`) referencing filtered shards.
- [x] Training modules support single GPU, DDP, FSDP, DeepSpeed.
- [x] Smoke configs (`configs/pilot_smoke.yaml`, `configs/mid_smoke.yaml`) for CPU verification runs.
- [x] Document logging overrides (`logging.backend=json logging.path=logs/<run>.json`) and ship sample logs in `logs/pilot_smoke.json`, `logs/mid_smoke.json`.
- [x] Provide placeholder checkpoint (tiny random weights) in `artifacts/examples/pilot_dummy.pt` to exercise eval scripts without training.

## 3. Evaluation Harness
- [x] Zero-shot CLI covering PIQA/HellaSwag/WinoGrande/ARC-E/C/BoolQ/SIQA.
- [x] NIAH long-context CLI.
- [x] Continual-learning CLI with sample segments YAML.
- [x] README section consolidating eval commands, pointing to sample checkpoint/log outputs.

## 4. Documentation
- [x] Convert `docs/stage1_plan.md` and `docs/stage2_plan.md` highlights into `docs/guide.md` (setup → data → training → eval).
- [x] Add `CHANGELOG.md` capturing v0.1 scope and open work (scaling to ≥500M, full corpus ingestion).
- [x] README badges (tests passing, python version, license) and quickstart snippet.
- [ ] `CONTRIBUTING.md` with TODO list and instructions for filing issues/PRs.

## 5. Automation
- [x] `scripts/run_e2e_smoke.sh` orchestrates `uv sync` → sample data prep → pilot smoke training (with checkpoint) → zero-shot eval (`piqa`).
- [x] GitHub Actions workflow (`.github/workflows/ci.yml`) running `uv run ruff check .`, `uv run mypy src`, and `uv run pytest` on pushes/PRs.

## 6. Release Packaging
- [ ] Tag `v0.1.0` once the above checkboxes are complete.
- [ ] Publish release notes summarizing current capabilities, hardware assumptions, and roadmap for scaling (H200 cluster, 500M+ configs, long-context >32k).

---

**Execution Order:**
1. Implement smoke configs/logs + data convenience scripts (sample done, full-scale pending).
2. Maintain README/guide/changelog as the single source of truth for setup/data/train/eval.
3. Expand automation (now `scripts/run_e2e_smoke.sh`) and add CI.
4. Cut release tag after verifying `scripts/run_e2e_smoke.sh` completes successfully on clean clone.

**Release procedure draft (v0.1.0)**
1. Start from a clean working tree (fresh clone recommended), then run `uv run bash scripts/run_e2e_smoke.sh DEVICE=cpu`.
2. Verify outputs: checkpoint under `artifacts/checkpoints/pilot_smoke`, logs under `logs/`, eval metrics under `eval/`.
3. Update `CHANGELOG.md` with final notes, confirm `docs/release_plan.md` all required items checked.
4. Tag `v0.1.0` and publish release notes summarizing capabilities, hardware assumptions, and next-step roadmap.
