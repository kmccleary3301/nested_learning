# Changelog

All notable changes to this project will be documented here. The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses semantic versioning once tagged releases begin.

## [Unreleased]
- Automation script to chain `uv sync → data sample → smoke train → eval`.
- GitHub Actions workflow covering `ruff`, `mypy`, and `pytest`.
- End-to-end release dry-run ahead of the `v0.1.0` tag.

## [0.1.0] - 2025-11-09
### Added
- PyTorch **2.9.0** / torchvision **0.24.0** environment managed via `uv` with reproducible `pyproject.toml` + `uv.lock`.
- HOPE block implementation (attention → TITAN memory → CMS + deep optimizers) with configurable level clocks and self-modifier wiring.
- Hydrated Hydra config tree for pilot, mid, target, and CPU-only smoke runs plus DDP/FSDP/DeepSpeed entrypoints.
- Data tooling: tokenizer trainer, corpus filtering, mixture processing, and `scripts/data/run_sample.sh` shortcut emitting stats under `data/mixtures/`.
- Evaluation suite: zero-shot benchmark CLI (PIQA/HellaSwag/WinoGrande/ARC/BoolQ/SIQA), Needle-in-a-Haystack generator, continual-learning forgetting analyzer.
- Sample artifacts (`artifacts/examples/pilot_dummy.pt`, `logs/pilot_smoke.json`, `logs/mid_smoke.json`) for reproducing eval commands without lengthy training.
- Documentation set (`docs/stage1_plan.md`, `docs/stage2_plan.md`, `docs/data_pipeline.md`, `docs/guide.md`) outlining architecture, scaling strategy, and onboarding.

### Changed
- README rewritten with badges, quickstart commands, and references to the new guide + release checklist.
- Logging defaults clarified (`logging.backend=json|wandb`), with instructions for saving structured metrics under `logs/`.

### Known gaps
- Release automation and CI are tracked in `docs/release_plan.md`.
- Scaling guidance for >100 B token corpora pending additional storage + GPU availability.
