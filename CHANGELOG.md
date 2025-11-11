# Changelog

All notable changes to this project will be documented here. The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses semantic versioning once tagged releases begin.

## [Unreleased]
### Added
- Flash/SDPA-backed self-attention path with safe fallbacks, unlocking PyTorch 2.9 SDPA kernels by default.
- Hydra toggles for bf16 autocast (`train.mixed_precision.*`), `torch.compile` (`train.compile.*`), and fused optimizers.
- Muon + AdamW hybrid optimizer option exposed via `optim.type=muon`, routing ≥2D matrices through `torch.optim.Muon`.
- Test-time memorization flags (`--memorize*`) documented in README + `docs/guide.md`, matching TITAN eval behavior.
- Automation helpers: `scripts/run_e2e_smoke.sh` documented in Quickstart, plus new `scripts/run_cpu_ddp_smoke.sh` for CPU-only DDP/gloo smoke coverage.

### Changed
- Repository license metadata now matches the shipped Apache-2.0 text; badges updated accordingly.
- README and guide refreshed with performance knobs, optimizer guidance, and memorization instructions so release consumers have a single source of truth.
- Release checklist tracks the new CPU DDP smoke script to keep packaging instructions aligned with available tooling.

### Upcoming
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
