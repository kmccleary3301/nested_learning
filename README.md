# Nested Learning Reproduction

![CI](https://github.com/kmccleary3301/nested_learning/actions/workflows/ci.yml/badge.svg)
![Security](https://github.com/kmccleary3301/nested_learning/actions/workflows/security.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20to%203.12-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.9.0-red)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Status](https://img.shields.io/badge/tests-smoke--ready-lightgrey)

Mechanism-level reproduction of Google's Nested Learning (HOPE) architecture (HOPE blocks, CMS, and Self‑Modifying TITANs), matching the quality bar set by lucidrains' TITAN reference while remaining fully open-source and `uv` managed.

Faithfulness scope (high level):
- ✅ HOPE / CMS / Self‑Modifying Titans update rules + wiring (mechanism-level)
- ✅ Tensor-level invariants covered by unit tests (teach-signal, δℓ, CMS chunking, causality)
- ✅ Boundary-target online chunking + optional attention-cache carry path are implemented
- ⚠️ Stable default uses stop-grad online writes; an experimental single-process boundary-state mode supports differentiable write paths
- ⚠️ Multi‑GPU mechanism-auditing online updates are not supported in this repo (DDP disables some features)

Paper reference pin:
- Source: `google_papers/Nested_Learning_Full_Paper/Nested_Learning_Full_Paper.md`
- SHA-256: `7524af0724ac8e3bad9163bf0e79c85b490a26bc30b92d96b0bdf17a27f9febc`

## Quickstart
```bash
uv python install 3.12
uv sync --all-extras
uv run nl doctor --json > logs/runtime_doctor.json
uv run bash scripts/data/run_sample.sh
uv run nl smoke --config-name pilot_smoke --device cpu
uv run bash scripts/run_smoke.sh pilot  # CPU-friendly HOPE block smoke test
uv run bash scripts/run_e2e_smoke.sh    # sync + sample data + smoke train + zeroshot eval
uv run bash scripts/run_mechanism_audit_smoke.sh
uv run python scripts/eval/zeroshot.py \
  --config configs/hope/pilot.yaml \
  --checkpoint artifacts/examples/pilot_dummy.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --tasks piqa --max-samples 32 --device cpu
```

## Requirements
- Python 3.10-3.12
- PyTorch 2.9.x+ (golden environment in this repo uses 2.9.x)
- `uv` (recommended for development) or `pip` for package-style usage

## Compatibility
- Support tiers and OS/runtime matrix: `docs/COMPATIBILITY_MATRIX.md`
- Versioning/stability policy: `docs/VERSIONING_POLICY.md`
- Golden repro environment: Python 3.12 + `uv lock` + PyTorch 2.9.x

## Installation (pip-first)
1. Create and activate a virtual environment.
2. Install Torch first (CPU/CUDA wheel selection is backend-specific).
3. Install this project.

CPU example:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch>=2.9,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

CUDA example (adjust index URL to your CUDA runtime):
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch>=2.9,<3" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

## Setup (uv dev workflow)
```bash
uv python install 3.12
uv sync --all-extras
```

Developer checks:
- `uv run ruff check .`
- `uv run mypy src`
- `uv run pytest`
- `uv run bash scripts/checks/run_fidelity_ci_subset.sh`
- `uv run python scripts/checks/compliance_report.py --config configs/pilot.yaml --output eval/compliance_report.json`

## CLI
The package ships with `nl` for portable workflows across local/dev/prod environments.

```bash
# runtime compatibility snapshot
uv run nl doctor --json

# architecture/config smoke on chosen device
uv run nl smoke --config-name pilot_smoke --device cpu --batch-size 1 --seq-len 8

# static fidelity checks for a config
uv run nl audit --config-name pilot_paper_faithful

# train with Hydra overrides
uv run nl train --config-name pilot --override train.device=cuda:1 --override train.steps=100
```

`python -m nested_learning ...` is also supported.

## First 30 Minutes
Use this path for a fast first success on CPU:

```bash
uv sync --all-extras
uv run bash scripts/data/run_sample.sh
uv run bash scripts/run_smoke.sh pilot
uv run bash scripts/run_mechanism_audit_smoke.sh
```

This confirms:
- data/tokenizer pipeline is operational,
- model/training loop runs end-to-end,
- cadence checks pass for a mechanism-auditing smoke run.

## Data Pipeline
1. **Tokenizer training**
   ```bash
   uv run python scripts/data/train_tokenizer.py \
     --manifest configs/data/refinedweb_mixture.yaml \
     --vocab-size 32000 \
     --output-dir artifacts/tokenizer/refinedweb_mix \
     --log-file data/mixtures/refinedweb_mix_tokenizer.json
   ```
2. **Corpus filtering + sharding**
   ```bash
   uv run python scripts/data/process_mixture.py \
     configs/data/refinedweb_mixture_filtered.yaml \
     --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
     --log-file data/mixtures/refinedweb_mix_filtered_shards.json
   ```
3. **Sample pipeline** (downloads/licensed datasets, filters, shards, records stats)
   ```bash
   uv run bash scripts/data/run_sample.sh
   ```
4. **Full pipeline** (set env vars like `RW_LIMIT`, `WIKI_LIMIT`, etc. to scale ingestion)
  ```bash
  uv run bash scripts/data/run_full.sh  # default ~50k docs per corpus; increase limits as needed
  ```

### Data Troubleshooting
- If `scripts/data/run_sample.sh` cannot find `artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model`, rerun:
  ```bash
  uv run bash scripts/data/run_sample.sh
  ```
  The script auto-trains the tokenizer when missing.
- If `scripts/data/run_full.sh` fails with `Bad split: train. Available splits: ['test']`, use split fallback:
  ```bash
  FALLBACK_SPLIT=test uv run bash scripts/data/run_full.sh
  ```
  You can also override per-corpus splits (for example `RW_SPLIT=test`).

## Training
- Single GPU / CPU:
  ```bash
  uv run nl train --config-name pilot_smoke
  ```
- Apple Silicon (MPS, if available):
  ```bash
  uv run nl train --config-name pilot_smoke --override train.device=mps
  ```
- Script-based entrypoint (legacy-compatible):
  ```bash
  uv run python train.py --config-name pilot_smoke
  ```
- DDP (torchrun):
  ```bash
  torchrun --nproc_per_node=2 train_dist.py --config-name mid
  ```
- CPU-only DDP smoke (verifies `gloo` backend and deterministic seeding):
  ```bash
  uv run bash scripts/run_cpu_ddp_smoke.sh
  ```
- FSDP (see `docs/FSDP_SCALING_GUIDE.md` for VRAM/batch sizing):
  ```bash
  # 760M run
  torchrun --nproc_per_node=2 train_fsdp.py --config-name hope/mid_fsdp
  # 1.3B run
  torchrun --nproc_per_node=2 train_fsdp.py --config-name hope/target_fsdp
  ```
- DeepSpeed (requires `deepspeed` installed separately):
  ```bash
  deepspeed --num_gpus=2 train_deepspeed.py --config-name target \
    deepspeed.config=configs/deepspeed/zero3.json
  ```

### Mechanism-auditing presets (HOPE / Nested Learning)

Use the mechanism-auditing preset configs (single GPU):

```bash
uv run python train.py --config-name pilot_paper_faithful
# HOPE self-mod variant:
uv run python train.py --config-name pilot_selfmod_paper_faithful
```

Notes:
- These presets set `data.batch_size=1` to avoid cross-sample fast-memory sharing.
- Online chunking supports one-token overlap **or** explicit boundary-target mode (`train.online_boundary_targets=true`).
- Optional attention-state carry across chunks is available in training via `train.online_carry_attention_cache=true`.
- The exact sequence/segment/chunk/buffer semantics are documented in `docs/STREAMING_CONTRACT.md`.

Overrides:
- `optim.type=m3` (paper optimizer option)
- `train.steps=...` / `train.device=...`

See `docs/PAPER_COMPLIANCE.md` for full fidelity notes.
See `docs/STREAMING_CONTRACT.md` for the precise streaming/update contract used by this repo.

## Scope Boundaries (Current)
- This repo targets mechanism-auditing fidelity, not full paper-scale results parity.
- Boundary-state gradient-through-write exists as an experimental constrained path; it is not yet treated as production/full-scale paper reproduction.
- Distributed mechanism-auditing path for boundary-target + attention-cache carry is not implemented.

### Pilot (3 B tokens) workflow
1. Ensure TMUX session:
   ```bash
   tmux new -s pilot_train
   ```
2. Launch the long run on `cuda:1` (≈52 h wall clock):
   ```bash
   set -a && source git.env && set +a
   export UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy
   uv run python train.py --config-name pilot \
     logging.enabled=true logging.backend=wandb \
     logging.project=nested-learning logging.run_name=pilot-main-$(date +%Y%m%d%H%M%S) \
     train.device=cuda:1
   ```
3. Checkpoints appear in `artifacts/checkpoints/pilot/step_*.pt` every 1 000 steps; the accompanying W&B run captures full telemetry.
4. Copy the final checkpoint, config, logs, and eval JSON/CSV into `artifacts/pilot_release/` for distribution.

## Logging
Set `logging.enabled=true` in Hydra configs (or override via CLI) to send metrics to W&B (default). For local JSON logs, use `logging.backend=json logging.path=logs/run.json`. Sample outputs reside in `logs/` and `artifacts/examples/`.

## Evaluation
- Zero-shot:
  ```bash
  uv run python scripts/eval/zeroshot.py \
  --config configs/hope/mid.yaml \
  --checkpoint checkpoints/mid/step_000100.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --tasks all --max-samples 200 --device cuda:0
  ```
  Use `uv run python scripts/eval/zeroshot.py --list-tasks` to display the full benchmark roster (PIQA, HellaSwag, WinoGrande, ARC-E/C, BoolQ, SIQA, CommonsenseQA, OpenBookQA). See `docs/zeroshot_eval.md` for details.
- Needle-in-a-Haystack:
  ```bash
  uv run python scripts/eval/niah.py \
    --config configs/hope/mid.yaml \
    --checkpoint checkpoints/mid/step_000100.pt \
    --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
    --context-lengths 2048 4096 8192 --samples-per-length 20
  ```
- Continual-learning forgetting:
  ```bash
  uv run python scripts/eval/continual.py \
    --config configs/hope/mid.yaml \
    --checkpoints checkpoints/mid/step_000050.pt checkpoints/mid/step_000100.pt \
    --segments-yaml configs/data/continual_segments_sample.yaml \
    --batch-size 4 --max-batches 10 --memorize --memorize-steps 2
  ```
  Plot forgetting curves via `uv run python scripts/eval/plot_forgetting.py --continual-json eval/continual_mid.json`.
- Long-context diagnostics:
  ```bash
  uv run python scripts/eval/passkey.py --config configs/hope/pilot.yaml --checkpoint artifacts/checkpoints/pilot/step_230000.pt \
    --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model --samples 64 --memorize

  uv run python scripts/eval/pg19_perplexity.py --config configs/hope/pilot.yaml --checkpoint artifacts/checkpoints/pilot/step_230000.pt \
    --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model --max-samples 64
  ```

Evaluation summaries are written to `eval/` alongside per-task JSON metrics.

### Test-time memorization toggles
Every evaluator supports TITAN-style memorization so you can reproduce test-time adaptation:
```bash
uv run python scripts/eval/zeroshot.py \
  ... \
  --memorize \
  --memorize-steps 2 \
  --memorize-use-correct-answer \
  --memorize-no-reset  # optional: retain updates across samples
  --memorize-paths titan,cms_fast \
  --memorize-surprise-threshold 0.01
```
- `--memorize` turns on the learner with one LMS step per example by default.
- `--memorize-steps` controls the number of adaptation passes per prompt.
- `--memorize-use-correct-answer` injects ground-truth text during memorization for ablations.
- `--memorize-no-reset` carries memories across samples; omit it to reset every question.
- `--memorize-paths` restricts which levels receive teach-signal updates (`titan`, `cms_fast`, or `all`).
- `--memorize-surprise-threshold` gates updates on average teach-signal norm, matching the paper’s surprise trigger.

Memorization metrics (baseline vs adaptive) are emitted alongside task accuracy for easy comparisons.

## Architecture variants
Select the paper-defined variant via `model.block_variant` in Hydra configs:
- `hope_attention` (paper HOPE-Attention): `Attention → CMS` (paper-defined).
- `hope_selfmod` (paper HOPE scaffold): `Self-modifying Titans (Eqs. 83–93; Eq. 91 residual MLP memories) → CMS` with (by default) **fixed q** and **local conv window=4**, plus chunked updates via `model.self_mod_chunk_size` (others) and `model.self_mod_chunk_size_memory` (M_memory). See `docs/PAPER_COMPLIANCE.md` for the “differentiable read / update-pass writes” semantics.
- `hope_hybrid` (legacy): `Attention + TitanMemory + CMS` (exploratory; not paper-defined).
- `transformer` (baseline): `Attention → MLP` (no TITAN/CMS learning updates; useful for Phase 2 comparisons).

Self-modifying Titans knobs (ablation-friendly, paper-aligned):
- `model.self_mod_objective` (`l2` vs `dot`), `model.self_mod_use_rank1_precond` (DGD-like preconditioner), `model.self_mod_use_alpha` (weight-decay/retention gate), `model.self_mod_stopgrad_vhat`, `model.self_mod_momentum`, `model.self_mod_adaptive_q`, `model.self_mod_local_conv_window`.

## Fast state (Nested Learning semantics)
In-context updates can run against a per-context fast state so meta parameters never change:
- `HOPEModel.init_fast_state()` / `TitanOnlyModel.init_fast_state()` returns a `ModelFastState`.
- `MemorizeConfig.use_fast_state=true` (default) requires passing `fast_state` into `memorize_tokens()` / `memorize_sequence()`; evaluation scripts handle this automatically.
- Training can also run update passes against a per-batch fast state via `train.use_fast_state=true` (meta+delta fast state: meta params are learnable; online updates write deltas only). If `data.batch_size>1`, CMS/TITAN fast state is shared across the batch; use `data.batch_size=1` for strict per-context semantics. See `docs/PAPER_COMPLIANCE.md`.

## Releases
Before tagging or announcing a new checkpoint, work through:
- `docs/release_checklist.md` (model/eval artifact release bundle)
- `docs/PACKAGE_RELEASE_CHECKLIST.md` (package/GitHub/PyPI release flow)
- `docs/PYPI_TRUSTED_PUBLISHING.md` (one-time OIDC setup for TestPyPI/PyPI)

For versioning semantics and breaking-change expectations, see `docs/VERSIONING_POLICY.md`.

For reproducibility bug reports, use `docs/BUG_REPORT_CHECKLIST.md`.

## Performance & optimizer options
- **Mixed precision:** enable bf16 autocast via `train.mixed_precision.enabled=true train.mixed_precision.dtype=bf16` (already enabled in pilot/mid/target configs).
- **`torch.compile`:** accelerate attention/core loops by toggling `train.compile.enable=true train.compile.mode=max-autotune`; failure falls back to eager unless `train.compile.strict=true`.
- **Muon hybrid (default):** all HOPE configs now set `optim.type=muon`, routing ≥2D tensors through PyTorch 2.9's Muon optimizer while embeddings/norms stay on AdamW. Training logs emit `optim.muon_param_elems` / `optim.adamw_param_elems` so you can confirm the split.
- **Fused AdamW fallback:** override with `optim.type=adamw optim.fused=auto` if Muon is unavailable or if you want to compare against the AdamW ablation in `reports/ablations.md`.
- **Surprise gating:** set `model.surprise_threshold=<float>` to gate all inner updates. By default the surprise metric is the average L2 norm of the (scaled/clipped) teach signal (`model.surprise_metric=l2`); you can also use `loss` or `logit_entropy` for ablations. Evaluation CLIs expose `--memorize-surprise-threshold` for ad-hoc gating.

All Hydra knobs can be overridden from the CLI or composed via config groups (`configs/hope/*.yaml`). Use these flags in tandem with `scripts/run_e2e_smoke.sh` (automation) or `scripts/run_cpu_ddp_smoke.sh` (CPU-only determinism check) to validate releases quickly.

## Documentation & References
- `docs/IMPLEMENTATION_STATUS.md` – current mechanism-level status matrix.
- `docs/PAPER_COMPLIANCE.md` – equation-to-code fidelity notes and explicit boundaries.
- `docs/STREAMING_CONTRACT.md` – exact sequence/segment/chunk/update semantics.
- `docs/release_checklist.md` – release readiness checklist.
- `docs/data_pipeline.md` – large-scale sharding/tokenizer workflow.
- `docs/scaling_guidance.md` – roadmap for expanding data + compute footprints.
- `docs/stage2_plan.md` – Stage 2 architecture + experiment roadmap.
- `docs/PHASE_2_PLAN.md` – detailed Phase 2 execution plan.
- `docs/stage2_progress.md` – progress tracker for the latest faithfulness remediation sprint.
- `docs/experiments_report.md` – draft paper covering completed experiments.
- `docs/future_directions.md` – prioritized roadmap after the initial release.
- `reports/stage2_smoke.md` – exact commands/artifacts for the release-ready smoke workflow.
- `docs/FSDP_SCALING_GUIDE.md` – dual-RTX 6000 Ada instructions for the mid/target FSDP configs.
- `google_papers/` – PDFs/markdown of Nested Learning & TITAN papers.
- `CHANGELOG.md` – user-facing changes per release.

## Contributing
1. Run formatting/tests (`uv run ruff check .`, `uv run pytest`).
2. Document new configs or scripts in the relevant docs under `docs/` and update `CHANGELOG.md`.
3. Open a PR referencing the relevant NL/TITAN spec sections and tests.
