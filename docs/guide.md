# Nested Learning Reproduction Guide

This guide is a self-contained reference for reproducing Google's Nested Learning (HOPE) architecture within this repository. It captures the expectations set by the Nested Learning and TITAN papers, plus the quality bar demonstrated by lucidrains’ TITAN implementation. Follow it end-to-end to prepare the environment, process data, train smoke models, and run the evaluation suite.

---

## 1. Objectives & Scope
- **Goal:** Provide a faithful, open-source reproduction of HOPE blocks (attention → TITAN memory → CMS self-modifier) ready for community scaling.
- **Hardware targets:** Works on CPU for smoke tests; optimized for dual RTX 6000 Ada. Larger configs document how to extend to future H200 nodes.
- **Outputs:** Clean PyTorch 2.9 codebase, reproducible data + training workflows, documented evaluation harness, and guidance for future releases.

---

## 2. Environment & Tooling
| Requirement | Notes |
|-------------|-------|
| Python | 3.12 (managed via `uv python install 3.12`) |
| Package manager | `uv` 0.9+; isolates dependencies per project |
| Framework | PyTorch **2.9.0** + `torchvision 0.24.0`, `torchaudio 2.9.0` |
| Extras | `einops`, `hydra-core`, `datasets`, `sentencepiece`, `wandb`, `langdetect` |

**Setup commands**
```bash
uv python install 3.12
uv sync --all-extras
```

Sanity check:
```bash
uv run python -c "import torch; print(torch.__version__)"
```

Optional dev tooling:
```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

---

## 3. Repository Layout
| Path | Purpose |
|------|---------|
| `src/nested_learning/` | HOPE block, TITAN memory, CMS, deep optimizers, scheduler utilities |
| `configs/hope/*.yaml` | Hydra configs for pilot/mid/target scales plus smoke variants |
| `scripts/data/` | Tokenizer training, corpus filtering, mixture processing, sample shortcuts |
| `scripts/run_smoke.sh` | CPU-friendly pilot/mid smoke training entry point |
| `scripts/run_e2e_smoke.sh` | (Added in release workflow) Chains sync → data sample → smoke train → eval |
| `scripts/eval/*.py` | Zero-shot, NIAH, continual-learning evaluators |
| `artifacts/` | Tokenizers, checkpoints, example logs |
| `docs/` | Plans, this guide, release checklist, data pipeline notes |

---

## 4. Quickstart Workflow
1. **Install deps:** `uv sync --all-extras`
2. **Sample data:** `uv run bash scripts/data/run_sample.sh` (downloads + filters RefinedWeb/Wiki/C4/SlimPajama/code samples, shards them, records stats in `data/mixtures/refinedweb_mix_filtered_shards.json`).
3. **Smoke training:** `uv run bash scripts/run_smoke.sh pilot` (runs CPU pilot config, saves checkpoints to `artifacts/checkpoints/pilot_smoke/`).
4. **Zero-shot sanity:** `uv run python scripts/eval/zeroshot.py --tasks piqa --max-samples 32 --checkpoint artifacts/examples/pilot_dummy.pt --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model --config configs/hope/pilot.yaml --device cpu`
5. **Full automation (optional):** `uv run bash scripts/run_e2e_smoke.sh` once created; it orchestrates steps 1–4, checks logs under `logs/`.

All commands default to CPU; pass `--device cuda:0` (or `cuda:1`) to leverage GPUs.

---

## 5. Data Pipeline
1. **Tokenizer training:** `scripts/data/train_tokenizer.py` consumes the manifest in `configs/data/refinedweb_mixture.yaml`. Outputs go to `artifacts/tokenizer/refinedweb_mix/`.
2. **Filtering:** `scripts/data/filter_corpus.py` enforces language/length/dedup constraints. Sample shortcuts live in `scripts/data/run_sample.sh`; large-scale filtering + sharding is scripted via `scripts/data/run_full.sh` (override env vars like `RW_LIMIT`, `WIKI_LIMIT`, `CODE_LIMIT` or point the script at pre-downloaded corpora before launching it in tmux).
3. **Sharding:** `scripts/data/process_mixture.py configs/data/refinedweb_mixture_filtered.yaml --tokenizer-path <model> --log-file data/mixtures/refinedweb_mix_filtered_shards.json` converts cleaned text into token shards under `data/shards/<dataset>/`.
4. **Sample stats:** JSON logs under `data/mixtures/` summarize sequence counts, token throughput, and shard sizes for reproducibility.

Full-scale runs: use `scripts/data/run_full.sh` to generate the `_full` shards referenced by `configs/hope/mid.yaml` / `configs/mid_stage2.yaml`. If you only need the filtered sample shards, override the shard paths on the CLI, e.g.:
```bash
uv run torchrun --nproc_per_node=2 train_dist.py --config-name mid_stage2 \
  data.mixture.sources[0].shards_dir=data/shards/refinedweb_filtered \
  data.mixture.sources[1].shards_dir=data/shards/wikipedia_filtered \
  ...
```
Document disk needs (≈2 TB for 100 B tokens) before launching large jobs.

---

## 6. Training
- **Hydra configs:** `configs/hope/pilot.yaml`, `mid.yaml`, `target.yaml` define HOPE block depth, level clocks, optimizer stacks. Smoke configs (`pilot_smoke.yaml`, `mid_smoke.yaml`) limit steps and swap in synthetic/tokenized samples.
- **Entry points:**
  - Single GPU/CPU: `uv run python train.py --config-name pilot_smoke`
  - DDP: `torchrun --nproc_per_node=2 train_dist.py --config-name mid`
  - FSDP: `torchrun --nproc_per_node=2 train_fsdp.py --config-name mid`
  - DeepSpeed: `deepspeed --num_gpus=2 train_deepspeed.py --config-name target deepspeed.config=configs/deepspeed/zero3.json`
- **Logging:** Set `logging.backend=json logging.path=logs/<run>.json` to write structured metrics (level firings, CMS norms, self-mod deltas). W&B is supported by flipping `logging.backend=wandb`.
- **Artifacts:** Checkpoints land in `artifacts/checkpoints/<run>/step_xxxxxx.pt` with accompanying optimizer + clock states.

---

## 7. Evaluation
| Script | Purpose |
|--------|---------|
| `scripts/eval/zeroshot.py` | PIQA, HellaSwag, WinoGrande, ARC-E/C, BoolQ, SIQA; specify `--tasks all` for full sweep |
| `scripts/eval/niah.py` | Needle-in-a-Haystack / long-context recall up to 32k tokens (extendable) |
| `scripts/eval/continual.py` | Measures forgetting across sequential dataset segments described via YAML |

Example zero-shot run:
```bash
uv run python scripts/eval/zeroshot.py \
  --config configs/hope/mid.yaml \
  --checkpoint artifacts/examples/pilot_dummy.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --tasks piqa,hellaswag,winogrande \
  --device cuda:0 --max-samples 200
```

Evaluation outputs are JSON summaries stored under `eval/<task>/<timestamp>.json` (created automatically).

---

## 8. Testing & QA
- `uv run pytest` exercises level clocks, CMS updates, and optimizer plumbing.
- `uv run ruff check .` enforces style.
- `uv run mypy src` ensures type safety for critical abstractions.
- Integration smoke tests (`scripts/run_smoke.sh`) plus end-to-end automation (`scripts/run_e2e_smoke.sh`) confirm wiring before releases.

---

## 9. Release & Contribution Notes
- Track release readiness via `docs/release_plan.md`.
- `CHANGELOG.md` summarizes user-facing changes per version.
- Planned `docs/release_plan.md` tasks: data convenience script, README consolidation, automation script, GitHub Actions.
- Contributions: open issues describing HOPE feature gaps, attach references to the NL/TITAN papers (copies live under `google_papers/`).

---

## 10. References
- **Nested Learning paper + transcript:** `google_papers/Nested_Learning/`
- **TITANs paper + lucidrains repo:** `google_papers/TITANs/`, `ref_repos/titans-pytorch`
- **Planning transcript:** `docs/planner_convo_01.md`
- **Stage roadmaps:** `docs/stage1_plan.md`, `docs/stage2_plan.md`

Use this guide as the primary onboarding document; it links to deeper plans when extra context is needed.
