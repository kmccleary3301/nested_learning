# Scaling Guidance – Nested Learning Reproduction

This document describes how to extend the current smoke-tested Nested Learning (HOPE) stack to larger datasets, hardware targets, and experiment scopes without changing the core codebase.

---

## 1. Hardware Tiers
| Tier | GPUs | VRAM | Usage |
|------|------|------|-------|
| **Dev / Smoke** | CPU or 1× RTX 6000 Ada | 0–48 GB | Pipeline validation, unit/integration tests |
| **Pilot** | 2× RTX 6000 Ada | 48 GB each | `configs/hope/pilot.yaml`, seq len ≤2K, ≤5 B tokens |
| **Mid** | 4–8× RTX 6000 Ada (or single H200) | 48–80 GB | `configs/hope/mid.yaml`, seq len 4K, 30 B tokens |
| **Target** | Future dual H200 | 141 GB each | `configs/hope/target.yaml`, seq len 8K+, 100 B tokens |

Recommendations:
- Prefer `cuda:1` on dual-GPU workstations to keep `cuda:0` free for interactive workloads.
- Enable `torch.set_float32_matmul_precision("high")` on Ampere+/Ada to benefit from BF16 kernels.
- For >2 GPUs, switch to `train_dist.py` (DDP) or `train_fsdp.py` with `train.fsdp.auto_wrap_min_params` tuned to 2 M for HOPE blocks.

---

## 2. Storage & Data Layout
| Corpus Slice | Sample (current) | Full target | Disk (approx) |
|--------------|------------------|-------------|---------------|
| RefinedWeb / FineWeb proxy | 2k docs → 4 shards | 4 B docs | 1.2 TB |
| Wikipedia EN | 1k docs | Full dump | 70 GB |
| C4 EN | 1k docs | 400 M docs | 300 GB |
| SlimPajama | 1k docs | 600 B tokens | 450 GB |
| CodeParrot clean | 1k files | 50 B tokens | 200 GB |

Scaling procedure:
1. **Raw ingestion:** Stage compressed corpora under `data/raw/` (ensure ≥3 TB free for target runs).
2. **Filtering:** Drive `scripts/data/run_full.sh` with env vars (e.g., `RW_LIMIT=1000000 WIKI_LIMIT=250000`) to control per-corpus document counts. The script wraps `filter_corpus.py` + `process_mixture.py` + tokenizer training for the `configs/data/refinedweb_mixture_full.yaml` manifest. Keep `--force-exit` to make failures loud.
3. **Tokenizer retrain:** Rerun `scripts/data/train_tokenizer.py` with combined manifest once filtered corpora exceed ~100 M tokens to avoid domain skew. Store models under `artifacts/tokenizer/<mixture_name>/`.
4. **Sharding:** Update `configs/data/refinedweb_mixture_filtered.yaml` with new `max_records` and `sequence_length` values (e.g., 2048 for pilot, 4096 for mid, 8192 for target). Run `scripts/data/process_mixture.py` pointing to new tokenizer and filtered text.
5. **Stats:** Version log files in `data/mixtures/` (`*_shards.json`, `*_tokenizer.json`) for each scale; include total tokens, sequences, and shards to keep reproducibility audit trail.

---

## 3. Training Scale-Up
1. **Pilot (≤160 M params):** Use `train.py --config-name hope/pilot` on dual RTX 6000. Set `train.device=cuda:1`, `train.checkpoint.enable=true`, `train.checkpoint.save_interval=500`. Expect ~3 mins per 100 steps on synthetic data.
2. **Mid (≈760 M params):** Launch via `torchrun --nproc_per_node=2 train_dist.py --config-name hope/mid train.device=cuda --train.steps=5000`. Feed mixture shards from `data/shards/*_filtered`. Enable gradient checkpointing if activations exceed memory (see `model.gradient_checkpointing` flag in config).
3. **Target (≥1.3 B params):** Prefer DeepSpeed or FSDP. Example:
   ```bash
   torchrun --nproc_per_node=2 train_fsdp.py --config-name hope/target \
     train.fsdp.auto_wrap_min_params=4000000 \
     train.checkpoint.enable=true train.checkpoint.dir=checkpoints/target
   ```
   When H200 cluster arrives, increase `train.data.seq_len` to 8192 and switch attention backend to FlashAttention (set `model.attn.impl=flash` in config).
4. **Logging:** Route to W&B for long runs (`logging.backend=wandb`). For on-prem, keep JSON logs under `logs/<run>.json` and ship to artifact storage after each training window.
5. **Optimizer tuning:** Stage 2 experiments call for Muon/DeepMomentum variants. Define new entries under `model.optimizers` keyed by level names; adjust `lr`, `beta` per level clock frequency.

---

## 4. Evaluation Expectations
| Stage | Checkpoints | Eval commands |
|-------|-------------|---------------|
| Smoke | `artifacts/checkpoints/pilot_smoke/step_000010.pt` | `scripts/eval/zeroshot.py` with `--max-samples 32`, NIAH `--samples-per-length 3`, continual sample segments |
| Pilot | Every 1k steps | `--tasks all`, `niah` with `context-lengths 4k 8k`, continual segments covering refinedweb/wikipedia/code |
| Mid/Target | Every 1k steps (rotated) | Full tasks + ARC-C, BoolQ, SIQA entire validation, NIAH up to 32k tokens, continual across ≥5 segments |

Archive outputs under `eval/<run>/<timestamp>.json` to make comparisons easy. Use the provided `eval/zeroshot_full_smoke.json`, `eval/niah_smoke.json`, and `eval/continual_smoke.json` as formatting references.

---

## 5. Roadmap to H200 Cluster
1. **Data:** Mirror filtered corpora to shared storage; ensure tokenizer + shard manifests are versioned with commit SHAs.
2. **Compute:** Port launcher scripts to Slurm or Kubernetes. Provide templates under `scripts/infra/` (TODO) with environment exports (`MASTER_ADDR`, `MASTER_PORT`).
3. **Long-context:** Integrate block-sparse or state-space attention when sequence lengths exceed 32k tokens. Keep `context_lengths` in config and Hydra overrides to toggle.
4. **Reliability:** Add resumption guidelines in `docs/release_plan.md` (FSDP resume path). Store checkpoints in object storage with lifecycle policies.

This document should be updated whenever new corpora, hardware, or launch scripts are introduced so contributors can quickly understand how to move beyond the smoke-tested baseline.
