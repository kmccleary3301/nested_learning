# FSDP/ZeRO Scaling Guide (RTX 6000 Ada Dual-GPU Rig)

This note captures the configuration we will use for the Stage 2 mid (≈760 M) and target (≈1.3 B) HOPE models when running on the dual RTX 6000 Ada workstation (2× 48 GB). It accompanies the new Hydra configs `configs/hope/mid_fsdp.yaml` and `configs/hope/target_fsdp.yaml`.

## Hardware & Software Assumptions
- 2× NVIDIA RTX 6000 Ada (48 GB each)
- CUDA 12.4, PyTorch 2.9, `uv` environment
- NCCL backend, FSDP via `torch.distributed.fsdp`
- Checkpoints stored under `artifacts/checkpoints/{mid_fsdp,target_fsdp}`

## Config summary

| Model | Params | Config | Per-rank micro-batch | Global batch (nranks=2) | Expected VRAM | Notes |
|-------|--------|--------|----------------------|-------------------------|---------------|-------|
| HOPE mid | ~760 M (dim 1024, 24L) | `configs/hope/mid_fsdp.yaml` | 8 sequences × 2048 tokens | 16×2048 tokens | 43–45 GB | bf16 activations, Muon outer optimizer, NL inner optimizer, gradient checkpointing, FSDP auto-wrap ≥2 M params |
| HOPE target | ~1.3 B (dim 1536, 32L) | `configs/hope/target_fsdp.yaml` | 4 sequences × 2048 tokens | 8×2048 tokens | 46–48 GB | Slightly smaller per-rank batch to stay under 48 GB; Muon + checkpointing identical to mid config |

Both configs default to:
- `optim.type = muon` (outer optimizer) with `nl_l2_precond` inner updates already wired through model lvl optimizers.
- bf16 autocast (`train.mixed_precision.enabled = true, dtype = bf16`).
- Gradient checkpointing via `model.gradient_checkpointing = true` (saves ~3 GB per rank).
- `train.compile.enable = false` (Torch.compile can be toggled on after validation).
- FSDP auto-wrap policy set via `train.fsdp.auto_wrap_min_params`.

## Launch commands

```bash
# Mid model, 2 GPUs
UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy \
uv run torchrun --nproc_per_node=2 train_fsdp.py \
  --config-name hope/mid_fsdp logging.run_name=mid-fsdp-${USER}

# Target model, 2 GPUs
UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy \
uv run torchrun --nproc_per_node=2 train_fsdp.py \
  --config-name hope/target_fsdp logging.run_name=target-fsdp-${USER}
```

To resume from a checkpoint, set `train.checkpoint.resume_path` (path to `step_xxxxxx.pt`). State dicts use FSDP’s full-state sharding with CPU offload for rank 0.

## ZeRO / DeepSpeed note

For multi-node runs or larger batch sizes, leverage `train_deepspeed.py` with `configs/deepspeed/zero3.json`. The per-model configs above can be reused by passing `--config-name hope/mid_fsdp` together with `DEEPSPEED_CONFIG=configs/deepspeed/zero3.json`.

## Logging & Monitoring

- JSON metrics live at `logs/mid_fsdp_metrics.json` or `logs/target_fsdp_metrics.json`.
- W&B logging is enabled by default (`project = nested-learning`).
- Additional telemetry (teach-signal norms, projector stats, CMS chunk samples) already flows through the model update metrics; ensure your W&B dashboard visualizes:
  - `layer*.titan.titan.grad_norm`
  - `layer*.titan.titan.ctx_norm` / `proj_norm`
  - `layer*.cms.cms_fast.chunk_samples`, etc.

## Checklist before starting a long run
1. `uv run pytest` (ensure faithfulness tests pass).
2. `nvidia-smi` — GPUs idle and temps normal.
3. Confirm dataset shards (`data/shards/*_full/`) available locally.
4. W&B credentials set (`WANDB_API_KEY`).
5. For target config, consider setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.

This guide should let collaborators pick up the FSDP configs immediately without reverse-engineering the Hydra hierarchy.
