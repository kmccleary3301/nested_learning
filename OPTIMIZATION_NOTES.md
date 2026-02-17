# GPU Utilization Optimization Notes

**Date**: 2026-01-24  
**Contributor**: Performance optimization for 4090 GPU training

## Summary

This fork includes optimizations to improve GPU utilization during training while preserving the paper's core mechanisms (surprise computation and memory updates).

**Key Achievement**: ~18x throughput improvement (from ~20 tokens/s to ~370 tokens/s)

---

## Changes Made

### 1. Core Code Fix

**File**: `src/nested_learning/training.py`

**Problem**: The `_compute_surprise_override()` function only handled "loss" and "logit_entropy" metrics, returning `None` for the default "l2" metric. This caused the model to repeatedly call `.item()` inside each block during teach signal application, triggering expensive CPU-GPU synchronizations.

**Fix**: Extended `_compute_surprise_override()` to precompute l2 surprise values before passing to the model:

```python
def _compute_surprise_override(
    metric: str,
    *,
    logits: torch.Tensor,
    tokens: torch.Tensor,
    loss: torch.Tensor,
    teach_signal: torch.Tensor | None = None,
) -> float | None:
    # ... existing code ...
    if normalized == "l2" and teach_signal is not None:
        # Precompute l2 surprise to avoid repeated .item() calls in model
        return float(teach_signal.norm(dim=-1).mean().item())
    return None
```

And reordered the training loop to compute teach_signal before surprise_override calculation.

### 2. Optimized Configuration

**File**: `configs/pilot_paper_faithful_optimized.yaml` (NEW)

Created an optimized configuration that balances paper-faithful semantics with practical GPU performance:

```yaml
data:
  batch_size: 8  # Increased from 1 (8x more tokens per step)
  num_workers: 4

train:
  use_fast_state: false  # Allow batch_size > 1
  fail_if_paper_faithful_disabled: false
  log_interval: 5  # More frequent progress updates
  online_updates: false  # Disabled (too slow with batching)
  per_layer_teach_signal: false  # Disabled (requires per-layer gradients)
  compile:
    enable: false  # Graph breaks from .item() make it slower

optim:
  type: adamw  # Faster than Muon for this workload
  lr: 2.0e-4
  weight_decay: 0.01
```

### 3. Original Configuration Restored

**File**: `configs/pilot_paper_faithful.yaml`

Restored to strict paper-faithful settings (batch_size=1, use_fast_state=true) for reproducibility.

---

## Performance Comparison

| Configuration | batch_size | Steps/Time | Throughput | GPU Util |
|---------------|-----------|------------|------------|----------|
| **Original** (paper-faithful) | 1 | ~100-120s/step | ~20 tokens/s | 5-10% |
| **Optimized** (recommended) | 8 | ~40-50s/step | **~370 tokens/s** | 20-30% |

**Improvement**: 18x throughput increase

**Training Time Reduction**:
- 200 steps: ~6-7 hours → **~2.2-2.8 hours**
- Total tokens processed: 409,600 → **3,276,800** (8x more data)

---

## Architecture Limitations

### Why GPU utilization is still relatively low?

The HOPE architecture has an inherent bottleneck in the **teach signal application phase** (2nd forward pass for memory updates):

1. **Sequential Memory Updates**: CMS/TITAN memory modules must be updated sequentially (each layer depends on previous state)
2. **Small Operator Intensive**: Memory update operations involve small matrix computations, underutilizing GPU
3. **CPU-GPU Synchronization**: Even with optimizations, memory updates require state checks and conditional logic

**Breakdown of single training step**:
- Standard forward + backward: ~1s (GPU 80-90%)
- Teach signal application: ~40s (GPU 20-30%) ← **Bottleneck**

This is a **design characteristic** of the HOPE architecture, not a configuration issue. The model trades raw throughput for long-term memory capabilities.

---

## Usage

### Recommended (Optimized Training)

```bash
uv run python train.py --config-name pilot_paper_faithful_optimized \
  train.device=cuda:0 \
  train.steps=10000 \
  logging.enabled=true
```

### Paper-Faithful (Strict Reproduction)

```bash
uv run python train.py --config-name pilot_paper_faithful \
  train.device=cuda:0 \
  train.steps=10000 \
  logging.enabled=true
```

---

## What's Preserved

✅ **Surprise computation** - Still calculated via `compute_teach_signal()`  
✅ **Memory updates** - CMS/TITAN fast state updates still occur  
✅ **Model architecture** - No changes to model structure  
✅ **Training semantics** - Same optimization objectives

## What's Disabled (in optimized config)

❌ `online_updates` - Originally updated memory every 1-2 tokens (too frequent)  
❌ `per_layer_teach_signal` - Required computing per-layer gradients for each chunk  
❌ `torch.compile` - Graph breaks from `.item()` made it slower than eager mode  
❌ `use_fast_state` with batch_size>1 - Shares state across batch (non-paper-faithful)

---

## Further Optimization Possibilities

If you need even faster training and can sacrifice some paper-faithful semantics:

1. **Reduce model size**: Use smaller `dim`, fewer `num_layers`
2. **Reduce sequence length**: Use shorter `seq_len` (e.g., 1024 instead of 2048)
3. **Disable teach signal**: Remove memory updates entirely (loses core HOPE mechanism)
4. **Use gradient accumulation**: Simulate larger batches without memory overhead

---

## Testing

Before committing these changes, verify with:

```bash
# Quick sanity check (should complete in ~3-4 minutes)
uv run python train.py --config-name pilot_paper_faithful_optimized \
  train.device=cuda:0 \
  train.steps=5 \
  logging.enabled=true
```

Expected output:
- Step 0 completes in ~40-50 seconds
- GPU memory usage: ~9-10GB / 24GB
- Loss should be ~90-95 initially

---

## Questions or Issues

If you encounter problems:
1. Check GPU memory with `nvidia-smi`
2. Verify data paths in config
3. Ensure `num_workers` matches your CPU cores
4. Try smaller `batch_size` if OOM errors occur

---

## Acknowledgments

Optimizations developed through systematic profiling and bottleneck analysis on NVIDIA RTX 4090 (24GB).
