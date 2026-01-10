# Paper Compliance / Fidelity Guide (Nested Learning / HOPE)

This doc explains the **fidelity‑critical behaviors** (what the paper relies on) and how they map to this repo’s code, flags, and tests.

It is deliberately **mechanism‑focused**: you can use it to answer “did we implement the architecture/update rules correctly?” without requiring full‑scale training reproduction.

## Scope

**In scope**
- HOPE blocks (attention + CMS + TITAN/self‑mod paths) and the *nested/online* update mechanism.
- Correct teach‑signal alignment (LM head vs embedding), per‑layer local error signals (δℓ), and chunk‑accumulated CMS updates.
- A paper‑style optimizer option (M3) alongside practical defaults.

**Out of scope (today)**
- Full bi‑level meta‑learning experiments over explicit task episodes (outer objective over tasks + inner adaptation per task).
- Results parity at the original paper’s compute scale.

## Semantic contract (important)

This repo focuses on **mechanism-level fidelity** (update rules + dataflow) with explicit tests.

- **Differentiable reads:** the forward pass used to compute the outer LM loss is standard autograd.
- **Stop‑grad writes:** online memory updates are applied in an explicit update pass (typically under `torch.no_grad()`), so we do **not** backprop through online writes and we do **not** implement the paper’s chunk‑parallel boundary‑state gradient training procedure.
- **Meta initialization (fast-state mode):** when `train.use_fast_state=true`, meta parameters are not mutated by online updates, but the *read-path* meta parameters still receive outer gradients:
  - CMS/TITAN fast state uses **meta+delta** (forward uses `meta + delta`; updates write deltas only).
  - HOPE‑SelfMod uses a detached per‑context state, but the read path uses a **straight‑through meta gradient** link so the meta initialization remains trainable.

## Quick start: “paper‑faithful mechanisms” (single GPU)

The most paper‑faithful execution path in this repo is **single‑GPU** `train.py`, because it supports both:
1) **per‑layer δℓ** teach signals and  
2) **online chunked training** where later tokens’ loss/gradients can see earlier memory updates.

Minimal smoke:

```bash
uv run python train.py --config-name pilot_paper_faithful train.steps=5
```

Note: the paper-faithful presets set `data.batch_size=1` to avoid cross-sample memory sharing
when `train.use_fast_state=true`.

Optional: select the paper optimizer variant for the *outer* step:

```bash
uv run python train.py --config-name pilot_paper_faithful train.steps=5 optim.type=m3
```

Paper-faithful HOPE self-mod variant:

```bash
uv run python train.py --config-name pilot_selfmod_paper_faithful train.steps=5
```

## Paper‑Faithful vs Practical Mode (Matrix)

This repo supports both paper‑faithful mechanisms (for auditing correctness) and practical defaults (for running pilots quickly).

| Mechanism | Paper-faithful intent | This repo (single GPU) | Notes / Tests |
|---|---|---|---|
| Teach‑signal alignment | δ uses LM head weights | `compute_teach_signal()` matches autograd grad | `tests/test_teach_signal.py` |
| Per‑layer δℓ | block‑local error signals | `train.per_layer_teach_signal=true` | `tests/test_teach_signal.py` |
| Online chunked training | later tokens can “see” earlier inner updates | `train.online_updates=true` (chunk size clamped to ≥2) | `src/nested_learning/training.py`, `tests/test_cms.py` |
| CMS chunk accumulation | sum over token deltas per chunk | `cms_chunk_reduction="sum"` default | `tests/test_cms.py`, `tests/test_cms_delta_rule.py` |
| CMS partial-chunk flush | update on final partial chunk | `model.cms_flush_partial_at_end=true` | `tests/test_cms_flush_partial.py` |
| CMS LayerNorm | paper is architecture-light; norm is optional | `model.cms_use_layernorm=true` (default) | `tests/test_cms.py` |
| HOPE‑SelfMod local conv | local conv window=4 (paper HOPE module) | `SelfModifyingTitansConfig.local_conv_window=4` default (causal depthwise) | `tests/test_selfmod_local_conv.py` |
| HOPE‑SelfMod fixed q | paper: `q_t = x_t W_q` non‑adaptive | `SelfModifyingTitansConfig.adaptive_q=false` default | `tests/test_selfmod_adaptive_q.py` |
| HOPE‑SelfMod Eq. (91) skip | no projection skip term (`w_skip`) | `model.self_mod_use_skip=false` (paper-faithful presets) | `tests/test_residual_mlp_memory.py` |
| HOPE‑SelfMod read/write separation | differentiable read; stopgrad through writes | forward uses differentiable read; updates occur only in explicit update pass | `tests/test_selfmod_grad_flow.py`, `tests/test_hope_selfmod_update_pass.py` |
| Fast‑state isolation | per‑context inner updates without mutating meta params, while read‑path meta init remains learnable | `train.use_fast_state=true` | CMS/TITAN use **meta+delta**; HOPE‑SelfMod read path uses straight‑through meta gradients. Meta params remain unchanged during updates and still receive outer grads (`tests/test_hope_selfmod_fast_state_meta_unchanged.py`, `tests/test_fast_state_meta_grads.py`, `tests/test_fast_state_selfmod_meta_grads.py`, `tests/test_fast_state_forward_equivalence.py`, `tests/test_fast_state_batch_semantics.py`) |
| Surprise metric | paper “surprise” trigger | `model.surprise_metric=l2` (default); also `loss`, `logit_entropy` | `tests/test_surprise_metric.py`, `tests/test_faithfulness_harness.py` |
| Outer optimizer | M3 option exists | `optim.type=m3` | `tests/test_m3.py` |
| Outer param policy | include memory initial states in meta-update | `optim.param_policy=all` | `tests/test_optimizer_param_policy.py` |
| DDP fail-fast | avoid silent paper-divergent fallbacks | `train.fail_if_paper_faithful_disabled=true` | `tests/test_distributed_fail_fast.py` |
| Multi‑GPU | (not required by paper) | DDP disables `online_updates` + `per_layer_teach_signal`; FSDP uses offline updates | documented below |

## Concepts → implementation mapping

### 1) Outer parameters vs inner (“fast”) procedure

In this codebase:
- **Outer update** = the standard optimizer step (`optimizer.step()`) on the model parameters after backprop.
- **Inner update** = memory/fast updates applied *outside* the gradient graph using teach signals (δ), e.g. CMS updates and self‑modifying TITAN updates.

Where:
- Outer loop: `src/nested_learning/training.py` (`run_training_loop`)
- Inner update calls: inside the training loop after backward:
  - `base_model(tokens, teach_signal=...)` or `base_model(tokens, teach_signals=[...])`
- The update logic lives in the block implementations:
  - `src/nested_learning/hope/block.py`

### 2) “Levels” and update frequencies

Levels are represented explicitly as `LevelSpec` entries with independent `update_period`s.

Where:
- Specs: `src/nested_learning/levels.py`
- Config surface (Hydra): `model.titan_level` and `model.cms_levels` in `configs/*.yaml`
- Enforcement:
  - Online CMS buffering + update‑period gating in `src/nested_learning/hope/block.py`
  - Level optimizer tick/step orchestration in `src/nested_learning/optim/manager.py`

### 3) Teach signal alignment (LM head gradient proxy)

The global teach signal is an approximation to **dL/dh**, where `h` is the hidden state **before** the LM head. This approximation must align to the LM head weights.

Where:
- Weight tying is explicit: `src/nested_learning/model.py` (`self.lm_head.weight = self.embed.weight`)
- Teach signal implementation: `src/nested_learning/training.py` (`compute_teach_signal`)
- Unit coverage: `tests/test_teach_signal.py`

### 4) Per‑layer local error signals (δℓ)

When enabled, we compute a teach signal **per block output** (δℓ) via autograd and route it into each block’s update path.

Where:
- Block output capture: `src/nested_learning/model.py` (`forward_with_block_outputs`)
- δℓ computation: `src/nested_learning/training.py` (`_compute_layer_teach_signals`)
- Routing to blocks: `src/nested_learning/model.py` (`teach_signals=[...]`)
- Unit coverage: `tests/test_teach_signal.py` (shape + matching expectations)

Flag:
- `train.per_layer_teach_signal=true`

### 5) Chunked online training (read‑after‑write for *loss*, not just updates)

This is the core “gradient propagation across frequencies” concern:

If you compute the LM loss on a full sequence **once**, and only apply memory updates after the backward pass, then later tokens’ loss does not reflect earlier inner updates.

To make later tokens “see” earlier inner updates during training, we support an **online chunked training** mode:
- Split the sequence into chunks.
- For each chunk:
  1) forward → loss  
  2) `loss.backward()` **accumulating** gradients across chunks (we do not zero grads per chunk)  
  3) apply inner updates in `torch.no_grad()`  
  4) proceed to the next chunk with updated memory
- At the end, we do a single outer `optimizer.step()`.

Where:
- `src/nested_learning/training.py` (search for `online_updates`)

Flags:
- `train.online_updates=true`
- `train.online_chunk_size=0` (auto‑infer a chunk size from the minimum CMS update period)

### 6) CMS update semantics (per‑token δ + sum‑over‑chunk accumulation)

CMS updates are applied using:
- **per‑token δ targets** (no chunk‑mean broadcast), and
- **sum‑over‑chunk reduction** for the CMS update loss (rather than mean), which preserves the “accumulate over C tokens” semantics.

We implement the CMS local objective via a **gradient-shaping construction**:
- `_chunk_loss()` chooses a target `t = stopgrad(prediction − δ)` so that `∂loss/∂prediction ∝ δ` under the chosen mask and reduction.
- This matches the paper’s δ-based local learning rule while letting us implement the update via standard autograd.
- Verified by `tests/test_cms_delta_rule.py`.

Where:
- Chunk loss reduction: `src/nested_learning/hope/block.py` (`_chunk_loss`, `cms_chunk_reduction="sum"`)
- Online buffering by update_period and “pop exactly C tokens”: `src/nested_learning/hope/block.py` (`_CmsBuffer`, `_pop_buffer_chunk`, `_cms_forward_online`)
- Unit coverage:
  - `tests/test_cms.py` (online updates affect later tokens; update_period gating)

Notes:
- In the Hydra configs, CMS chunk reduction / online toggles are **paper‑faithful defaults** inside the HOPE block configs. They are not currently exposed as top‑level YAML keys; changing them requires a small code change (we can expose them if contributors want to run explicit divergence ablations).
- `model.cms_flush_partial_at_end` is exposed because it affects correctness when sequence lengths are not exact multiples of update periods.

### 7) Self‑modifying TITAN path (always‑on)

Self‑modifying TITAN updates run in the update pass; they do not require the teach signal to be nonzero, but they **do** require an explicit update call (i.e., passing `teach_signal`/`teach_signals` to trigger the update pass).

Where:
- `src/nested_learning/hope/block.py` (self‑mod update path)
- Unit coverage: `tests/test_selfmod_online.py`

### 8) Outer optimizer options (including paper M3)

Default outer optimizer in configs is practical and reproducible (`optim.type=muon` hybrid with AdamW fallback for 1D params).

Paper option:
- `optim.type=m3` selects the M3 optimizer (multi‑scale momentum + Newton‑Schulz orthogonalization).

Where:
- `src/nested_learning/optim/m3.py`
- Unit coverage: `tests/test_m3.py`

## Distributed training caveats (important)

Paper‑faithful mode is currently focused on `train.py` (single‑GPU).

- **DDP (`train_dist.py`)**: calls the shared training loop, but explicitly disables:
  - per‑layer teach signals (`train.per_layer_teach_signal`)
  - online chunked training (`train.online_updates`)
  because these require capturing block outputs and applying sequential inner updates in a way that is not yet DDP‑safe in this repo.
  - If you want to avoid silent fallback behavior, set `train.fail_if_paper_faithful_disabled=true` to raise instead of disabling.

- **FSDP (`train_fsdp.py`)**: currently uses a simpler “offline” update pass with a global teach signal after each outer step. It does not yet implement per‑layer δℓ or online chunked training.

If you need paper‑faithful mechanics at multi‑GPU scale, the next engineering task is to port the `train.py` online/per‑layer flow to FSDP (or a custom DDP scheme) while keeping correctness tests.

## Verification checklist (fast)

Run the fidelity tests:

```bash
uv run pytest \
  tests/test_teach_signal.py \
  tests/test_cms.py \
  tests/test_cms_flush_partial.py \
  tests/test_selfmod_online.py \
  tests/test_m3.py \
  tests/test_residual_mlp_memory.py \
  tests/test_selfmod_local_conv.py \
  tests/test_selfmod_adaptive_q.py \
  tests/test_selfmod_grad_flow.py \
  tests/test_hope_selfmod_update_pass.py \
  tests/test_cms_delta_rule.py \
  tests/test_selfmod_dgd_linear.py \
  tests/test_optimizer_param_policy.py \
  tests/test_distributed_fail_fast.py
```

Confirm you’re running with the intended features:
- training logs include `teach_signal_norm` and per‑layer update telemetry (e.g. `layer0.cms.cms_fast.grad_norm`) when an update pass runs.

## Known gaps / intentionally deferred work

- Full task‑episode meta‑learning evaluation loops are not implemented.
- Multi‑GPU paper‑faithful training (online + per‑layer δℓ) is not yet implemented.
- Large‑scale results reproduction is not a requirement for claiming mechanism fidelity in this repo.
