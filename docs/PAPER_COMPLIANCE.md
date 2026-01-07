# Paper Compliance Notes (HOPE / Nested Learning)

This document summarizes which fidelity‑critical behaviors are implemented and how to enable them.

## Must‑Use Flags for Paper‑Faithful Runs

### Online CMS updates (read‑after‑write)
- **Config fields**:
  - `model.*.cms_online_updates: true` (default in block configs)
  - `train.online_updates: true`
  - `train.online_chunk_size: 0` (auto‑infer min CMS update period)
- **Effect**: CMS updates apply between chunks, so later tokens see updated CMS parameters.

### Per‑layer local error signals (δℓ)
- **Config field**: `train.per_layer_teach_signal: true`
- **Effect**: uses autograd to compute per‑block teach signals and routes them into each block’s update path.

### Self‑modifying Titans always‑on
- **Config field**: `model.*.selfmod_online_updates: true` (default in `HOPESelfModBlockConfig`)
- **Effect**: self‑modifying Titans updates run even when no external teach signal is provided.

### CMS chunk accumulation semantics
- **Config field**: `model.*.cms_chunk_reduction: "sum"` (default)
- **Effect**: per‑token δ contributions are **summed** across the chunk (Eq. 71‑style accumulation).

### M3 optimizer (paper option)
- **Config field**: `optim.type: "m3"`
- **Notes**: M3 is implemented per Algorithm 1 (multi‑scale momentum + Newton‑Schulz orthogonalization).

## Implementation Notes / Divergence Controls

- **Partial chunk handling**: online updates trigger only when a full update period is reached; remaining tail tokens do not trigger an update (matches “update every C steps” interpretation).
- **Chunking vs update_period**: `train.online_chunk_size` controls how many tokens are processed before an online update pass; CMS levels still update on their own `update_period` boundaries within that stream.
- **Surprise gating**: CMS updates honor `model.surprise_threshold`; self‑mod updates are always‑on unless explicitly disabled.
- **Non‑paper additions**: Muon/AdamW hybrid remains available; use `optim.type: "m3"` for paper‑faithful runs.

## Tests Covering Fidelity

- `tests/test_cms.py`:
  - update‑period gating behavior
  - online updates altering later‑token outputs
- `tests/test_teach_signal.py`:
  - teach signal matches autograd gradient
  - per‑layer teach signal shapes
- `tests/test_selfmod_online.py`:
  - self‑mod updates occur without external teach signal
- `tests/test_m3.py`:
  - M3 performs updates and slow‑momentum activation
