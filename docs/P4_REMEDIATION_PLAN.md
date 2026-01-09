# P4 Remediation Plan — Status & Tracking (Paper-Faithful HOPE/Nested Learning)

This file started as an execution checklist for the P4 “paper faithfulness” sprint. It is now maintained as a **status page** so contributors can quickly see what’s implemented, what is verified by tests, and what follow‑ups remain.

For the canonical “how to run paper‑faithful mode” guide, see `docs/PAPER_COMPLIANCE.md`.

## Status (core remediation)

**P0/P1 core faithfulness items:** complete.

Implemented behaviors (with pointers):
- **Self‑modifying TITAN path always-on** during the inner/update pass (does not require an external teach signal).  
  Code: `src/nested_learning/hope/block.py`  
  Test: `tests/test_selfmod_online.py`
- **CMS update semantics** use per‑token δ targets and **sum‑over‑chunk** accumulation (no chunk‑mean broadcast).  
  Code: `src/nested_learning/hope/block.py` (`_chunk_loss`, `_CmsBuffer`, `_pop_buffer_chunk`)  
  Test: `tests/test_cms.py`
- **Online CMS read‑after‑write** behavior (later tokens can see updated CMS weights when using the online training path).  
  Code: `src/nested_learning/hope/block.py` (`_cms_forward_online`) + `src/nested_learning/training.py` (`train.online_updates`)  
  Test: `tests/test_cms.py` (`test_cms_online_updates_affect_later_tokens`)
- **Per‑layer local error signals (δℓ)** computed via autograd and routed into each block.  
  Code: `src/nested_learning/model.py` (`forward_with_block_outputs`, `teach_signals`) + `src/nested_learning/training.py` (`_compute_layer_teach_signals`)  
  Test: `tests/test_teach_signal.py`
- **Paper optimizer option (M3)** implemented and selectable via `optim.type=m3`.  
  Code: `src/nested_learning/optim/m3.py`  
  Test: `tests/test_m3.py`

Docs/telemetry added:
- Paper‑faithful run flags + code mapping: `docs/PAPER_COMPLIANCE.md`
- README “paper‑faithful mode” snippet: `README.md`
- Per‑layer update telemetry (e.g. `layerX.cms.*`) emitted via `HOPEModel._gather_block_stats()`.

## Remaining follow-ups (optional hardening, not required for “implemented correctly”)

These are improvements that strengthen the validation story or reduce ambiguity, but they are not required to claim the core mechanism is implemented:

- [ ] Add an explicit unit test that demonstrates **per‑token δ vs chunk‑mean broadcast** leads to different update directions (sanity test on a toy CMS).
- [ ] Add a “two chunks vs one chunk” regression test to lock in chunk boundary semantics in `train.online_updates` mode.
- [ ] Expose `cms_online_updates` / `cms_chunk_reduction` / `selfmod_online_updates` as Hydra config toggles (currently paper‑faithful defaults live in the HOPE block configs).
- [ ] Port the `train.py` online/per‑layer δℓ path to multi‑GPU (FSDP or custom DDP) so paper‑faithful mode scales beyond single GPU.

