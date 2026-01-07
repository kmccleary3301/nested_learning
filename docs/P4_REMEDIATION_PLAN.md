# P4 Remediation Plan — Paper-Faithful HOPE/Nested Learning

This checklist converts the planner critique into a concrete, ordered execution plan. It is scoped to **faithfulness-first** changes and minimal tests that prove correctness at small scale. Large-scale training is explicitly out of scope.

## P0 — Must-Fix Correctness (paper faithfulness)

### P0.1 Self‑modifying Titans always‑on in HOPE forward
- [ ] Add a non‑memorize path where `HOPESelfModBlock.forward` runs self‑mod updates whenever a fast state exists.
- [ ] Decide default behavior:
  - Option A: require explicit `self_mod_mode="online"` in config.
  - Option B: default to online when `fast_state` is provided (safer for backward compatibility).
- [ ] Add a small unit test confirming state changes without `teach_signal`.
- [ ] Document behavior toggle in `docs/experiments_report.md` or a compliance doc.

### P0.2 CMS update semantics: per‑token δ and **sum‑over‑chunk**
- [ ] Replace chunk‑mean δ broadcast with per‑token δ targets.
- [ ] Ensure update aggregation sums contributions over the chunk (Eq. 71), not average.
- [ ] Verify correct handling of partial chunks and masking.
- [ ] Add a minimal test with opposing per‑token δ to validate gradient direction.

### P0.3 Online CMS forward (read‑after‑write)
- [ ] Implement chunked CMS forward that updates CMS at chunk boundaries and uses the updated CMS for the next chunk.
- [ ] Ensure behavior is optional and gated by config (to avoid accidental regressions).
- [ ] Add a test showing outputs differ between online vs offline CMS for chunked input.

### P0.4 Layer‑wise δℓ signals (or explicit documented surrogate)
- [ ] Implement per‑block δ extraction via hooks (`retain_grad`) or explicit backward pass.
- [ ] Route δℓ into each block’s update path.
- [ ] If full δℓ is not feasible, explicitly document the surrogate (global teach signal) and gate it by config.
- [ ] Add a test comparing δℓ to global teach in a 2‑block toy model.

### P0.5 Implement M3 optimizer (paper‑accurate option)
- [ ] Implement M3 per paper (Newton–Schulz orthogonalization + multi‑scale momentum).
- [ ] Add a config option to choose `optim.type=m3`.
- [ ] Add a minimal update‑direction test on toy tensors to validate step.
- [ ] Keep existing Muon+AdamW path as baseline; document differences.

## P1 — Tests + Documentation to defend faithfulness

### P1.1 Minimal correctness tests (unit)
- [ ] Teach‑signal vs autograd gradient match (single‑layer toy).
- [ ] CMS per‑token δ vs chunk‑mean broadcast mismatch test.
- [ ] Self‑mod single‑step analytic check (linearized memory).
- [ ] Chunk boundary semantics test (two chunks vs one chunk).

### P1.2 Documentation: Paper compliance
- [ ] Add `docs/paper_compliance.md` mapping equations → code.
- [ ] Clearly label divergences and the toggle flags that isolate them.
- [ ] Update `README.md` with a “Paper‑faithful mode” section.

### P1.3 Telemetry & logging
- [ ] Log per‑level update stats with chunk index and surprise value.
- [ ] Emit “self‑mod online enabled” flags at run start.

## P2 — Optional (math‑preserving) improvements

### P2.1 Performance hygiene
- [ ] Avoid O(N²) online CMS in long sequences via chunk batching.
- [ ] Reduce overhead in CMS fast‑params functional_call path.

### P2.2 Config clarity
- [ ] Separate “chunk size” vs “update period” semantics in configs.
- [ ] Document which values match paper defaults.

