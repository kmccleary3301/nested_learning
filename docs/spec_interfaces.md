# Interface Notes for Nested Learning Modules

## LevelClock / LevelSpec (`nested_learning.levels`)
- `LevelSpec`: name, update_period, warmup, jitter, optimizer binding.
- `LevelClock`: tracks global step, exposes `should_update(name)` and `record_update(name)`; keeps timeline for logging.

## AssocMemory (`nested_learning.assoc_memory`)
- Abstracts retrieval (`forward`) and writeback (`update`). Concrete memories (TITAN, CMS) implement this along with optional `reset_state` from `SupportsReset` protocol.

## CMS (`nested_learning.cms` – forthcoming)
- Chain of MLPs with per-level clocks; includes `forward` for retrieval/composition and `maybe_update` for gated parameter updates.

## TITAN Memory (`nested_learning.titan.memory` – forthcoming)
- Learnable long-term memory approximating lucidrains implementation; provides `score_surprise` + `update` to support self-modifier pathways.

## SelfModifier & Deep Optimizers (`nested_learning.hope.self_mod`, `nested_learning.optim.deep`) 
- SelfModifier: neural updater that emits parameter deltas based on (key, value, error).
- Deep optimizers: generalize momentum/Adam with pluggable associative memories.

## HOPE Block/Model (`nested_learning.hope.block`, `nested_learning.model`)
- Composition of attention backbone, TITAN retrieval, CMS consolidation, and self-mod updates, assembled into full autoregressive model.

These notes ensure consistency across workstreams and act as inline documentation while implementing the remaining modules.
