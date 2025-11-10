# Stage 1 Plan – Nested Learning (HOPE) Architecture Reproduction

This document specifies the full execution plan for Stage 1: reproducing the Nested Learning (NL) architecture—specifically the HOPE model comprising multi-frequency levels, TITAN-style memory, Continuum Memory System (CMS), self-modification pathways, and deep optimizers. It is self-contained and assumes no additional context.

---

## 1. Executive Summary
- **Objective:** Build a modular PyTorch implementation of HOPE (self-modifying TITAN + CMS) faithful to Google’s Nested Learning paradigm, ready for subsequent Stage 2 training and result replication.
- **Approach:** Codify NL’s level-clock abstraction, associative memories, CMS chain, deep optimizer family, and HOPE block, using lucidrains’ TITANs repository as a reference while adhering to project constraints (PyTorch LTS, `uv` tooling, isolated environment).
- **Outputs:** A tested codebase under `src/nested_learning/`, configuration files, reproducible environment definition, and documentation describing design choices, APIs, and usage.

---

## 2. Scope Definition

### 2.1 Goals
- Translate the NL paper/blog specification into production-ready modules (levels, memories, optimizers, HOPE assembly).
- Provide runnable training/evaluation harnesses for architecture-level smoke tests (synthetic or small corpora).
- Document APIs, update schedules, and configuration knobs so Stage 2 can focus on scaling experiments without refactoring.

### 2.2 Non-goals
- Large-scale data processing, hyper-parameter sweeps, or benchmark reproduction (Stage 2).
- Deployment/serving optimizations beyond correctness and basic throughput profiling.
- Alternative paradigms (e.g., SSM-only models) except where they interface with HOPE as baselines.

---

## 3. Reference Inputs & Artifacts
- `docs/planner_convo_01.md` – canonical breakdown of NL, CMS, HOPE, deep optimizers, and evaluation expectations.
- `ref_repos/titans-pytorch` – lucidrains’ TITAN implementation for memory update heuristics and attention-memory interfacing.
- NL paper + Google blog (quoted in planner doc) – ground-truth equations and frequency semantics.

---

## 4. Technical Ground Rules
- **Environment:** Use `uv` for Python toolchain management and package installation. Target Python 3.12.
- **Framework:** Latest PyTorch LTS release (currently PyTorch 2.4 LTS) with CUDA extensions as needed.
- **Project Layout:** PEP 517-compliant `pyproject.toml`; source under `src/`.
- **Device Support:** CPU + GPU; design APIs to support FSDP/DeepSpeed later without refactor.
- **Licensing:** Mirror repository license; respect lucidrains’ MIT terms if code is adapted (document attributions).

---

## 5. Deliverables & Acceptance Criteria

| Deliverable | Description | Acceptance Criteria |
|-------------|-------------|---------------------|
| Environment spec | `uv.lock`, `pyproject.toml`, setup docs | Fresh clone reproducibly installs all deps via `uv sync` |
| Core library | `src/nested_learning/` modules for levels, memories, optimizers, CMS, HOPE model | Unit tests cover scheduler logic, CMS composition, optimizer updates |
| Training harness | `train.py`, config files, CLI entry points | Can run smoke training on toy data, exercising all update pathways |
| Documentation | README + module docstrings + this plan | Explains architecture, configuration, and extension points |
| Test suite | Pytest-based tests + lint config | CI (or local run) passes formatter, type hints (optional), and tests |

---

## 6. Work Breakdown Structure (WBS)

### 6.1 Workstream A – Environment & Tooling
1. Initialize `uv` project (`uv init`, `uv python install 3.12`).
2. Define dependencies: `torch==2.4.*`, `torchvision` (if needed), `torchaudio`, `einops`, `numpy`, `regex`, config libs (`hydra-core` or `omegaconf`), pytest stack.
3. Configure `ruff` + `mypy` for lint/type checks; add pre-commit optional hooks.
4. Document setup commands in `README`.

### 6.2 Workstream B – Specification Capture
1. Translate NL equations into implementation notes (update frequencies, associative memories, CMS formulas).
2. Produce interface contracts for each module (inputs/outputs, shapes, update semantics).
3. Annotate dependencies between modules (e.g., HOPEBlock uses LevelClock, TitanMemory, CMS, SelfModifier).

### 6.3 Workstream C – Core Abstractions & Utilities
1. Implement `LevelSpec`, `LevelClock`, and scheduling utilities (`nested_learning/levels.py`).
2. Define `AssocMemory` abstract base class with retrieval and update hooks.
3. Provide helper mixins for instrumentation (e.g., timeline logging) to aid debugging of frequency gating.

### 6.4 Workstream D – Memory Systems
1. **TITAN Memory:** Port/extend lucidrains’ implementation into `nested_learning/titan/memory.py`, exposing `forward`, `score_surprise`, and `update` APIs.
2. **Continuum Memory System:** Implement CMS chain per Eqs. (30–31) with configurable depth, widths, and update periods. Include gradient-suppressed update path for frozen steps.
3. **Interfacing:** Ensure CMS supports both training-time gradient descent and online mini-updates triggered by LevelClock.

### 6.5 Workstream E – Self-Modifier & Deep Optimizers
1. Implement deep optimizer family (`optim/deep_momentum.py`): preconditioned momentum, L2 objective variant, DMGD, Muon-equivalent.
2. Build `SelfModifier` module that ingests key/value/error tensors and emits parameter deltas (low-rank or coordinate).
3. Connect self-modifier outputs to TITAN memory + CMS parameters with safeguards (clipping, EMA, update dropout).

### 6.6 Workstream F – HOPE Blocks & Model Assembly
1. Define attention backbone choices (full attention, sliding-window, optional DeltaNet stub).
2. Assemble `HOPEBlock` combining attention, TITAN memory retrieval, CMS consolidation, and gated self-modification.
3. Stack blocks into `HOPEModel` with embeddings, layer norms, and LM head.
4. Provide configuration schemas (YAML/JSON) for standard model sizes (e.g., 120M pilot).

### 6.7 Workstream G – Training & Evaluation Harness
1. Build dataset/tokenizer adapters (placeholder uses synthetic data or small text set for smoke testing).
2. Implement training loop with two phases per batch (outer loss + optional inner self-mod updates) per planner guidance.
3. Add evaluation scripts for perplexity on validation split and simple long-context retrieval toy tasks.
4. Integrate logging (stdout/JSON) to capture level firing statistics and update magnitudes.

### 6.8 Workstream H – Quality Assurance & Documentation
1. Write unit tests for LevelClock, CMS update gating, deep optimizers, self-modifier stability checks, and HOPEBlock forward shapes.
2. Add integration tests running a few dozen steps on dummy data verifying that scheduled updates change parameters.
3. Document module usage, config options, and extension hooks; update README with quickstart.

---

## 7. Sequence & Dependencies
1. **A → B:** Environment must exist before codifying specs to ensure type imports and linting work.
2. **C depends on B:** Abstractions rely on finalized interfaces.
3. **D & E parallel:** Memory systems and self-modifier/optimizers can progress concurrently once base abstractions exist.
4. **F waits on C–E:** HOPE assembly requires all underlying modules.
5. **G depends on F:** Training harness needs a constructed model.
6. **H spans entire project:** Tests/docs evolve alongside implementation but finalize after F/G stabilize.

---

## 8. Testing & Validation Strategy
- **Unit tests:** Deterministic clocks, optimizer math (compare against analytical expectations), CMS partial updates.
- **Property tests:** Ensure parameter updates occur only when `LevelClock` fires; verify gradient isolation on frozen steps.
- **Integration tests:** End-to-end HOPEModel forward/backward on short sequences; confirm self-mod updates change memory weights without affecting unrelated modules.
- **Performance sanity:** Profiling scripts to measure per-step time and memory for pilot configs; ensure inner-loop updates add bounded overhead.

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Unclear self-modifier signals from paper | Incorrect update dynamics | Start with residual-based “surprise” as per planner notes; keep interface pluggable for future signals |
| Deep optimizer instability | Divergence during inner updates | Implement norm clipping, small inner lrs, EMA smoothing, and gated application frequency |
| CMS update cost | Excess compute for long chains | Support selective activation, mixed precision, and caching of frozen states |
| Reference drift from lucidrains repo | API mismatch | Treat ref repo as read-only reference; document deviations and add adapter tests |

---

## 10. Open Questions (resolved)
1. **Tokenizer / corpus:** Use the TinyStories dataset (CC BY-SA) with a SentencePiece unigram tokenizer at 32k vocabulary; synthetic data remains for unit tests only.
2. **Configuration management:** Adopt Hydra + OmegaConf for experiment configs (already added as dependencies); keep dataclass helpers for type hints.
3. **Hardware target:** Assume access to 2× RTX 6000 Ada (48 GB each) for validation; Stage 1 smoke tests should also run on a single GPU or CPU fallback.

---

## 11. Next Steps After Approval
1. Stand up the `uv` environment and scaffold repository layout.
2. Implement Workstreams C–F iteratively with accompanying tests.
3. Run end-to-end smoke training to validate architecture wiring, then pause for Stage 2 planning.
