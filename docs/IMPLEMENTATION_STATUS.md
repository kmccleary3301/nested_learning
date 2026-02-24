# Implementation Status (Source of Truth)

This table is the canonical mechanism-status map for this repo.

| Mechanism | Status | Evidence |
|---|---|---|
| Teach-signal alignment (LM head tied to embeddings) | Implemented | `src/nested_learning/model.py`, `src/nested_learning/training.py`, `tests/test_teach_signal.py`, `tests/test_tied_weight_guard.py` |
| Per-layer local teach signals (`δ_l`) | Implemented (single-process path) | `src/nested_learning/model.py`, `src/nested_learning/training.py`, `tests/test_teach_signal.py` |
| CMS chunk accumulation + cross-call cadence | Implemented | `src/nested_learning/hope/block.py`, `tests/test_cms.py`, `tests/test_cms_cross_call.py`, `tests/test_model_streaming_cadence.py` |
| CMS finalize/partial flush semantics | Implemented | `src/nested_learning/hope/block.py`, `tests/test_cms_flush_partial.py`, `docs/STREAMING_CONTRACT.md` |
| Online chunking (overlap mode) | Implemented | `src/nested_learning/training.py`, `tests/test_online_chunking.py` |
| Online chunking (boundary-target mode) | Implemented | `src/nested_learning/training.py`, `configs/pilot_paper_faithful.yaml`, `tests/test_online_chunking.py` |
| Optional attention-cache carry across chunk calls | Implemented (single-process path) | `src/nested_learning/backbones.py`, `src/nested_learning/model.py`, `tests/test_attention_cache.py`, `tests/test_eval_state.py` |
| Strict runtime guardrails | Implemented | `src/nested_learning/training.py`, `tests/test_strict_streaming_contract.py`, `tests/test_distributed_fail_fast.py`, `tests/test_fast_state_batch_semantics.py` |
| Training algorithm mode banner/validation | Implemented (`two_pass_stopgrad_updates`, `boundary_state_grad_through_write`) | `src/nested_learning/training.py`, `tests/test_strict_streaming_contract.py` |
| Boundary-state gradient-through-write algorithm mode | Experimental (single-process constrained path) | `src/nested_learning/training.py`, `tests/test_boundary_state_mode.py`, `tests/test_algorithm_mode_grad.py`, `docs/PAPER_COMPLIANCE.md` |
| Online-updates fast-state invariant (`online_updates && !use_fast_state`) | Implemented (warn/error guard) | `src/nested_learning/training.py`, `tests/test_strict_streaming_contract.py` |
| Inner optimizer mapping (`nl_l2_precond`) | Implemented (best-effort mapping) | `src/nested_learning/optim/deep.py`, `tests/test_optim.py`, `docs/PAPER_COMPLIANCE.md` |
| Surprise-gated update flow | Implemented | `src/nested_learning/model.py`, `src/nested_learning/hope/block.py`, `src/nested_learning/titan/model.py` |
| Test-time memorization path in eval harnesses | Implemented | `src/nested_learning/memorize.py`, `scripts/eval/*.py`, `tests/test_memorization.py` |
| Compliance automation report | Implemented | `scripts/checks/compliance_report.py`, `scripts/checks/run_fidelity_ci_subset.sh`, `scripts/run_mechanism_audit_smoke.sh` |
| Doc-to-code reference guard (anti-overclaim drift) | Implemented | `scripts/checks/verify_docs_refs.py`, `.github/workflows/ci.yml`, `tests/test_verify_docs_refs.py` |
| Portable package/CLI entrypoints (`nl`, `python -m nested_learning`) | Implemented | `src/nested_learning/cli.py`, `src/nested_learning/__main__.py`, `tests/test_cli_tooling.py`, `pyproject.toml` |
| Cross-platform smoke + wheel install CI gates | Implemented | `.github/workflows/ci.yml` (`cross-platform-smoke`, `wheel-install-smoke`) |
| Package release automation (tag -> TestPyPI/PyPI) | Implemented | `.github/workflows/release.yml`, `docs/PACKAGE_RELEASE_CHECKLIST.md` |
| Full boundary-state gradient-through-write algorithm from paper | Partially implemented (experimental) | Constrained single-process mode exists; not yet treated as production/full-scale parity (`docs/PAPER_COMPLIANCE.md`) |
| Distributed mechanism-auditing parity for online/per-layer/boundary-cache path | Deferred | DDP strict fail-fast + documented limits (`src/nested_learning/training.py`, `scripts/run_cpu_ddp_smoke.sh`) |
| Paper-scale training/eval reproduction | Deferred | Explicitly out of sprint scope (`docs/PAPER_COMPLIANCE.md`) |

## Validation Entrypoints

- Fidelity subset: `scripts/checks/run_fidelity_ci_subset.sh`
- Mechanism-auditing smoke: `scripts/run_mechanism_audit_smoke.sh`
- Full tests: `uv run pytest`
