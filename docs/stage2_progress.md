# Stage 2 Progress

Last updated: `2026-02-24`

## Sprint Status

- **A-series (algorithm-mode + boundary-state fidelity):** Done
- **B-series (docs/usability/data-script robustness):** Done
- **C-series (cadence/compliance/mechanism tests):** Done
- **D-series (security/release hygiene gates):** Done
- **E-series (paper-compliance reconciliation + reproducibility):** Done
- **F-series (final validation + reporting):** In progress (documentation/report closure)
- **P0-series (packaging/CLI/runtime portability foundation):** Done (`nl doctor/smoke/train/audit`, `python -m nested_learning`)
- **P1-series (distribution/CI/release scaffolding):** Done (compat matrix, pip-first README, cross-platform smoke CI, release workflow)

## Done Criteria

1. Boundary-state path is runnable, guarded, and explicitly marked experimental.
2. Paper-faithful configs are explicit and test-covered.
3. Checkpoint metadata and release manifest include algorithm + online flags.
4. Data scripts have deterministic split fallback and `--help` smoke checks in CI.
5. Docs reference checks validate both file paths and markdown anchors.
6. Security gates block accidental binary/artifact tracking.
7. Compliance reports are generated from current configs and included in `reports/`.

## Validation Snapshot

- `uv run ruff check .` -> pass
- `uv run mypy src` -> pass
- `bash scripts/checks/run_fidelity_ci_subset.sh` -> pass
- `uv run pytest -q` -> pass

## Generated Compliance Artifacts

- `reports/compliance_summary_pilot_paper_faithful.json`
- `reports/compliance_summary_pilot.json`
- `reports/cadence_mechanism_audit_smoke.json`
- `reports/compliance_mechanism_audit_smoke.json`
- `reports/security_release_gate.md`
