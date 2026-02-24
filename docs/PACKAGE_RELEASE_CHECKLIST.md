# Package Release Checklist (PyPI/GitHub)

Use this checklist for package distribution releases (separate from checkpoint/artifact releases).

## Pre-Release (RC)

- [ ] `uv run ruff check .`
- [ ] `uv run mypy src`
- [ ] `uv run pytest -q`
- [ ] `uv build`
- [ ] `uvx twine check dist/*`
- [ ] wheel install smoke works outside repo tree:
  - [ ] `python -m venv /tmp/nl-wheel`
  - [ ] `pip install dist/*.whl`
  - [ ] `python -m nested_learning --help`
  - [ ] `python -m nested_learning doctor --json`
  - [ ] `python -m nested_learning smoke --config-name pilot_smoke --device cpu --batch-size 1 --seq-len 8`
- [ ] `CHANGELOG.md` updated with:
  - [ ] release highlights
  - [ ] breaking changes (or explicit “none”)
- [ ] `README.md` reflects current compatibility tiers and install guidance.
- [ ] Trusted Publishing configured per `docs/PYPI_TRUSTED_PUBLISHING.md`.
- [ ] Tag created for RC (`vX.Y.ZrcN`) and TestPyPI publish succeeds.

## Final Release

- [ ] Re-run validation checks listed above.
- [ ] Promote release notes for `vX.Y.Z`.
- [ ] PyPI publish workflow succeeds via Trusted Publishing (OIDC).
- [ ] GitHub Release published with:
  - [ ] changelog summary
  - [ ] migration notes (if any)
  - [ ] links to docs and compatibility matrix

## Post-Release

- [ ] Confirm install from PyPI in a clean environment.
- [ ] Confirm `nl doctor` and `nl smoke` on at least one non-maintainer machine or CI lane.
- [ ] Open follow-up issues for deferred release items.
