# Versioning and Stability Policy

This repository follows SemVer-style versioning with explicit 0.x constraints.

## Current Phase: 0.x

Before `1.0.0`, stability guarantees are intentionally limited:
- `0.x.y` patch releases should be non-breaking for normal workflows.
- `0.X.0` minor releases may include breaking changes to config schema, defaults, CLI behavior, or checkpoint metadata.

## Public Surface

Stable-ish surfaces (prioritized for compatibility):
- `nl` CLI commands and flags
- Hydra config schema for primary shipped configs
- checkpoint sidecar metadata fields used by verification tooling

Explicitly unstable surfaces:
- internal Python module APIs
- experimental mechanism paths and ablation-only options
- ad hoc scripts under `scripts/` unless documented as stable entrypoints

## Breaking Change Handling

When a release introduces breakage:
1. call it out in `CHANGELOG.md`,
2. include migration notes,
3. keep old behavior behind compatibility flags where reasonable for at least one minor cycle.

## Golden Environment vs Supported Ranges

- Golden reproduction environment: lockfile-based (`uv lock`, Python 3.12, PyTorch 2.9.x).
- Package metadata supports broader compatibility ranges for portability.
- If range installs diverge from golden behavior, prefer golden env for paper-faithful runs.

