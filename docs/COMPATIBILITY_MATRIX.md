# Compatibility Matrix

This document defines the support contract for runtime/backends.

## Support Tiers

- **Tier 1 (Supported):** CI-tested on every PR; regressions treated as bugs.
- **Tier 2 (Supported with caveats):** tested periodically/partially; backend caveats apply.
- **Tier 3 (Best-effort):** community-supported; no guaranteed CI lane.
- **Unsupported:** intentionally out of scope; fail-fast when correctness is at risk.

## Matrix

| OS | Python | CPU | CUDA (NVIDIA) | MPS | ROCm |
|---|---|---|---|---|---|
| Linux x86_64 | 3.10-3.12 | Tier 1 (import/eval/smoke) | Tier 1 (import/eval/smoke/full training) | Unsupported | Tier 3 |
| macOS Apple Silicon | 3.10-3.12 | Tier 2 (import/eval/smoke) | Unsupported | Tier 2 (import/eval), Tier 3 (smoke) | Unsupported |
| macOS Intel | 3.10-3.12 | Tier 2 (import/eval), Tier 3 (smoke) | Unsupported | Unsupported | Unsupported |
| Windows | 3.10-3.12 | Tier 2 (import/eval), Tier 3 (smoke) | Tier 3 (user-managed) | Unsupported | Unsupported |

Notes:
- CPU full-scale training is not a supported target.
- Strict paper-faithful online-update semantics in distributed settings remain constrained by design.
- Numerical parity across backend families (CUDA/MPS/ROCm) is not guaranteed.

## Runtime Degradation Policy

At runtime, unsupported performance features should degrade gracefully:
- if flash/mem-efficient SDPA is unavailable, use math SDPA;
- if `torch.compile` is unavailable/disabled, continue without compile;
- if requested mixed precision is unsupported on the backend, degrade to fp32 and log it.

Use `nl doctor --json` to capture capability snapshots in machine-readable form.

## Golden Environment

For reproducibility of this repository’s published artifacts, prefer:
- Python 3.12
- PyTorch 2.9.x
- `uv lock` / `uv sync --all-extras --dev`

The package metadata allows broader install ranges for portability, while the lockfile remains the canonical dev/test environment.

