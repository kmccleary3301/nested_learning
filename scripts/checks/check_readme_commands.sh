#!/usr/bin/env bash
set -euo pipefail

# Keep README's core CLI guidance executable in CI.
uv run nl --help >/dev/null
uv run nl doctor --json >/dev/null
uv run nl smoke --help >/dev/null
uv run python -m nested_learning --help >/dev/null

echo "README command smoke checks passed."

