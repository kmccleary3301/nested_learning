#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-pilot}

if [[ "${MODE}" == "pilot" ]]; then
  echo "[Smoke] Running pilot config on CPU"
  uv run python train.py --config-name pilot_smoke
elif [[ "${MODE}" == "mid" ]]; then
  echo "[Smoke] Ensuring filtered shards exist"
  if [[ ! -d "data/shards/refinedweb_filtered" ]]; then
    echo "Filtered shards missing. Generate them first via configs/data/refinedweb_mixture_filtered.yaml"
    exit 1
  fi
  echo "[Smoke] Running mid mixture config on CPU"
  uv run python train.py --config-name mid_smoke
else
  echo "Usage: scripts/run_smoke.sh [pilot|mid]"
  exit 1
fi
