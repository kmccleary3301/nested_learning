#!/usr/bin/env bash

set -euo pipefail

# Force CPU execution so torchrun selects the gloo backend.
export CUDA_VISIBLE_DEVICES=""

uv run torchrun --standalone --nproc_per_node=2 train_dist.py --config-name pilot_smoke "$@"
