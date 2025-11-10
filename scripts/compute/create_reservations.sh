#!/usr/bin/env bash
set -euo pipefail

# Example Slurm reservations for Stage 2 (edit dates/times as needed).

PARTITION="gpu-a6000"
ACCOUNT="${ACCOUNT:-research}"

function reserve() {
  local name="$1"
  local start="$2"
  local duration="$3"
  local nodes="$4"
  scontrol create reservation="Name=${name},StartTime=${start},Duration=${duration},Nodes=${nodes},PartitionName=${PARTITION},Users=${USER},Accounts=${ACCOUNT}"
}

# Pilot run (1 node, 2 GPUs)
reserve "NL_Pilot" "2025-02-10T08:00:00" "3-00:00:00" 1

# Ablations (1 node)
reserve "NL_Ablations" "2025-02-13T08:00:00" "2-00:00:00" 1

# Mid-scale (2 nodes)
reserve "NL_Mid" "2025-02-17T08:00:00" "10-00:00:00" 2

# Mid evals (1 node)
reserve "NL_MidEval" "2025-02-27T08:00:00" "2-00:00:00" 1

# Target warmup (2 nodes)
reserve "NL_TargetWarmup" "2025-03-03T08:00:00" "3-00:00:00" 2

# Target full run (2 nodes)
reserve "NL_TargetFull" "2025-03-06T08:00:00" "14-00:00:00" 2

# Final evals (1 node)
reserve "NL_FinalEval" "2025-03-20T08:00:00" "3-00:00:00" 1

echo "Submitted reservations for Stage 2 (check with scontrol show reservation)."
