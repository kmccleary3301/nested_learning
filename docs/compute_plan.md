# Compute Reservation Plan (Stage 2)

## Hardware
- Cluster: 2× nodes with dual NVIDIA RTX 6000 Ada (48 GB VRAM) + 64-core CPU + 512 GB RAM.
- Scheduler: Slurm (partition `gpu-a6000`), 2 nodes available concurrently.

## Reservations
| Phase | Resources | Duration | Window | Purpose |
|-------|-----------|----------|--------|---------|
| Pilot run | 1 node (2× A6000) | 3 days | Week 1 (Mon–Wed) | 160 M param sanity run, tokenizer validation |
| Ablations | 1 node | 2 days | Week 1 (Thu–Fri) | Self-modifier/CMS toggles at pilot scale |
| Mid-scale | 2 nodes | 10 days | Weeks 2–3 | 760 M training to 30 B tokens + evals |
| Mid evals | 1 node | 2 days | Week 3 (end) | Zero-shot + NIAH scripts on mid checkpoint |
| Target warmup | 2 nodes | 3 days | Week 4 (start) | 1.3 B config dry run (short token budget) |
| Target full run | 2 nodes | 14 days | Weeks 4–6 | 1.3 B / 100 B tokens |
| Final evals | 1 node | 3 days | Week 6 | Long-context + continual learning |

## Actions
1. Submit Slurm reservations (`scripts/compute/create_reservations.sh`) for the windows above; tag jobs with `NL-Stage2`.
2. Pre-stage datasets/token shards on node-local NVMe before each run to avoid network bottlenecks.
3. Enable checkpoint mirroring to shared storage every 12 hours for resilience.
4. Maintain utilization log in `reports/compute_usage.md` (to be created after first run).
