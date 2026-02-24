---
name: Performance regression
about: Report a training / eval performance drop vs. baseline
title: "[Perf] "
labels: ["performance", "needs-triage"]
assignees: []
---

## Summary
Describe the regression and the baseline you’re comparing against.

## Baseline
- Config / checkpoint:
- Metrics (loss / ppl / eval scores):

## Repro steps
Exact commands with overrides, plus hardware details.

## Environment
- OS:
- Python:
- Torch:
- Backend (`cpu` / `cuda` / `mps` / `rocm`):
- GPU/accelerator model (if any):

If using ROCm: this project currently treats ROCm support as best-effort. Include HIP/ROCm version and exact torch build.

## Logs / artifacts
Attach relevant logs, W&B links, or JSON eval files.

## Suspected cause
Optional theory or related commits/PRs.
