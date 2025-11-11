# Planned Ablations – Pilot Run

This document tracks the ablation studies we intend to run once the 3 B-token pilot checkpoint is available. The goal is to isolate the contributions of teach-signal scaling, CMS chunk accumulation, self-modifiers, and optimizer choices (AdamW vs Muon) before moving to larger configs.

## 1. Teach-signal schedule
| Variant | Description | Status | Notes |
|---------|-------------|--------|-------|
| Baseline | Warmup 2 k → decay 120 k→140 k (current pilot config) | ⏳ | Use pilot checkpoint |
| No decay | Warmup only, no decay | ⏳ | Expect higher plasticity, risk of instability |
| Per-level scale | Different teach_scale per CMS level | ⏳ | Requires config changes |

## 2. CMS chunk accumulation
| Variant | Description | Status | Notes |
|---------|-------------|--------|-------|
| Full CMS | Chunk accumulation + telemetry (default) | ✅ smoke-tested | Verified via `tests/test_cms.py` |
| No chunking | Update each token (Transformer-like) | ⏳ | Compare long-context recall |
| Sparse chunks | Update every 512 tokens only | ⏳ | Stress long-term retention |

## 3. Self-modifier toggles
| Variant | Description | Status |
|---------|-------------|--------|
| Enabled | SelfModifier active (default) | ⏳ pilot |
| Disabled | Freeze self-modifier params | ⏳ |
| Teach-only | Teach signal applied but self-mod not updated | ⏳ |

## 4. Optimizer swaps
| Variant | Description | Status |
|---------|-------------|--------|
| AdamW fused | `optim.type=adamw` with fused kernels (default) | ✅ smoke-tested |
| Muon hybrid | `optim.type=muon` for ≥2D params, AdamW for embeddings/bias | ⏳ pilot |
| Full Muon | Force Muon everywhere | ⏳ (needs stability check) |

## 5. Automation hooks
| Tool | Purpose | Status | Notes |
|------|---------|--------|-------|
| `scripts/package_pilot_release.sh` | Copies latest pilot checkpoint/config/logs into `artifacts/pilot_release/` and updates metadata | ✅ | Use after every significant checkpoint (e.g., 1k-step milestones) so collaborators can download a coherent bundle. |
| `scripts/eval/run_pilot_suite.sh` | Runs zero-shot, NIAH (up to 64k), and continual harnesses (plus optional TITAN baseline) with memorization flags enabled | ✅ | Set `HOPE_CHECKPOINT`, `TITAN_*`, etc., to reuse for each ablation. Outputs land under `eval/`. |

## 5. Evaluation checklist per ablation
1. Run zero-shot suite (`scripts/eval/zeroshot.py --tasks all --memorize ...`).
2. Run extended NIAH (`--context-lengths 2048 --context-lengths 4096 --context-lengths 8192 --context-lengths 16384 --context-lengths 32768 --context-lengths 65536`).
3. Run continual-learning harness with memorization toggles (`--memorize --memorize-steps 2 --memorize-no-reset` and baseline run without memorization).
4. Record metrics in `artifacts/pilot_release/` (JSON/CSV) and summarize deltas here.

_Status legend:_ ✅ complete, ⏳ pending, 🔄 running, ⚠️ blocked.

## 6. Reference snapshot – Pilot step 600 (HOPE)
| Eval | Command | Output | Notes |
|------|---------|--------|-------|
| Zero-shot (PIQA, 32 samples, memorize on) | `scripts/eval/zeroshot.py --config configs/pilot.yaml --checkpoint artifacts/pilot_release/checkpoint.pt --tasks piqa --max-samples 32 --memorize --memorize-steps 2 --memorize-use-correct-answer` | `eval/zeroshot_pilot.json` → accuracy **0.5625** | Matches smoke baseline; ready baseline for future ablations. |
| NIAH (2k/4k contexts, 1 sample each) | `scripts/eval/niah.py ... --context-lengths 2048 --context-lengths 4096 --samples-per-length 1` | `eval/niah_pilot.json` → both accuracies **0.0** | As expected this early; use as reference once longer contexts show gains. |
| Continual (sample segments, 1 batch each) | `scripts/eval/continual.py ... --batch-size 2 --max-batches 1` | `eval/continual_pilot.json` → per-segment CE ≈ 39–43 | Provides initial forgetting baseline before running CMS/self-mod ablations. |

All outputs are copied to `artifacts/pilot_release/` via `scripts/package_pilot_release.sh` for reproducibility.
