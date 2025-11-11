# Sprint Plan – Stage 2 Pilot & Results Sprint

**Window:** Nov 10 – Nov 17, 2025 (7 days)  
**Goal:** Produce reproducible pilot-scale HOPE checkpoints + evaluation packs, validate the data/infra path for mid-scale runs, and capture documentation that unlocks Stage 2 scaling.  
**Success Criteria:**
1. Pilot (≈160 M params, 3 B tokens) runs end-to-end with checkpoints + W&B logs.
2. Zero-shot, NIAH, and continual-learning eval JSONs produced for the pilot checkpoint and compared to TITAN baseline.
3. Data provenance + environment setup documented so collaborators can rerun without context.
4. Reports updated with pilot metrics, open issues, and risk mitigations.

## Constraints & Resources
- **Hardware:** dual RTX 6000 Ada (48 GB) → default to `cuda:1` for single-GPU jobs; DDP uses both.
- **Framework:** PyTorch 2.9 + torchvision 0.24 via `uv`.
- **Data:** RefinedWeb/FineWeb proxy mix (`data/shards/*_full`) already filtered; tokenizer at `artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model`.
- **Tracking:** W&B project `hope-stage2`; JSON logs under `logs/`.
- **Blocking risks:** insufficient storage for 100 B tokens, instability when teach_scale > 0.1, and long eval runtimes (>30 min) that must run inside tmux.

## Workstreams & Detailed TODOs

### P0 – Faithfulness & Critique Remediation (must finish before P1+)
| ID | Task | Subtasks | Acceptance |
|----|------|----------|------------|
| F1 | **Weight tying + teach-signal fix** | (a) Tie `lm_head.weight` to `embed.weight` in HOPE + TITAN models. (b) Rewrite `compute_teach_signal` to time-shift targets and use the head weight. (c) Emit teach-signal norms in logs. (d) Add `tests/test_teach_signal.py` verifying finite-difference gradient vs analytic teacher. | CI test passes; pilot smoke log contains `teach_signal_norm` with finite values. |
| F2 | **CMS chunk accumulation** | (a) Introduce per-level buffers sized to `update_period`. (b) Trigger optimizer updates only when buffer fills, then clear. (c) Add telemetry (count, L2 magnitude). (d) Unit test that `update_period=3` produces exactly one update every three ticks. | Unit test green; logs show stepped CMS updates aligned with periods. |
| F3 | **L2 regression inner rule (Eq. 27–29)** | (a) Add `variant="nl_l2_precond"` to deep optimizer. (b) Plumb activations into optimizer context. (c) Config flag + defaults in `configs/hope/*.yaml`. (d) Toy regression test demonstrating objective decrease. | Test passes; pilot smoke with `variant` enabled logs preconditioner stats. |
| F4 | **Test-time memorization path** | (a) Add CLI flags to eval scripts. (b) Implement Titan memory updates during eval when `memorize=True`. (c) Optional CMS fast-level updates. (d) Synthetic integration test verifying improved accuracy on memorization-enabled run. | `scripts/tests/test_memorization_eval.py` green; eval JSONs contain `_memorize` variants. |
| F5 | **PyTorch perf upgrades (SDPA, autocast, compile, fused AdamW)** | (a) Replace attention with manual QKV + SDPA (Flash support). (b) Wrap train step in `torch.autocast(..., dtype=torch.bfloat16)` with config switch. (c) Add guarded `torch.compile`. (d) Enable fused AdamW. | Pilot smoke runtime comparison recorded; fallback path works on CPU (smoke test). |
| F6 | **Muon outer optimizer option** | (a) Detect `torch.optim.Muon`. (b) Split param groups (matrices→Muon, others→AdamW). (c) Config knob `optim.outer.type`. (d) Document trade-offs + log metrics. | Pilot smoke completes with `optim.outer.type=muon_mix`; README/env docs updated. |
| F7 | **Seeding + backend robustness** | (a) Hydra-level `seed` field; set Python/NumPy/Torch seeds. (b) DataLoader worker init + manual seed for rng. (c) Auto-select DDP backend; allow override. (d) Add CPU DDP CI job. | Two identical runs (same seed) produce identical log traces; CI includes CPU DDP smoke. |
| F8 | **License & packaging polish** | (a) Align `pyproject.toml` license with Apache-2.0. (b) Ensure referenced scripts ship (`scripts/run_e2e_smoke.sh`). (c) README update for memorization, Muon, env. | Lint job verifies license metadata; README instructions reviewed/approved. |

### 1. Data & Environment Readiness
| ID | Task | Details | Owner | Status | Deliverable |
|----|------|---------|-------|--------|-------------|
| D1 | Corpus inventory | Verify presence + integrity of `data/shards/*_full` (RefinedWeb, Wikipedia, C4, RedPajama, Code) and update stats in `data/mixtures/refinedweb_mix_full_shards.json`. | KM | Not Started | Updated JSON + short log |
| D2 | Provenance doc | Extend `docs/data_pipeline.md` with acquisition commands, licensing notes, and shard counts per corpus. | KM | Not Started | PR-ready doc |
| D3 | Tokenizer lock | Record checksum + training command for `artifacts/tokenizer/...32k.model` in `docs/data_pipeline.md`; script to assert checksum before runs. | KM | Not Started | Script `scripts/data/check_tokenizer.py` + doc snippet |
| D4 | Env matrix | Confirm `uv.lock` captures torch 2.9 stack; add `docs/env_matrix.md` describing GPU driver, CUDA runtime, `uv` commands, and fallback instructions. | KM | Not Started | Env doc + verified `uv pip list` diff |

### 2. Pilot Training Execution
| ID | Task | Details | Owner | Status | Deliverable |
|----|------|---------|-------|--------|-------------|
| P1 | Upgrade `configs/hope/pilot.yaml` | Move from synthetic toy config to 160 M spec (layers=12, dim=768, seq=2048, batch ladder). Include teach schedule, optimizer, data mixture hook. | KM | Not Started | Updated YAML + changelog |
| P2 | Dry-run smoke | `uv run python train.py --config-name hope/pilot --train.steps=50 --train.device=cuda:1` using sample shards to confirm stability/logging. | KM | Not Started | Log `logs/pilot_smoke.json` |
| P3 | Full pilot launch | tmux-managed torchrun/DeepSpeed to 3 B tokens (≈150k steps). Enable checkpoints every 1k steps in `artifacts/checkpoints/pilot`. | KM | Not Started | Checkpoints + `logs/pilot_full.json` + W&B link |
| P4 | Monitoring hooks | Ensure gradient/teach-scale stats captured (level update magnitudes, CMS norms). Implement metrics in `src/training/callbacks.py`. | KM | Not Started | Callback code + metrics in log |
| P5 | Artifact packaging | Copy best checkpoint, config snapshot, log, and eval metadata to `artifacts/pilot_release/`. | KM | Not Started | Structured folder for sharing |

### 3. Evaluation & Baselines
| ID | Task | Details | Owner | Status | Deliverable |
|----|------|---------|-------|--------|-------------|
| E1 | Zero-shot sweep | Run `scripts/eval/zeroshot.py` on pilot + TITAN baseline checkpoints (PIQA, HellaSwag, WinoGrande, ARC-E/C, BoolQ, SIQA, OpenBookQA). Store JSON under `eval/zeroshot_pilot_v1.json`. | KM | Not Started | Eval JSON + summary table |
| E2 | NIAH curve | Expand `scripts/eval/niah.py` to 2k→64k contexts, add CLI for seeds/batch. Plot accuracy vs. length and save PNG + CSV in `reports/plots/niah_pilot.*`. | KM | Not Started | CSV + plot |
| E3 | Continual-learning bench | Finalize `scripts/eval/continual.py` to iterate through `configs/data/continual_segments_full.yaml`, log forgetting metrics, and compare to TITAN baseline. | KM | In Progress (per scaffolding) | JSON `eval/continual_pilot.json` + diff vs. baseline |
| E4 | Baseline rerun | Reproduce TITAN-only run matching pilot data/time (use `configs/mid_titan_baseline.yaml` adjusted for 160 M). Document differences and store checkpoints. | KM | Not Started | Baseline checkpoint + eval JSON |

### 4. Documentation & Reporting
| ID | Task | Details | Owner | Status | Deliverable |
|----|------|---------|-------|--------|-------------|
| R1 | Update `docs/experiments_report.md` | Add pilot training summary, metrics tables, and comparison vs. TITAN. Include open issues + next actions. | KM | Not Started | Updated report |
| R2 | `docs/stage2_progress.md` refresh | Append sprint log entries (date, what ran, pointers to artifacts). | KM | Not Started | Progress section |
| R3 | `reports/ablations.md` | Stub section for pilot-scale ablations (self-modifier on/off, CMS depth). Outline command templates even if runs pending. | KM | Not Started | Markdown updates |
| R4 | Release checklist | Update `docs/release_checklist.md` (or add if missing) with pilot deliverables, git tags, and artifact verification steps. | KM | Not Started | Checklist doc |

### 5. Outreach & Coordination
| ID | Task | Details | Owner | Status | Deliverable |
|----|------|---------|-------|--------|-------------|
| O1 | Issues roadmap | Open GitHub issues for P1–P5, E1–E4, R1–R4 to invite contributions; include artifact links. | KM | Not Started | Issue list |
| O2 | README updates | Highlight pilot deliverables, add “How to reproduce pilot run” section with commands + data requirements. | KM | Not Started | README diff |
| O3 | Community sync | Draft short update in `docs/POSTS.md` for Discord/Twitter after pilot results land (link to dashboards). | KM | Not Started | Draft |

## Execution Order & Dependencies
1. **D1–D4** unblock everything; complete before P1.
2. **P1 → P2 → P3** sequential; P4 instrumentation can merge during P2/P3 once metrics tested.
3. **P3 completion** gates E1–E3; E4 can run in parallel using existing baseline config.
4. **R1–R4** depend on eval outputs; draft skeletons early to keep pace.
5. Outreach tasks (O1–O3) happen once initial pilot artifacts exist.

## Tracking & Reporting
- Update `TODO.md` and this sprint doc daily with status (☐/△/✓).
- Maintain tmux session names (`pilot_full`, `pilot_eval`, `pilot_baseline`) and log paths.
- Push commits frequently; tag `v0.2.0-pilot` when criteria met.
- Capture blockers in `docs/stage2_progress.md` so future shifts have context.

## Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Pilot instability when teach_scale > 0.05 | Implement gradient clipping per level, adaptive schedules, and fallback to 0.05 if divergence occurs. |
| Data storage pressure during 3 B-token run | Stream shards using lazy loader; clean intermediate caches under `tmp/` after runs. |
| Eval runtime (NIAH up to 64k) | Batch contexts, reuse cached passkeys, and run in tmux `pilot_eval`. |
| Artifact drift | Snapshot `uv.lock`, configs, and tokenizer hash into `artifacts/pilot_release/metadata.json`. |

---

This sprint plan is self-contained; executing the tasks above will deliver a fully documented pilot-scale reproduction plus the infrastructure needed for mid-scale Stage 2 runs.
