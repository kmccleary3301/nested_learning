# Release Checklist (Stage 2)

Use this list before tagging/publishing any checkpoint bundle.

## Faithfulness & Tests
- [ ] `uv run pytest` (especially `tests/test_teach_signal.py`, `tests/test_cms.py`, `tests/test_optim.py`, `tests/test_memorization.py`)
- [ ] Inner optimizer variant (`nl_l2_precond`) enabled in configs
- [ ] Teach-signal log shows finite norms across recent steps
- [ ] CMS chunk telemetry confirms expected update cadence
- [ ] `bash scripts/run_cpu_ddp_smoke.sh` (CPU DDP determinism)
- [ ] `bash scripts/tests/run_passkey_smoke.sh` (synthetic memorization)

## Artifacts
- [ ] Checkpoint `.pt` + `.yaml` + `.meta.json` (with tokenizer hash) in `artifacts/...`
- [ ] Tokenizer model + checksum JSON included
- [ ] Eval JSON/CSV (zero-shot, NIAH, continual) appended to `eval/`
- [ ] Checkpoint report filled from `docs/templates/checkpoint_report.md`
- [ ] Long-context extras (passkey, PG-19) + forgetting plots saved (`eval/passkey_*.json`, `eval/pg19_*.json`, `reports/plots/*.png`)
- [ ] Run `uv run python scripts/checkpoint/verify.py --checkpoint <path>` on every artifact

## Data & Provenance
- [ ] `data/manifest/refinedweb_full_manifest.json` updated if mixture changed
- [ ] `scripts/data/validate_mixture.py --manifest ...` report archived
- [ ] Tokenizer coverage JSON generated via `scripts/data/check_tokenizer_coverage.py`
- [ ] Coverage guard run (`scripts/checks/tokenizer_coverage_guard.py`) and JSON attached

## Logging & Monitoring
- [ ] W&B run link recorded in report
- [ ] Local JSON logs copied to `logs/`
- [ ] Memorizations stats (surprise counts, Titan/CMS updates) summarized

## Distribution
- [ ] README references any new scripts/configs
- [ ] Issue templates / release notes updated if new features shipped
- [ ] (Optional) Outreach draft added to `docs/POSTS.md`

Check these boxes before pushing tags or announcing new checkpoints so collaborators can reproduce results confidently.
