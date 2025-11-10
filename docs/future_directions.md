# Future Directions – Nested Learning Reproduction

This roadmap outlines high-impact areas for contributors once the initial public release is out. Items are organized by theme and roughly prioritized.

---

## 1. Scaling the Architecture
1. **Longer Runs (≥3B tokens):** Use FSDP or DeepSpeed ZeRO to train the 760 M config on the filtered `_full` shards. Target at least 3B tokens so HOPE’s long-context advantages can emerge.
2. **Target Config (1.3 B / 100 B tokens):** Prepare configs and launcher scripts for multi-node environments (Slurm, Kubernetes). Emphasize reproducible manifests and resume logic.
3. **Context Expansion:** Integrate FlashAttention2 or block-sparse attention to push context lengths beyond 32k tokens. Update `scripts/eval/niah.py` accordingly.

## 2. Evaluation & Analysis
1. **Full Benchmark Suite:** Extend `scripts/eval/zeroshot.py` to include ARC-E/C, BoolQ, SIQA by default with standard prompts. Automate results aggregation into Markdown tables.
2. **Long-Context Benchmarks:** Add Passkey, PG19, and retrieval tasks besides Needle-in-a-Haystack.
3. **Continual Learning:** Create larger segment manifests (e.g., Wikipedia by year) and compute forgetting metrics across dozens of checkpoints.

## 3. Optimization & HPO
1. **Teach-Scale Scheduling:** Explore cosine or per-level schedules; integrate gradient clipping hyperparameters through Hydra sweeps.
2. **Optimizer Variants:** Try Muon/DeepMomentum for TITAN/CMS updates. Compare against simple SGD/Adam baselines.
3. **Automated Sweeps:** Wire up lightweight HPO (Ray Tune, Ax) for pilot configs to test teach_scale, clip, and CMS depth combinations.

## 4. Data & Tooling
1. **Dataset Expansion:** Add book/video/code corpora, ensure licensing compliance, and document provenance.
2. **Tokenizer Experiments:** Evaluate alternative vocab sizes or SentencePiece BPE to see if certain domains benefit.
3. **CI Enhancements:** Add GPU-aware smoke tests (e.g., GitHub self-hosted runner) to catch regressions in dual-GPU workflows.

## 5. Documentation & Community
1. **Release Notes:** Publish structured release notes with each tagged version (capabilities, limitations, roadmap).
2. **Contributor Guides:** Document coding standards, logging conventions, and how to submit new configs/evals.
3. **Experiment Tracking:** Encourage use of the `docs/experiments_report.md` template for all major runs to keep the public record up to date.

---

Contributors are welcome to pick any of these items (or propose new ones) via GitHub issues or pull requests. Please cross-reference this file so efforts stay coordinated.*** End Patch
