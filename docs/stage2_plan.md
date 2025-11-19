# Stage 2 Plan – Nested Learning (HOPE) Results Reproduction

This document details Stage 2 goals: reproduce the key experimental results from Google’s Nested Learning (HOPE) paper/blog using the Stage 1 codebase. It is self-contained and assumes Stage 1 deliverables (architecture, training harness, tests, `uv` environment) are ready.

---

## 1. Objectives
- Train HOPE models at multiple scales (pilot 160 M, target 760 M and 1.3 B parameters) using public corpora approximating the paper’s data mix.
- Reproduce headline metrics: perplexity on pretraining validation, zero-shot scores on reasoning benchmarks, long-context recall, and continual-learning forgetting curves.
- Provide complete experiment artefacts: configs, logs, checkpoints, evaluation scripts, and analysis notebooks.

---

## 2. Scope

### 2.1 In scope
- Data pipeline build-out (tokenization, sharding, streaming) for ≥100 B tokens.
- Distributed training scripts (FSDP/DeepSpeed) with logging + checkpointing.
- Evaluation harnesses for LM, QA/reasoning, long-context (NIAH variants), and continual-learning tasks.
- Ablations mirroring the paper (self-modifier toggles, CMS depth, optimizer variants, attention replacements).

### 2.2 Out of scope
- Non-HOPE architectures beyond the paper’s baselines (Transformer, TITAN, SAMBA, DeltaNet) except insofar as comparisons require reimplementation.
- Deployment/serving.
- Hyper-parameter sweeps beyond reproducing reported configs.

---

## 3. Data & Tokenization

| Component | Choice | Notes |
|-----------|--------|-------|
| Tokenizer | SentencePiece unigram 32k (shared with Stage 1) | Train on combined corpus below; manifest in `configs/data/refinedweb_mixture.yaml` |
| Base corpus | RefinedWeb / FineWeb proxy (≈600 B tokens) | Deduplicated, filtered for quality |
| Supplements | Books3 (if license permits), Stack/Code subset, Wikipedia, C4, RedPajama CC | Provide balanced mixture to mimic broad-domain data |
| Reasoning eval data | PIQA, HellaSwag, WinoGrande, ARC-E/C, SIQA, BoolQ | Use HF datasets; no training on eval splits |
| Long-context eval | Needles-in-a-Haystack (Passkey/Number/Word), PG19, NarrativeQA | Scripts to synthesize passkey tasks to 512k tokens |
| Continual tasks | Streaming Wikipedia by year + domain shift (news → code → conversations); synthetic permuted classes for stress-test | Track forgetting via accuracy drop on earlier segments |

Data pipeline tasks:
1. Acquire corpora (cc-by or permissible) into `data/raw/`.
2. Normalize & filter (language detection, length bounds, dedup).
3. Train tokenizer, store at `artifacts/tokenizer/spm_unigram_32k.model`.
4. Shard dataset into binary `.bin` or HF streaming format with 2048-token sequences + metadata.

---

## 4. Training Strategy

### 4.1 Model scales
| Name | Params | Layers | Dim | Heads | Sequence | Tokens |
|------|--------|--------|-----|-------|----------|--------|
| Pilot | 160 M | 12 | 512 | 8 | 2k | 3 B |
| Mid | 760 M | 24 | 1024 | 16 | 4k | 30 B |
| Target | 1.3 B | 32 | 1536 | 24 | 8k | 100 B |

Level schedules (example):
- TITAN level: update every 8/16/32 steps for Pilot/Mid/Target.
- CMS levels: {fast = 1, mid = 4, slow = 32, ultra = 128} update periods, gated by warmups.

### 4.2 Optimizers
- Outer weights: AdamW (β1=0.9, β2=0.95 for high LR stability), cosine decay, warmup 2k steps.
- Inner memories: DeepMomentum variants (preconditioned momentum for TITAN, DMGD for CMS).
- Gradient clipping: 1.0 outer, 0.3 inner deltas; update dropout 0.2 on self-mod outputs.

### 4.3 Distributed setup
- Framework: PyTorch FSDP (full-shard) or DeepSpeed ZeRO-3.
- Precision: BF16 activations/weights, FP32 master weights; optional FlashAttention for context.
- Checkpointing every 1k steps with partitioned state (model + optimizer + level clocks).
- Logging via WandB/MLflow with structured metrics (loss, ppl, level update magnitudes, memory norms).

### 4.4 Curriculum
1. Pilot run on 3 B tokens to validate pipeline, run ablations quickly (<=12 GPU-days).
2. Scale to 30 B tokens (760 M) once metrics stable; capture full eval suite.
3. Final 1.3 B / 100 B run with refined hyper-params and longer contexts (8k); integrate long-context tasks in training via mixture-of-lengths.

---

## 5. Evaluation Plan

### 5.1 Language Modeling
- Validation perplexity on held-out RefinedWeb shards plus WikiText-103.
- Log per-domain ppl to monitor forgetting when streaming.

### 5.2 Zero-shot Benchmarks
- Implement script `scripts/eval/zeroshot.py` pulling HF datasets (initial PIQA support, extend to others).
- Metrics: accuracy for PIQA/HellaSwag/WinoGrande/ARC-E/C/SIQA/BoolQ; match table from paper.

### 5.3 Long-context (NIAH)
- Generate custom sequences via `scripts/eval/niah.py` (currently scaffolds synthetic pass-key prompts); extend to context lengths up to 512k tokens.
- Evaluate recall accuracy vs. context length; compare HOPE vs. Transformer/TITAN baseline checkpoints.

### 5.4 Continual Learning
- Streaming tasks: sequential corpora (e.g., Wiki by year). After each segment, evaluate on all previous segments to compute average forgetting (Δ accuracy/perplexity). Use `scripts/eval/continual.py` with a segments YAML describing shard directories and checkpoint ordering.
- Use HOPE’s level stats to correlate update frequency with forgetting reduction.

### 5.5 Ablations
1. Self-modifier disabled.
2. CMS depth variations (k=1 vs. 3 vs. 4 levels).
3. Deep optimizer variants per level.
4. Attention backbone swap (full vs. sliding-window vs. DeltaNet).

Each ablation run uses Pilot scale unless specified; record metrics in `reports/ablations.md`.

---

## 6. Deliverables & Acceptance Criteria
| Deliverable | Criteria |
|-------------|----------|
| Data pipeline | Scripts in `scripts/data/`, tokenizer artefacts, documentation; reproducible shards |
| Training configs | Hydra YAMLs under `configs/hope/{pilot,mid,target}.yaml`; include optimizer, level schedules |
| Distributed training scripts | `train_dist.py`, launchers for FSDP/DeepSpeed with resume support |
| Evaluation suite | CLI tools for LM, zero-shot, NIAH, continual forgetting; CI test on small checkpoints |
| Reports | Markdown/Notebook summaries of metrics vs. baselines; highlight deviations |

---

## 7. Work Breakdown (Stage 2)
1. **Data Engineering** – ingest/filter/pack corpora; train tokenizer; unit tests for sharding.
2. **Infra & Configs** – Hydra config tree, logging integration, distributed launcher templates.
3. **Scaling Training** – pilot → mid → target runs; monitor; adjust hyper-params.
4. **Evaluation** – implement LM + zero-shot harness, NIAH generator, continual-learning scripts.
5. **Ablations & Analysis** – run targeted toggles, plot results, compare to paper.
6. **Documentation & Release** – write experiment logs, dataset README, reproduction checklists. Keep `docs/release_checklist.md` updated and treat it as the gate for tagging/publishing checkpoints.

Each workstream tracked via `TODO.md` or issue tracker; dependencies: (1) before (3/4), etc.

---

## 8. Timeline (indicative)
| Week | Milestone |
|------|-----------|
| 1 | Data pipeline + tokenizer complete; pilot configs ready |
| 2 | Pilot training + ablations; evaluation harness validated |
| 3–4 | Mid-scale (760 M) training + zero-shot/NIAH evals |
| 5–6 | Target (1.3 B) training, long-context + continual learning results |
| 7 | Ablations finalized, comparison vs. baselines, publish report |

---

## 9. Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Dataset licensing/availability | Stick to permissive corpora; document provenance |
| Compute instability at 100 B tokens | Use gradient checkpointing, monitor memory, schedule restarts |
| Eval drift vs. paper | Match prompt templates from Eleuther harness; verify tokenization alignment |
| Long-context efficiency | Integrate FlashAttention2 or block-sparse attention for >32k tokens |
| Continual learning metrics noisy | Average over multiple seeds; use bootstrapped confidence intervals |

---

## 10. Exit Criteria
- Matching (or within tolerance of) reported perplexity and zero-shot accuracy at 760 M and 1.3 B.
- Demonstrated long-context recall advantage over Transformer baseline at ≥256k tokens.
- Documented continual-learning improvements (reduced forgetting) with plots.
- All scripts/configs reproducible via `uv run` workflows; README updated with instructions.
