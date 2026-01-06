# Continual Classification Evaluation (CLINC / Banking77 / DBpedia14)

The Nested Learning paper highlights **class-incremental continual learning** in the text classification
domain (CLINC, Banking77, DBpedia). This repo provides a lightweight, implementation-first harness that
treats classification as **generative label selection**:

- Prompt: `Text: ... \nLabel:`
- Score each candidate label by log-probability of the label string
- Optionally apply HOPE/TITAN/CMS **test-time memorization** after each example (fast-state by default)

## Script

Use `scripts/eval/continual_classification.py`.

### Smoke run (CPU)

```bash
uv run python scripts/eval/continual_classification.py \
  --config configs/pilot_smoke.yaml \
  --checkpoint artifacts/checkpoints/pilot_smoke/step_000010.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --dataset clinc \
  --smoke \
  --device cpu \
  --output eval/continual_cls_smoke.json
```

### Memorization-enabled run

```bash
uv run python scripts/eval/continual_classification.py \
  --config configs/hope/pilot_attention.yaml \
  --checkpoint artifacts/checkpoints/pilot/step_230000.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --dataset banking77 \
  --task-size 10 --train-per-label 25 --eval-per-label 25 \
  --memorize --memorize-steps 1 \
  --memorize-paths titan,cms_fast \
  --memorize-surprise-threshold 0.02 \
  --device cuda:0 \
  --output eval/continual_cls_banking77.json
```

Notes:
- `--task-size` controls class increments (how many labels per task).
- `--memorize-no-reset` (default) keeps the fast-state across examples/tasks, matching a continual setting.
- For “pure baseline” continual evaluation, omit `--memorize`.

### Offline / local JSONL

If you don’t want to rely on HuggingFace downloads, supply a JSONL file:

```bash
uv run python scripts/eval/continual_classification.py \
  --config configs/pilot_smoke.yaml \
  --checkpoint artifacts/checkpoints/pilot_smoke/step_000010.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --local-jsonl data/local_continual_fixture.jsonl \
  --task-size 3 --train-per-label 2 --eval-per-label 2 \
  --smoke --device cpu \
  --output eval/continual_cls_local.json
```

Each line must be: `{"text": "...", "label": "..."}`.

## Output

The JSON contains:
- `task_accuracy_matrix[i][j]`: accuracy on task `i` evaluated after finishing task `j`
- `avg_accuracy_final`: average accuracy after the last task
- `avg_forgetting`: average (`max_acc_i - final_acc_i`) across tasks

This harness is intentionally lightweight so the community can refine the exact protocol to match the
paper’s class-incremental schedules and reporting conventions.

## Plotting

```bash
uv run python scripts/eval/plot_continual_classification.py \
  --continual-json eval/continual_cls_banking77.json \
  --output reports/plots/continual_cls_banking77.png
```
