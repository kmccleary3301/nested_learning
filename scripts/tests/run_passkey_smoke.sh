#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="artifacts/checkpoints/pilot_smoke"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/step_000010.pt"
TOKENIZER="tests/data/tiny_tokenizer.model"
OUTPUT_JSON="eval/passkey_ci.json"

rm -rf "${CHECKPOINT_DIR}"

echo "[passkey-ci] training pilot_smoke for 10 steps"
uv run python train.py --config-name pilot_smoke

echo "[passkey-ci] running synthetic passkey eval with memorization"
uv run python scripts/eval/passkey.py \
  --config configs/pilot_smoke.yaml \
  --checkpoint "${CHECKPOINT_PATH}" \
  --tokenizer-path "${TOKENIZER}" \
  --samples 8 \
  --filler-sentences 32 \
  --device cpu \
  --output "${OUTPUT_JSON}" \
  --memorize \
  --memorize-steps 1

uv run python - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("eval/passkey_ci.json").read_text())
delta = data.get("accuracy_delta", 0.0)
if delta < 0:
    raise SystemExit(f"Memorization delta negative: {delta}")
print(f"[passkey-ci] Memorization delta OK ({delta:.3f})")
PY
