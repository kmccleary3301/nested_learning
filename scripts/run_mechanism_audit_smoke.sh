#!/usr/bin/env bash
set -euo pipefail

LOG_PATH="${LOG_PATH:-logs/mechanism_audit_smoke.json}"
CADENCE_OUT="${CADENCE_OUT:-reports/cadence_mechanism_audit_smoke.json}"
COMPLIANCE_OUT="${COMPLIANCE_OUT:-reports/compliance_mechanism_audit_smoke.json}"

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

uv run python train.py --config-name pilot_paper_faithful \
  train.steps=1 \
  train.device=cpu \
  train.online_chunk_size=9 \
  data.source=synthetic \
  +data.vocab_size=32000 \
  data.seq_len=9 \
  +data.dataset_size=8 \
  data.batch_size=1 \
  data.num_workers=0 \
  logging.enabled=true \
  logging.backend=json \
  logging.path="${LOG_PATH}"

uv run python scripts/checks/verify_update_cadence.py \
  --log-path "${LOG_PATH}" \
  --metric-prefix layer0.cms.cms_mid \
  --total-tokens 8 \
  --update-period 4 \
  --output "${CADENCE_OUT}"

uv run python scripts/checks/compliance_report.py \
  --config configs/pilot.yaml \
  --cadence-report "${CADENCE_OUT}" \
  --output "${COMPLIANCE_OUT}"

echo "[mechanism-audit-smoke] completed: ${LOG_PATH} + ${CADENCE_OUT} + ${COMPLIANCE_OUT}"
