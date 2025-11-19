#!/usr/bin/env bash
#
# Bundle the latest pilot checkpoint + metadata into artifacts/pilot_release/.
# Usage:
#   scripts/package_pilot_release.sh [checkpoint_path]
# If no path is provided, the newest file under artifacts/checkpoints/pilot is used.

set -euo pipefail

RELEASE_DIR="artifacts/pilot_release"
CHECKPOINT_DIR="artifacts/checkpoints/pilot"
CONFIG_PATH="configs/pilot.yaml"
LOG_PATTERNS=( "logs/pilot_train*.log" "logs/pilot_train*.json" "logs/pilot_relaunch*.log" "logs/pilot_relaunch*.json" )
METADATA_PATH="${RELEASE_DIR}/metadata.json"
MANIFEST_PATH="${RELEASE_DIR}/MANIFEST.txt"

mkdir -p "${RELEASE_DIR}"

if [[ $# -ge 1 ]]; then
  CHECKPOINT="$1"
else
  CHECKPOINT=$(ls -1t ${CHECKPOINT_DIR}/step_*.pt 2>/dev/null | head -n 1 || true)
fi

if [[ -z "${CHECKPOINT}" ]]; then
  echo "[package] No checkpoint found. Pass the path explicitly or ensure ${CHECKPOINT_DIR}/step_*.pt exists."
  exit 1
fi

CHECKPOINT_BASENAME=$(basename "${CHECKPOINT}")
DEST_CKPT="${RELEASE_DIR}/checkpoint.pt"
cp "${CHECKPOINT}" "${DEST_CKPT}"

# Copy config snapshot
cp "${CONFIG_PATH}" "${RELEASE_DIR}/config.yaml"

# Copy relevant logs (if they exist)
LOG_DEST="${RELEASE_DIR}/logs"
mkdir -p "${LOG_DEST}"
shopt -s nullglob
for pattern in "${LOG_PATTERNS[@]}"; do
  for log_path in ${pattern}; do
    cp "${log_path}" "${LOG_DEST}/"
  done
done
shopt -u nullglob

# Update metadata stub with checkpoint information if present
if [[ -f "${METADATA_PATH}" ]]; then
  python - "$CHECKPOINT_BASENAME" "$METADATA_PATH" <<'PY' || true
import json, sys, pathlib
ckpt = sys.argv[1]
path = pathlib.Path(sys.argv[2])
meta = json.loads(path.read_text())
meta["checkpoint_step"] = ckpt
path.write_text(json.dumps(meta, indent=2))
PY
fi

# Emit manifest with quick reference info
{
  echo "Pilot Release Manifest"
  echo "======================"
  echo "Checkpoint: ${CHECKPOINT_BASENAME}"
  echo "Config: ${CONFIG_PATH}"
  echo "Logs copied from patterns: ${LOG_PATTERNS[*]}"
  date "+Packaged at: %Y-%m-%d %H:%M:%S"
} > "${MANIFEST_PATH}"

echo "[package] Release bundle updated:"
echo "  - ${DEST_CKPT}"
echo "  - ${RELEASE_DIR}/config.yaml"
echo "  - ${LOG_DEST}/"
echo "  - ${METADATA_PATH} (if present)"
