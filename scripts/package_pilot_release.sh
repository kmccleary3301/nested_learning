#!/usr/bin/env bash
#
# Bundle the latest pilot checkpoint + metadata into artifacts/pilot_release/.
# Usage:
#   scripts/package_pilot_release.sh [hope_checkpoint_path] [titan_checkpoint_path]
# If no path is provided, the newest file under artifacts/checkpoints/pilot is used.

set -euo pipefail

RELEASE_DIR="artifacts/pilot_release"
CHECKPOINT_DIR="artifacts/checkpoints/pilot"
CONFIG_PATH="configs/pilot.yaml"
LOG_PATTERNS=( "logs/pilot_train*.log" "logs/pilot_train*.json" "logs/pilot_relaunch*.log" "logs/pilot_relaunch*.json" )
METADATA_PATH="${RELEASE_DIR}/metadata.json"
MANIFEST_PATH="${RELEASE_DIR}/MANIFEST.txt"
EVAL_PATTERNS=( "eval/*_pilot.json" "eval/*_titan.json" )
PLOT_PATTERNS=( "reports/plots/continual_pilot_*.png" "reports/plots/continual_titan_*.png" )

mkdir -p "${RELEASE_DIR}"

copy_sidecars() {
  local ckpt_path="$1"
  local dest_prefix="$2"
  local src_prefix="${ckpt_path%.pt}"
  local exts=("sha256" "meta.json" "yaml")
  for ext in "${exts[@]}"; do
    local src="${src_prefix}.${ext}"
    if [[ -f "${src}" ]]; then
      cp "${src}" "${dest_prefix}.${ext}"
    fi
  done
}

copy_patterns() {
  local dest_dir="$1"
  shift
  mkdir -p "${dest_dir}"
  shopt -s nullglob
  for pattern in "$@"; do
    for path in ${pattern}; do
      cp "${path}" "${dest_dir}/"
    done
  done
  shopt -u nullglob
}

if [[ $# -ge 1 ]]; then
  HOPE_CHECKPOINT="$1"
else
  HOPE_CHECKPOINT=$(ls -1t ${CHECKPOINT_DIR}/step_*.pt 2>/dev/null | head -n 1 || true)
fi

TITAN_CHECKPOINT="${2:-}"

if [[ -z "${HOPE_CHECKPOINT}" ]]; then
  echo "[package] No checkpoint found. Pass the path explicitly or ensure ${CHECKPOINT_DIR}/step_*.pt exists."
  exit 1
fi

HOPE_CHECKPOINT_BASENAME=$(basename "${HOPE_CHECKPOINT}")
DEST_CKPT="${RELEASE_DIR}/checkpoint.pt"
cp "${HOPE_CHECKPOINT}" "${DEST_CKPT}"
copy_sidecars "${HOPE_CHECKPOINT}" "${RELEASE_DIR}/checkpoint"

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

# Copy latest eval outputs / plots if present.
copy_patterns "${RELEASE_DIR}" "${EVAL_PATTERNS[@]}"
copy_patterns "${RELEASE_DIR}/plots" "${PLOT_PATTERNS[@]}"

TITAN_RELEASE_BASENAME=""
if [[ -n "${TITAN_CHECKPOINT}" ]]; then
  if [[ ! -f "${TITAN_CHECKPOINT}" ]]; then
    echo "[package] TITAN checkpoint not found: ${TITAN_CHECKPOINT}"
    exit 1
  fi
  TITAN_BASENAME=$(basename "${TITAN_CHECKPOINT}")
  TITAN_RELEASE_BASENAME="titan_${TITAN_BASENAME}"
  cp "${TITAN_CHECKPOINT}" "${RELEASE_DIR}/${TITAN_RELEASE_BASENAME}"
  copy_sidecars "${TITAN_CHECKPOINT}" "${RELEASE_DIR}/titan_${TITAN_BASENAME%.pt}"
fi

# Update metadata stub with checkpoint information if present
if [[ -f "${METADATA_PATH}" ]]; then
  python - "$HOPE_CHECKPOINT_BASENAME" "$TITAN_RELEASE_BASENAME" "$METADATA_PATH" <<'PY' || true
import json, sys, pathlib
ckpt = sys.argv[1]
titan = sys.argv[2]
path = pathlib.Path(sys.argv[3])
meta = json.loads(path.read_text())
meta["checkpoint_step"] = ckpt
if titan:
    meta["titan_checkpoint_step"] = titan
path.write_text(json.dumps(meta, indent=2))
PY
fi

# Emit manifest with quick reference info
{
  echo "Pilot Release Manifest"
  echo "======================"
  echo "HOPE Checkpoint: ${HOPE_CHECKPOINT_BASENAME}"
  if [[ -n "${TITAN_RELEASE_BASENAME}" ]]; then
    echo "TITAN Checkpoint: ${TITAN_RELEASE_BASENAME}"
  fi
  echo "Config: ${CONFIG_PATH}"
  echo "Logs copied from patterns: ${LOG_PATTERNS[*]}"
  echo "Eval copied from patterns: ${EVAL_PATTERNS[*]}"
  echo "Plots copied from patterns: ${PLOT_PATTERNS[*]}"
  date "+Packaged at: %Y-%m-%d %H:%M:%S"
} > "${MANIFEST_PATH}"

echo "[package] Release bundle updated:"
echo "  - ${DEST_CKPT}"
echo "  - ${RELEASE_DIR}/checkpoint.* (sidecars, when available)"
if [[ -n "${TITAN_RELEASE_BASENAME}" ]]; then
  echo "  - ${RELEASE_DIR}/${TITAN_RELEASE_BASENAME}"
  echo "  - ${RELEASE_DIR}/titan_${TITAN_BASENAME%.pt}.* (sidecars, when available)"
fi
echo "  - ${RELEASE_DIR}/config.yaml"
echo "  - ${LOG_DEST}/"
echo "  - ${RELEASE_DIR}/*_pilot.json and ${RELEASE_DIR}/*_titan.json (when available)"
echo "  - ${RELEASE_DIR}/plots/ (when available)"
echo "  - ${METADATA_PATH} (if present)"
