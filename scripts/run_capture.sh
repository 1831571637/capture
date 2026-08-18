#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CAPTURE_CONFIG:-${PROJECT_ROOT}/config/lab_5090.env}"
CHECK_ONLY=false
STOP_FIRST=true
APP_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_capture.sh [options] [-- app arguments]

Options:
  --config PATH  Load a different shell-format environment profile.
  --check        Validate configuration without stopping processes or touching hardware.
  --no-stop      Do not call stop_capture_app.sh before a real launch.
  -h, --help     Show this help.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --config)
      if [[ "$#" -lt 2 ]]; then
        echo "[capture-runner] ERROR: --config requires a path" >&2
        exit 2
      fi
      CONFIG_FILE="$2"
      shift 2
      ;;
    --check)
      CHECK_ONLY=true
      shift
      ;;
    --no-stop)
      STOP_FIRST=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      APP_ARGS+=("$@")
      break
      ;;
    *)
      echo "[capture-runner] ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[capture-runner] ERROR: configuration file not found: $CONFIG_FILE" >&2
  echo "[capture-runner] Run: cp config/lab_5090.env.example config/lab_5090.env" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

if [[ -z "${LEROBOT_ROOT:-}" ]]; then
  echo "[capture-runner] ERROR: LEROBOT_ROOT is empty in $CONFIG_FILE" >&2
  exit 2
fi
if [[ ! -d "${LEROBOT_ROOT}/src/lerobot" || ! -f "${LEROBOT_ROOT}/tools/bi_x5_capture_app.py" ]]; then
  echo "[capture-runner] ERROR: LEROBOT_ROOT is not a compatible checkout: ${LEROBOT_ROOT}" >&2
  echo "[capture-runner] Expected src/lerobot/ and tools/bi_x5_capture_app.py" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
echo "[capture-runner] config=${CONFIG_FILE}"
echo "[capture-runner] lerobot=${LEROBOT_ROOT} conda_env=${CONDA_ENV:-lerobot51}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[capture-runner] ERROR: conda is not available on PATH" >&2
  exit 2
fi
conda run --no-capture-output -n "${CONDA_ENV:-lerobot51}" \
  python scripts/preflight_capture.py

if [[ "$CHECK_ONLY" == "true" ]]; then
  echo "[capture-runner] configuration check only; no process will be stopped and no hardware will be opened"
  DRY_RUN=true scripts/capture_realman_x5_force_aligned_app.sh "${APP_ARGS[@]}"
  exit 0
fi

if [[ "$STOP_FIRST" == "true" ]]; then
  scripts/stop_capture_app.sh || true
fi

exec scripts/capture_realman_x5_force_aligned_app.sh "${APP_ARGS[@]}"
