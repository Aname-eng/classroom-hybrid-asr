#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_EXE="${PYTHON_EXE:-$ROOT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "[ERROR] Python virtual environment not found at: $PYTHON_EXE" >&2
  exit 1
fi

exec "$PYTHON_EXE" "$ROOT_DIR/main.py" "$@"
