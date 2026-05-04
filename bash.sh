#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE2_DIR="$ROOT_DIR/stage2"
EPISODES="${1:-3}"

if [[ -x "$ROOT_DIR/.venv/Scripts/python.exe" ]]; then
  PYTHON="$ROOT_DIR/.venv/Scripts/python.exe"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
else
  PYTHON="python"
fi

cd "$STAGE2_DIR"
export PYTHONIOENCODING=utf-8

"$PYTHON" test_sim3d_stage2.py \
  --episodes "$EPISODES" \
  --model models_3d_s2/best/best_model.zip \
  --vecnorm models_3d_s2/vecnormalize.pkl
