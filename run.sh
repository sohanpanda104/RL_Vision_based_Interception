#!/usr/bin/env bash
set -eu
set -o pipefail 2>/dev/null || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive

  retry() {
    n=0
    until "$@"; do
      n=$((n + 1))
      if [ "$n" -ge 3 ]; then
        return 1
      fi
      echo "Command failed. Retrying in 5 seconds..."
      sleep 5
    done
  }

  retry apt-get update -o Acquire::Retries=3
  retry apt-get install -y --no-install-recommends --fix-missing \
    ca-certificates \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libosmesa6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3 \
    python3-pip \
    python3-venv
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Please install Python 3.10+ and retry."
  exit 1
fi

python3 - <<'PY'
import sys

if sys.version_info < (3, 10):
    print("ERROR: Python 3.10+ is required.")
    raise SystemExit(1)
PY

if [ -d ".venv" ] && [ ! -f ".venv/bin/activate" ]; then
  rm -rf .venv
fi

python3 -m venv .venv

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu"
python -m pip install -r requirements.txt

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p outputs run_logs

if [ -f "stage1/models_3d_s2/panda_pick_final.zip" ] && [ -f "stage1/models_3d_s2/vecnormalize.pkl" ]; then
  (
    cd stage1
    python stage1.py eval
  ) | tee run_logs/stage1_eval.log
fi

(
  cd stage2
  python sim3d_stage2.py train \
    --timesteps "${STAGE2_TIMESTEPS:-2048}" \
    --n-envs "${STAGE2_N_ENVS:-1}" \
    --eval-freq "${STAGE2_EVAL_FREQ:-1024}" \
    --eval-episodes "${STAGE2_EVAL_EPISODES:-1}" \
    --no-progress-bar
) | tee run_logs/stage2_train.log

(
  cd stage2
  python sim3d_stage2.py eval \
    --episodes "${STAGE2_RECORD_EPISODES:-1}" \
    --model models_3d_s2/panda_s2_final.zip \
    --vecnorm models_3d_s2/vecnormalize.pkl
) | tee run_logs/stage2_eval.log

(
  cd stage2
  python test_sim3d_stage2.py \
    --episodes "${STAGE2_TEST_EPISODES:-1}" \
    --model models_3d_s2/panda_s2_final.zip \
    --vecnorm models_3d_s2/vecnormalize.pkl \
    --video-dir models_3d_s2/test_videos
) | tee run_logs/stage2_test.log

(
  cd stage2
  python plot_training_metrics.py \
    --eval-file models_3d_s2/eval/evaluations.npz \
    --progress-file models_3d_s2/progress.csv \
    --out-dir models_3d_s2/plots
) | tee run_logs/stage2_plots.log

{
  echo "Logs:"
  find run_logs -type f | sort
  echo
  echo "Artifacts:"
  find stage1 stage2 outputs -type f \
    \( -name '*.zip' -o -name '*.pkl' -o -name '*.mp4' -o -name '*.png' -o -name '*.json' -o -name '*.npz' \) \
    | sort
} | tee outputs/artifact_manifest.txt

echo "Pipeline complete. See outputs/artifact_manifest.txt"
