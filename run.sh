#!/usr/bin/env bash
set -eu
set -o pipefail 2>/dev/null || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  APT_OPTS=(
    -o Acquire::Retries=5
    -o Acquire::ForceIPv4=true
    -o Acquire::http::No-Cache=true
    -o Acquire::http::Pipeline-Depth=0
  )

  retry_with_mirror_fallback() {
    local mirror_attempt=1
    local max_attempts=3
    
    while [ $mirror_attempt -le $max_attempts ]; do
      echo "Mirror attempt $mirror_attempt/$max_attempts..."
      
      # Try to run the command (apt-get install)
      if "$@" 2>&1 | tee /tmp/apt_output.log; then
        return 0
      fi
      
      # Check if libxcb-present0 is the issue
      if grep -q "libxcb-present0" /tmp/apt_output.log 2>/dev/null || grep -q "400  Bad Request" /tmp/apt_output.log 2>/dev/null; then
        echo "Detected libxcb-present0 mirror issue. Switching mirrors..."
        
        if [ $mirror_attempt -eq 1 ]; then
          echo "Trying deb.debian.org mirror..."
          sed -i 's|http://archive.ubuntu.com/ubuntu|http://deb.debian.org/ubuntu|g' /etc/apt/sources.list 2>/dev/null || true
        elif [ $mirror_attempt -eq 2 ]; then
          echo "Trying security.ubuntu.com mirror..."
          sed -i 's|http://deb.debian.org/ubuntu|http://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list 2>/dev/null || true
        fi
        
        apt-get clean >/dev/null 2>&1 || true
        apt-get update >/dev/null 2>&1 || true
      fi
      
      mirror_attempt=$((mirror_attempt + 1))
      
      if [ $mirror_attempt -le $max_attempts ]; then
        echo "Command failed. Switching strategy in 10 seconds..."
        sleep 10
      fi
    done
    
    return 1
  }

  retry_with_mirror_fallback apt-get "${APT_OPTS[@]}" install -y --no-install-recommends --fix-missing \
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

export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_RETRIES="${PIP_RETRIES:-10}"

python -m pip install --upgrade --retries "$PIP_RETRIES" --timeout "$PIP_DEFAULT_TIMEOUT" pip setuptools wheel
python -m pip install --retries "$PIP_RETRIES" --timeout "$PIP_DEFAULT_TIMEOUT" --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu"
python -m pip install --retries "$PIP_RETRIES" --timeout "$PIP_DEFAULT_TIMEOUT" -r requirements.txt

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
