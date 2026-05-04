#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

install_system_dependencies() {
  if ! command -v apt-get >/dev/null 2>&1; then
    log "apt-get not found; skipping Ubuntu package installation."
    return
  fi

  if [ "$(id -u)" -eq 0 ]; then
    APT_GET=(apt-get)
  elif command -v sudo >/dev/null 2>&1; then
    APT_GET=(sudo apt-get)
  else
    log "apt-get is available but this user is not root and sudo is missing."
    return 1
  fi

  export DEBIAN_FRONTEND=noninteractive
  log "Installing Ubuntu packages required for Python, PyBullet, Matplotlib, and video output."

  retry_apt() {
    local attempts=3
    local delay=2

    for ((attempt = 1; attempt <= attempts; attempt++)); do
      if "$@"; then
        return 0
      fi

      if [ "$attempt" -lt "$attempts" ]; then
        log "Command failed; retrying in ${delay}s (${attempt}/${attempts})."
        sleep "$delay"
      fi
    done

    return 1
  }

  retry_apt "${APT_GET[@]}" update -o Acquire::Retries=3
  retry_apt "${APT_GET[@]}" install -y --no-install-recommends --fix-missing \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libosmesa6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv
}

install_system_dependencies

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
RUN_ID="$(date '+%Y%m%d_%H%M%S')"
OUTPUT_DIR="outputs/${RUN_ID}"

mkdir -p "${OUTPUT_DIR}/logs"

log "Creating virtual environment at ${VENV_DIR}."
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required.")
print(f"Using Python {sys.version.split()[0]}")
PY

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

log "Upgrading packaging tools."
python -m pip install --upgrade pip setuptools wheel 2>&1 | tee "${OUTPUT_DIR}/logs/pip_bootstrap.log"

if [ "${SKIP_TORCH_CPU_INSTALL:-0}" != "1" ]; then
  log "Installing CPU-only PyTorch wheel."
  python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu" 2>&1 | tee "${OUTPUT_DIR}/logs/pip_torch.log"
fi

log "Installing Python dependencies from requirements.txt."
python -m pip install -r requirements.txt 2>&1 | tee "${OUTPUT_DIR}/logs/pip_requirements.log"

log "Verifying core imports."
python - <<'PY' 2>&1 | tee "${OUTPUT_DIR}/logs/import_check.log"
import gymnasium
import matplotlib
import numpy
import panda_gym
import pybullet
import stable_baselines3
import torch

print("gymnasium", gymnasium.__version__)
print("matplotlib", matplotlib.__version__)
print("numpy", numpy.__version__)
print("pybullet imported")
print("stable_baselines3", stable_baselines3.__version__)
print("torch", torch.__version__)
PY

RUN_STAGE1_EVAL="${RUN_STAGE1_EVAL:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"
STAGE2_TIMESTEPS="${STAGE2_TIMESTEPS:-4096}"
STAGE2_N_ENVS="${STAGE2_N_ENVS:-1}"
STAGE2_EVAL_FREQ="${STAGE2_EVAL_FREQ:-2048}"
STAGE2_EVAL_EPISODES="${STAGE2_EVAL_EPISODES:-1}"
STAGE2_RECORD_EPISODES="${STAGE2_RECORD_EPISODES:-1}"

if [ "${RUN_STAGE1_EVAL}" = "1" ]; then
  if [ -f "stage1/models_3d_s2/panda_pick_final.zip" ] && [ -f "stage1/models_3d_s2/vecnormalize.pkl" ]; then
    log "Running Stage 1 evaluation/video generation from the checked-in model."
    (
      cd stage1
      python stage1.py eval
    ) 2>&1 | tee "${OUTPUT_DIR}/logs/stage1_eval.log"
  else
    log "Stage 1 model artifacts were not found; skipping Stage 1 evaluation."
  fi
fi

if [ "${RUN_STAGE2}" = "1" ]; then
  log "Running Stage 2 training: timesteps=${STAGE2_TIMESTEPS}, n_envs=${STAGE2_N_ENVS}, eval_freq=${STAGE2_EVAL_FREQ}."
  python stage2/stage2.py train \
    --timesteps "${STAGE2_TIMESTEPS}" \
    --n-envs "${STAGE2_N_ENVS}" \
    --eval-freq "${STAGE2_EVAL_FREQ}" \
    --eval-episodes "${STAGE2_EVAL_EPISODES}" \
    --no-progress-bar 2>&1 | tee "${OUTPUT_DIR}/logs/stage2_train.log"

  log "Running Stage 2 evaluation/video generation."
  python stage2/stage2.py eval \
    --episodes "${STAGE2_RECORD_EPISODES}" \
    --model models_3d_s2/panda_s2_final.zip \
    --vecnorm models_3d_s2/vecnormalize.pkl 2>&1 | tee "${OUTPUT_DIR}/logs/stage2_eval.log"
fi

log "Writing artifact manifest."
{
  echo "Run output directory: ${OUTPUT_DIR}"
  echo
  echo "Logs:"
  find "${OUTPUT_DIR}/logs" -type f | sort
  echo
  echo "Generated or included artifacts:"
  for artifact_dir in stage1/models_3d_s2 models_3d_s2 stage2; do
    [ -d "${artifact_dir}" ] || continue
    find "${artifact_dir}" -type f \
      \( -name '*.zip' -o -name '*.pkl' -o -name '*.mp4' -o -name '*.png' -o -name '*.json' -o -name '*.npz' \)
  done | sort
} | tee "${OUTPUT_DIR}/artifact_manifest.txt"

log "Pipeline complete. See ${OUTPUT_DIR}/artifact_manifest.txt for saved outputs."
