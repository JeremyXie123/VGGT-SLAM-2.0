#!/bin/bash
#SBATCH --job-name=vggt-slam-2.0-setup
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
#
# Builds the VGGT-SLAM 2.0 environment on Killarney.
#
# Set SBATCH_ACCOUNT to your Slurm account, then submit from a scratch
# directory that already contains ./logs:
#   export SBATCH_ACCOUNT=your-account
#   cd ~/scratch && mkdir -p logs && sbatch ~/SLAM/VGGT-SLAM-2.0/slurm/slurm_setup_env.sh
#
# Override defaults with VGGT_SLAM_VENV, VGGT_SLAM_CACHE_HOME,
# VGGT_SLAM_SCRATCH, or VGGT_OMEGA_CKPT.
#
# Sized from measured usage (4.8 GB, 0.35 cores average, 6 min wall).
# The time allows for a first run that also downloads the 4.6 GB checkpoint.
#
# No GPU is requested: nothing here launches a CUDA kernel (compiling
# extensions needs nvcc, not a device). Verification runs separately in
# slurm_check_imports.sh, which does ask for one.
#
# Conda is not permitted on Alliance clusters, so this uses module load plus a
# virtualenv. Most packages come from PyPI rather than the Alliance wheelhouse,
# which carries only torch 2.13 (too new for the torch 2.3.1 this repo pins)
# and has no gtsam or viser 0.2.23 build.


set -euo pipefail

# sbatch runs a spooled copy of this script, so $BASH_SOURCE is useless. Ask
# SLURM for the submitted path instead.
JOB_INFO=$(scontrol show job -o "${SLURM_JOB_ID:?run via sbatch}")
SCRIPT_PATH="${JOB_INFO#*Command=}"
SCRIPT_PATH="${SCRIPT_PATH%% *}"
REPO=$(cd "${SCRIPT_PATH%/*}/.." && pwd)
cd "$REPO"

VENV="${VGGT_SLAM_VENV:-$HOME/scratch/venvs/vggt-slam-2.0}"
CKPT="${VGGT_OMEGA_CKPT:-$HOME/scratch/checkpoints/vggt-omega/vggt_omega_1b_512.pt}"
SCRATCH_ROOT="${VGGT_SLAM_SCRATCH:-$HOME/scratch/SLAM/VGGT-SLAM-2.0}"
CACHE_HOME="${VGGT_SLAM_CACHE_HOME:-$HOME/scratch/cache}"
export TORCH_HOME="$CACHE_HOME/torch"
export HF_HOME="$CACHE_HOME/huggingface"
export XDG_CACHE_HOME="$CACHE_HOME/xdg"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME"

echo "========================================="
echo "Job:    $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Node:   $(hostname)"
echo "Repo:   $(pwd)"
echo "Venv:   $VENV"
echo "Start:  $(date)"
echo "========================================="

# Modules. opencv must load before python: the wheelhouse opencv-python is a
# dummy wheel that only re-exposes the module's cv2.
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.14.0 python/3.11.5
module list

# Cover both GPU generations here (L40S sm_89, H100 sm_90).
export TORCH_CUDA_ARCH_LIST="8.9;9.0"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-4}"
export CUDA_HOME="${CUDA_HOME:-$EBROOTCUDA}"

if [[ -d "$VENV" ]]; then
    echo "Removing existing venv at $VENV"
    rm -rf "$VENV"
fi
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel

# gtsam-develop provides the SL(4) manifold support. Its cp311 builds are
# prereleases, and this cluster's pip advertises only the bare linux_x86_64
# platform tag, so it rejects the manylinux wheel. Fetch and unpack it with
# `installer`, which has no platform-tag gate. The dist-info it leaves behind
# makes the requirements.txt install below treat gtsam-develop as satisfied.
echo "==> Installing gtsam-develop"
pip install installer -q
GTSAM_WHEEL_URL=$(curl -sL --max-time 30 https://pypi.org/simple/gtsam-develop/ \
    | grep -oE 'href="[^"]*cp311-cp311-manylinux[^"]*x86_64\.whl[^"]*"' \
    | sed -E 's/^href="([^#"]*).*/\1/' \
    | tail -1)
if [[ -z "$GTSAM_WHEEL_URL" ]]; then
    echo "ERROR: no cp311 manylinux x86_64 wheel found for gtsam-develop" >&2
    exit 1
fi
GTSAM_WHEEL_FILE="/tmp/$(basename "$GTSAM_WHEEL_URL")"
curl -sL --max-time 120 -o "$GTSAM_WHEEL_FILE" "$GTSAM_WHEEL_URL"
python -m installer "$GTSAM_WHEEL_FILE"
rm -f "$GTSAM_WHEEL_FILE"

echo "==> Installing requirements.txt"
pip install -r requirements.txt

echo "==> Installing evo and torchmetrics for the evals scripts"
pip install evo
pip install --no-index torchmetrics

mkdir -p third_party

clone_and_install () {
    local url="$1" dir="$2"
    if [[ -d "third_party/$dir/.git" ]]; then
        echo "==> third_party/$dir already cloned"
    else
        echo "==> Cloning $url into third_party/$dir"
        git clone "$url" "third_party/$dir"
    fi
    pip install -e "./third_party/$dir"
}

echo "==> Salad"
clone_and_install https://github.com/Dominic101/salad.git salad

# VGGT_SPARK is no longer the backbone, but vggt.utils.geometry and
# vggt.utils.pose_enc are still imported by solver.py.
echo "==> VGGT (MIT-SPARK fork)"
clone_and_install https://github.com/MIT-SPARK/VGGT_SPARK.git vggt

echo "==> VGGT-Omega"
clone_and_install https://github.com/facebookresearch/vggt-omega.git vggt_omega

# Perception Encoder and SAM 3 are only used by main.py's run_os flag. Both
# pin numpy 2.1.2, which this cluster cannot install, so they are best effort.
echo "==> Perception Encoder (optional)"
clone_and_install https://github.com/facebookresearch/perception_models.git perception_models \
    || echo "WARNING: Perception Encoder install failed, run_os will not work"

echo "==> SAM 3 (optional)"
clone_and_install https://github.com/facebookresearch/sam3.git sam3 \
    || echo "WARNING: SAM 3 install failed, run_os will not work"

# sam3 needs pkg_resources (dropped in setuptools 81), triton, and pycocotools,
# none of which its own dependencies pull in.
echo "==> Installing sam3 runtime dependencies"
pip install "setuptools<81" "triton==2.3.0" pycocotools

# Results live on scratch because runs are large and regenerable. .gitignore
# matches `output` without a trailing slash on purpose, since git treats the
# symlink as a file.
echo "==> Setting up results directory"
mkdir -p "$SCRATCH_ROOT/output" "$SCRATCH_ROOT/logs"
for link in output logs; do
    if [[ -L "$link" ]]; then
        echo "    $link already linked to $(readlink "$link")"
    elif [[ -e "$link" ]]; then
        echo "    WARNING: $link exists and is not a symlink, leaving it alone" >&2
    else
        ln -s "$SCRATCH_ROOT/$link" "$link"
        echo "    $link linked to $SCRATCH_ROOT/$link"
    fi
done
mkdir -p slurm/logs

# facebook/VGGT-Omega is a gated repo, so the download needs an approved token.
echo "==> VGGT-Omega checkpoint"
if [[ -f "$CKPT" ]]; then
    echo "    Present: $CKPT ($(du -h "$CKPT" | cut -f1))"
else
    echo "    Missing, downloading to $CKPT"
    mkdir -p "$(dirname "$CKPT")"
    pip install -q "huggingface_hub[cli]"
    if ! python - "$CKPT" <<'PY_CKPT'
import os, shutil, sys
from huggingface_hub import hf_hub_download
dest = sys.argv[1]
path = hf_hub_download(
    repo_id="facebook/VGGT-Omega",
    filename="vggt_omega_1b_512.pt",
    cache_dir=os.path.join(os.path.dirname(dest), ".cache"),
)
if os.path.abspath(path) != os.path.abspath(dest):
    shutil.copyfile(path, dest)
print("downloaded to", dest)
PY_CKPT
    then
        echo "" >&2
        echo "ERROR: could not download the VGGT-Omega checkpoint." >&2
        echo "  1. Request access at https://huggingface.co/facebook/VGGT-Omega" >&2
        echo "  2. Authenticate: huggingface-cli login (or export HF_TOKEN=...)" >&2
        echo "  3. Re-run, or set VGGT_OMEGA_CKPT if you have the file already." >&2
        exit 1
    fi
fi

echo "==> Installing vggt_slam (editable)"
pip install -e .

echo "========================================="
echo "Environment ready: $VENV"
echo "Checkpoint:        $CKPT"
echo "Results root:      $SCRATCH_ROOT/output"
echo
echo "Use it in a job script with:"
echo "  module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.14.0 python/3.11.5"
echo "  source $VENV/bin/activate"
echo
echo "Verify it with:"
echo "  cd ~/scratch && sbatch $REPO/slurm/slurm_check_imports.sh"
echo "End: $(date)"
echo "========================================="
