#!/bin/bash
#SBATCH --job-name=vggt-slam-scene
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Runs VGGT-SLAM 2.0 on one scene. Results land in repo/output/<scene>/.
#
# Set SBATCH_ACCOUNT to your Slurm account, then submit from a scratch
# directory that already contains ./logs:
#   export SBATCH_ACCOUNT=your-account
#   cd ~/scratch && sbatch --job-name=vggt-slam-fr2-desk \
#     ~/SLAM/VGGT-SLAM-2.0/slurm/slurm_run_scene.sh \
#     ~/scratch/SLAM/datasets/TUM_RGBD/rgbd_dataset_freiburg2_desk/rgb
#
# <image_folder> may be absolute or relative to the repo root. The scene name
# is its basename, or its parent's basename when it is named "rgb" (TUM
# layout).
#
# Set any of SUBMAP_SIZE, MIN_DISPARITY, CONF_THRESHOLD, LC_THRES, MAX_LOOPS
# or VIS_FLOW=1 to pass the matching main.py flag. Unset ones fall back to
# main.py's own defaults.
#
# The dataset must already be on disk. This script does not download it.
#
# Sized from a 859 frame TUM sequence (14.5 GB peak, 0.8 cores average,
# 2.5 min wall). Memory grows with the number of submaps, so a much longer
# sequence may need more than 24G. Check a finished job with `seff <jobid>`
# and raise --mem if Memory Efficiency approaches 100%.

set -euo pipefail

# sbatch runs a spooled copy of this script, so $BASH_SOURCE is useless. Ask
# SLURM for the submitted path instead.
JOB_INFO=$(scontrol show job -o "${SLURM_JOB_ID:?run via sbatch}")
SCRIPT_PATH="${JOB_INFO#*Command=}"
SCRIPT_PATH="${SCRIPT_PATH%% *}"
REPO=$(cd "${SCRIPT_PATH%/*}/.." && pwd)
cd "$REPO"

resolve() { [[ "$1" = /* ]] && echo "$1" || echo "$REPO/$1"; }
IMAGE_FOLDER=$(resolve "${1:?Usage: sbatch slurm/slurm_run_scene.sh <image_folder>}")
IMAGE_FOLDER="${IMAGE_FOLDER%/}"
SCENE="$(basename "$IMAGE_FOLDER")"
if [[ "$SCENE" == "rgb" ]]; then
    SCENE="$(basename "$(dirname "$IMAGE_FOLDER")")"
fi

# Must match slurm_setup_env.sh's defaults.
VENV="${VGGT_SLAM_VENV:-$HOME/scratch/venvs/vggt-slam-2.0}"
CACHE_HOME="${VGGT_SLAM_CACHE_HOME:-$HOME/scratch/cache}"
export TORCH_HOME="$CACHE_HOME/torch"
export HF_HOME="$CACHE_HOME/huggingface"
export XDG_CACHE_HOME="$CACHE_HOME/xdg"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME"

echo "========================================="
echo "Job:    $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Node:   $(hostname)"
echo "Scene:  $SCENE"
echo "Images: $IMAGE_FOLDER"
echo "Start:  $(date)"
echo "========================================="

# Must match the modules the venv was built against.
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.14.0 python/3.11.5
source "$VENV/bin/activate"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ ! -d "$IMAGE_FOLDER" ]]; then
    echo "ERROR: $IMAGE_FOLDER not found" >&2
    exit 1
fi
n_frames=$(find "$IMAGE_FOLDER" -maxdepth 1 -type f | wc -l)
if [[ "$n_frames" -eq 0 ]]; then
    echo "ERROR: $IMAGE_FOLDER is empty" >&2
    exit 1
fi
echo "==> Found $n_frames files in $IMAGE_FOLDER"

EXP_FOLDER="output/$SCENE"
mkdir -p "$EXP_FOLDER"

ARGS=(--image_folder "$IMAGE_FOLDER" --log_results --log_path "$EXP_FOLDER/poses.txt")
[[ -n "${SUBMAP_SIZE:-}" ]] && ARGS+=(--submap_size "$SUBMAP_SIZE")
[[ -n "${MIN_DISPARITY:-}" ]] && ARGS+=(--min_disparity "$MIN_DISPARITY")
[[ -n "${CONF_THRESHOLD:-}" ]] && ARGS+=(--conf_threshold "$CONF_THRESHOLD")
[[ -n "${LC_THRES:-}" ]] && ARGS+=(--lc_thres "$LC_THRES")
[[ -n "${MAX_LOOPS:-}" ]] && ARGS+=(--max_loops "$MAX_LOOPS")
[[ -n "${VIS_FLOW:-}" ]] && ARGS+=(--vis_flow)

echo "==> Running VGGT-SLAM"
python main.py "${ARGS[@]}"

echo "========================================="
echo "Results: $EXP_FOLDER"
echo "End: $(date)"
echo "========================================="
