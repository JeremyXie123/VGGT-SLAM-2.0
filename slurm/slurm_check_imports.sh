#!/bin/bash
#SBATCH --job-name=vggt-slam-2.0-check
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Verifies the environment built by slurm_setup_env.sh.
#
# Set SBATCH_ACCOUNT to your Slurm account, then submit from a scratch
# directory that already contains ./logs:
#   export SBATCH_ACCOUNT=your-account
#   cd ~/scratch && sbatch ~/SLAM/VGGT-SLAM-2.0/slurm/slurm_check_imports.sh
#
# Sized from measured usage (1.4 GB, 20 s wall).
#
# Required checks failing exits nonzero. The run_os extras are optional and
# only reported, since their installs are best effort.

set -uo pipefail

JOB_INFO=$(scontrol show job -o "${SLURM_JOB_ID:?run via sbatch}")
SCRIPT_PATH="${JOB_INFO#*Command=}"
SCRIPT_PATH="${SCRIPT_PATH%% *}"
REPO=$(cd "${SCRIPT_PATH%/*}/.." && pwd)
cd "$REPO"

VENV="${VGGT_SLAM_VENV:-$HOME/scratch/venvs/vggt-slam-2.0}"
CACHE_HOME="${VGGT_SLAM_CACHE_HOME:-$HOME/scratch/cache}"
export TORCH_HOME="$CACHE_HOME/torch" HF_HOME="$CACHE_HOME/huggingface" XDG_CACHE_HOME="$CACHE_HOME/xdg"

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.14.0 python/3.11.5
source "$VENV/bin/activate"

python - <<'PY'
import importlib, sys

def check(label, fn, required=True):
    try:
        print(f"{label:22s} OK    {fn()}")
        return True
    except Exception as e:
        print(f"{label:22s} {'FAIL ' if required else 'SKIP '} {type(e).__name__}: {e}")
        return not required

def mod(name, attr="__file__"):
    return lambda: getattr(importlib.import_module(name), attr, None) or "OK"

import torch

def torch_info():
    assert torch.__version__.startswith("2.3.1"), f"torch is {torch.__version__}"
    return f"{torch.__version__} cuda {torch.version.cuda}"

def cuda_info():
    assert torch.cuda.is_available(), "no GPU visible"
    return torch.cuda.get_device_name(0)

# A real kernel launch. Extensions built for the wrong arch import fine and
# only fail here.
def cuda_kernel():
    a = torch.randn(256, 256, device="cuda")
    b = a @ a
    torch.cuda.synchronize()
    return f"matmul {tuple(b.shape)}"

# Constructing the LPIPS metric pulls its AlexNet weights, so a cache miss
# surfaces here rather than mid evaluation.
def lpips():
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
    return "AlexNet weights loaded"

# The VGGT-Omega import chain that main.py and solver.py depend on.
def omega_chain():
    from vggt_omega.models.vggt_omega import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_slam.vggt_omega_wrapper import VGGTOmegaModel
    return importlib.import_module("vggt_omega").__file__

print("=== required ===")
ok = all([
    check("torch", torch_info),
    check("cuda device", cuda_info),
    check("cuda kernel", cuda_kernel),
    check("gtsam", mod("gtsam")),
    check("viser", mod("viser", "__version__")),
    check("vggt_slam", mod("vggt_slam")),
    check("vggt (SPARK utils)", mod("vggt")),
    check("vggt_omega chain", omega_chain),
    check("torchmetrics LPIPS", lpips),
])

print("\n=== optional (run_os) ===")
for name in ("sam3", "perception_models", "core_vision", "pe", "timm"):
    check(name, mod(name), required=False)

print("\nRESULT:", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
PY
