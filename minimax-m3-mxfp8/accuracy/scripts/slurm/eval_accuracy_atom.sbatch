#!/bin/bash
# ATOM-side GSM8K accuracy, baseline or INT4 quick-reduce (ATOM_QR=INT4).
#   sbatch eval_accuracy_atom.sbatch                       # baseline
#   sbatch --export=ALL,ATOM_QR=INT4 eval_accuracy_atom.sbatch
#SBATCH --job-name=m3-acc-atom
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/vfs/homes/peiyuanz/m3-compare/logs/acc-atom-%j.out

set -euo pipefail
WORK=/mnt/vfs/homes/peiyuanz/m3-compare
source "$WORK/tools/common.env"

# ATOM native image; no vllm overlays apply. Fresh aiter JIT dir is REQUIRED
# (shared /mnt/scratch aiter-jit cache was built by a different aiter — stale
# .so files break pybind signatures).
env \
M3_IMAGE="$IMAGE_ATOM" \
M3_CONTAINER_PYTHONPATH="$WORK/tools" \
M3_ENROOT_CONFIG="$M3_ENROOT_CONF" \
M3_ENROOT_RUNTIME_PATH="/tmp/m3-compare-enroot-accatom-${SLURM_JOB_ID}" \
M3_AUDIT_ROOT="$WORK" \
M3_AITER_JIT_DIR=/mnt/scratch/peiyuanz/aiter-jit-m3c-atom \
M3_AITER_ROOT_DIR=/mnt/scratch/peiyuanz/aiter-root-m3c-atom \
    "$M3_HARNESS" /bin/bash -lc "ATOM_QR='${ATOM_QR:-}' ATOM_PORT='${ATOM_PORT:-8123}' bash $WORK/tools/run_accuracy_atom.sh"
