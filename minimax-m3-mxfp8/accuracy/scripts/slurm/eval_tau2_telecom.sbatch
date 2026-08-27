#!/bin/bash
# tau2-Telecom job: MiniMax-M3 (agent) + Qwen3-235B (user-sim), one node, 8 GPUs.
# Requires the Qwen judge/user-sim model to be downloaded first (see site.env
# SIM_MODEL_PATH). Submit:  sbatch eval_tau2_telecom.sbatch
#SBATCH --job-name=m3-tau2-vllm
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/vfs/homes/peiyuanz/m3-compare/logs/tau2-vllm-%j.out

set -euo pipefail
WORK=/mnt/vfs/homes/peiyuanz/m3-compare
source "$WORK/tools/common.env"

env \
M3_IMAGE="$IMAGE_VLLM" \
M3_CONTAINER_PYTHONPATH="$WORK/tools" \
M3_ENROOT_CONFIG="$M3_ENROOT_CONF" \
M3_ENROOT_RUNTIME_PATH="/tmp/m3-compare-enroot-tau2-${SLURM_JOB_ID}" \
M3_USE_MINIMAX_GRAPH_METADATA_OVERLAY=1 \
    "$M3_HARNESS" /bin/bash -lc "bash $WORK/tools/run_tau2_telecom.sh"
