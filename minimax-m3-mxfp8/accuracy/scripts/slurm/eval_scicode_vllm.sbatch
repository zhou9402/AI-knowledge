#!/bin/bash
# SciCode (official repo, inspect_ai integration) against vLLM (ac750 image)
# serving MXFP8 MiniMax-M3. Same enroot overlay as the NV-eval job
# (MINIMAX_GRAPH_METADATA is required for cudagraph capture).
#   sbatch --nodelist=vultr-mi355x-04 eval_scicode_vllm.sbatch
#SBATCH --job-name=m3-scicode-vllm
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/vfs/homes/peiyuanz/m3-compare/logs/scicode-vllm-%j.out

set -euo pipefail
WORK=/mnt/vfs/homes/peiyuanz/m3-compare
source "$WORK/tools/common.env"

env \
M3_IMAGE="$IMAGE_VLLM" \
M3_CONTAINER_PYTHONPATH="$WORK/tools" \
M3_ENROOT_CONFIG="$M3_ENROOT_CONF" \
M3_ENROOT_RUNTIME_PATH="/tmp/m3-compare-enroot-scicode-${SLURM_JOB_ID}" \
M3_USE_MINIMAX_GRAPH_METADATA_OVERLAY=1 \
    "$M3_HARNESS" /bin/bash -lc "SCICODE_PORT='${SCICODE_PORT:-8125}' SCICODE_TAG='${SCICODE_TAG:-x}' bash $WORK/tools/run_eval_scicode_vllm.sh"
