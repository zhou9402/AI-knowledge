#!/bin/bash
# Accuracy job: vLLM (ac750 image) + MXFP8 MiniMax-M3, GSM8K 5-shot.
#   sbatch accuracy_vllm.sbatch
#SBATCH --job-name=m3-acc-vllm
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/vfs/homes/peiyuanz/m3-compare/logs/acc-vllm-%j.out

set -euo pipefail
WORK=/mnt/vfs/homes/peiyuanz/m3-compare
source "$WORK/tools/common.env"

# MINIMAX_GRAPH_METADATA overlay is required: pristine ac750 image asserts in
# minimax_m3/common/indexer.py during cudagraph capture (see logs of job 2616).
env \
M3_IMAGE="$IMAGE_VLLM" \
M3_CONTAINER_PYTHONPATH="$WORK/tools" \
M3_ENROOT_CONFIG="$M3_ENROOT_CONF" \
M3_ENROOT_RUNTIME_PATH="/tmp/m3-compare-enroot-acc-${SLURM_JOB_ID}" \
M3_USE_MINIMAX_GRAPH_METADATA_OVERLAY=1 \
    "$M3_HARNESS" /bin/bash -lc "bash $WORK/tools/run_accuracy_clean.sh"
