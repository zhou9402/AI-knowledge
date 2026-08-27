#!/bin/bash
# NV-eval job: vLLM (ac750 image) + MXFP8 MiniMax-M3, GPQA-Diamond with
# NV-card sampling (temp=1.0, top_p=0.95, max_gen_toks=65536). Same overlay as
# the GSM8K accuracy job (MINIMAX_GRAPH_METADATA is required for cudagraph
# capture; see job 2616).
#   sbatch eval_nv_vllm.sbatch                                  # seed 0
#   sbatch --export=ALL,GPQA_SEED=2,GPQA_PORT=8212 eval_nv_vllm.sbatch
#SBATCH --job-name=m3-nveval-vllm
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/vfs/homes/peiyuanz/m3-compare/logs/nveval-vllm-%j.out

set -euo pipefail
WORK=/mnt/vfs/homes/peiyuanz/m3-compare
source "$WORK/tools/common.env"

env \
M3_IMAGE="$IMAGE_VLLM" \
M3_CONTAINER_PYTHONPATH="$WORK/tools" \
M3_ENROOT_CONFIG="$M3_ENROOT_CONF" \
M3_ENROOT_RUNTIME_PATH="/tmp/m3-compare-enroot-nveval-${SLURM_JOB_ID}" \
M3_USE_MINIMAX_GRAPH_METADATA_OVERLAY=1 \
    "$M3_HARNESS" /bin/bash -lc "GPQA_SEED=${GPQA_SEED:-0} GPQA_PORT=${GPQA_PORT:-8124} GPQA_TEMP=${GPQA_TEMP:-1.0} GPQA_MAX_TOKENS=${GPQA_MAX_TOKENS:-65536} bash $WORK/tools/${RUN_SCRIPT:-run_eval_nv_vllm.sh}"
