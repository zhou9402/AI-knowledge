#!/bin/bash
# AA-LCR variant job (see run_eval_aalcr_variant.sh for the variant env vars).
# Default: NO index-cache hf-overrides (later-added feature excluded).
#   sbatch --export=ALL,AALCR_TAG=noindexcache,AALCR_HF_OVERRIDES=none eval_aalcr_variant.sbatch
#SBATCH --job-name=m3-aalcr-var
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/vfs/homes/peiyuanz/m3-compare/logs/aalcr-var-%j.out

set -euo pipefail
WORK=/mnt/vfs/homes/peiyuanz/m3-compare
source "$WORK/tools/common.env"

env \
M3_IMAGE="$IMAGE_VLLM" \
M3_CONTAINER_PYTHONPATH="$WORK/tools" \
M3_ENROOT_CONFIG="$M3_ENROOT_CONF" \
M3_ENROOT_RUNTIME_PATH="/tmp/m3-compare-enroot-aalcrvar-${SLURM_JOB_ID}" \
M3_USE_MINIMAX_GRAPH_METADATA_OVERLAY=1 \
    "$M3_HARNESS" /bin/bash -lc "AALCR_TAG='${AALCR_TAG:-baseline}' AALCR_HF_OVERRIDES='${AALCR_HF_OVERRIDES:-}' AALCR_HF_OVERRIDES_FILE='${AALCR_HF_OVERRIDES_FILE:-}' AALCR_KV_DTYPE='${AALCR_KV_DTYPE:-fp8}' AALCR_M3_PORT='${AALCR_M3_PORT:-8340}' AALCR_JUDGE_PORT='${AALCR_JUDGE_PORT:-8341}' AALCR_REPEATS='${AALCR_REPEATS:-1}' bash $WORK/tools/run_eval_aalcr_variant.sh"
