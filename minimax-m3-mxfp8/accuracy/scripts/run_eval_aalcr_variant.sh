#!/usr/bin/env bash
# AA-LCR config-variant job: serve MiniMax-M3 with VARIANT flags, generate,
# then serve Qwen3-235B judge and score — all in one 8-GPU job (sequential TP4).
# Variants via env:
#   AALCR_HF_OVERRIDES  (default '{"text_config":{"use_index_cache":true,"index_topk_freq":4}}';
#                        set to '{}' to disable the index cache)
#   AALCR_KV_DTYPE      (default fp8; set 'auto' for bf16 KV)
#   AALCR_TAG           (results dir suffix, default from variant)
set -euo pipefail
cd "$(dirname "$0")"
source ./common.env

if [ -n "${AALCR_HF_OVERRIDES_FILE:-}" ]; then
  HF_OV=$(cat "$AALCR_HF_OVERRIDES_FILE")
elif [ -z "${AALCR_HF_OVERRIDES+x}" ]; then
  # default: upstream behavior, NO index cache (later-added feature excluded)
  HF_OV=none
else
  HF_OV="$AALCR_HF_OVERRIDES"   # "none" -> drop --hf-overrides entirely
fi
[ -z "$HF_OV" ] && HF_OV=none
KV="${AALCR_KV_DTYPE:-fp8}"
TAG="${AALCR_TAG:-baseline}"
RESULTS="$RESULT_DIR/aa-lcr-variant-$TAG-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS"
echo "[aalcr-variant] tag=$TAG kv=$KV hf_overrides=$HF_OV"
echo "[aalcr-variant] results -> $RESULTS"

JUDGE_MODEL=/mnt/vfs/homes/peiyuanz/models/Qwen3-235B-A22B-Instruct-2507-FP8
M3_PORT="${AALCR_M3_PORT:-8340}"
JUDGE_PORT="${AALCR_JUDGE_PORT:-8341}"

wait_health() { # port -> blocks until healthy
  for i in $(seq 1 720); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0
    sleep 5
  done
  return 1
}

HF_ARGS=()
if [ "$HF_OV" != "none" ]; then
  HF_ARGS=(--hf-overrides "$HF_OV")
fi

echo "[aalcr-variant] serving M3 (TP=$TP) on :$M3_PORT"
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 --port "$M3_PORT" \
  --tensor-parallel-size "$TP" \
  --trust-remote-code \
  --block-size "$BLOCK_SIZE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len 196608 \
  --enable-chunked-prefill \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs 32 \
  --language-model-only \
  --attention-backend TRITON_ATTN \
  --moe-backend aiter \
  --kv-cache-dtype "$KV" \
  --no-enable-prefix-caching \
  --served-model-name minimax-m3 \
  "${HF_ARGS[@]}" \
  >"$WORK_DIR/vllm_aalcr_variant_$TAG.log" 2>&1 &
M3_PID=$!
cleanup() { kill "$M3_PID" "${JUDGE_PID:-}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT

wait_health "$M3_PORT" || { echo "[aalcr-variant] M3 server failed"; tail -30 "$WORK_DIR/vllm_aalcr_variant_$TAG.log"; exit 1; }
echo "[aalcr-variant] M3 up, generating..."
python3 "$(dirname "$0")/aa_lcr.py" generate \
  --base-url "http://127.0.0.1:$M3_PORT/v1" \
  --model minimax-m3 \
  --data-dir "$WORK_DIR/extern/aa-lcr" \
  --out "$RESULTS/responses.jsonl" \
  --repeats "${AALCR_REPEATS:-1}" --concurrency 4 --max-tokens 32768 --temperature 1.0 --top-p 0.95
kill "$M3_PID" 2>/dev/null || true; wait "$M3_PID" 2>/dev/null || true
echo "[aalcr-variant] generation done, serving judge..."
sleep 30

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "$JUDGE_MODEL" \
  --host 127.0.0.1 --port "$JUDGE_PORT" \
  --tensor-parallel-size "$TP" --gpu-memory-utilization 0.9 \
  --max-model-len 32768 --no-enable-prefix-caching \
  --served-model-name judge \
  >"$WORK_DIR/vllm_aalcr_judge_variant_$TAG.log" 2>&1 &
JUDGE_PID=$!
wait_health "$JUDGE_PORT" || { echo "[aalcr-variant] judge server failed"; tail -30 "$WORK_DIR/vllm_aalcr_judge_variant_$TAG.log"; exit 1; }
echo "[aalcr-variant] judge up, judging..."
python3 "$(dirname "$0")/aa_lcr.py" judge \
  --base-url "http://127.0.0.1:$JUDGE_PORT/v1" \
  --model judge \
  --in "$RESULTS/responses.jsonl" \
  --out "$RESULTS/judged.jsonl"

echo "[aalcr-variant] all done -> $RESULTS"
