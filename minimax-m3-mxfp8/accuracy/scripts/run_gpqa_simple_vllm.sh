#!/usr/bin/env bash
# In-container runner: serve MXFP8 MiniMax-M3 with upstream vLLM (ac750 image)
# and run GPQA-Diamond with the simple-evals style prompt (explicit
# 'Answer: $LETTER' instruction) via gpqa_simple.py.
# Env: GPQA_PORT (default 8124), GPQA_SEED (default 0), GPQA_TEMP (default 1.0).
set -euo pipefail
cd "$(dirname "$0")"
source ./common.env

PORT="${GPQA_PORT:-8124}"
SEED="${GPQA_SEED:-0}"
TEMP="${GPQA_TEMP:-1.0}"
MAXTOK="${GPQA_MAX_TOKENS:-65536}"
LOG="$WORK_DIR/vllm_gpqa_simple_${PORT}.log"
GPQA_CSV="$WORK_DIR/extern/gpqa/gpqa_diamond.csv"
RESULTS="$RESULT_DIR/gpqa-simple-s${SEED}-t${TEMP}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS"
echo "[gpqa-simple] results -> $RESULTS"

[ -s "$GPQA_CSV" ] || { echo "[gpqa-simple] FATAL: $GPQA_CSV missing"; exit 1; }

echo "[gpqa-simple] serving $MODEL_PATH on :$PORT (TP=$TP)"
vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --trust-remote-code \
  --block-size "$BLOCK_SIZE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len 131072 \
  --enable-chunked-prefill \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs 128 \
  --language-model-only \
  --attention-backend TRITON_ATTN \
  --moe-backend aiter \
  --kv-cache-dtype fp8 \
  --no-enable-prefix-caching \
  >"$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "[gpqa-simple] waiting for server..."
for i in $(seq 1 720); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[gpqa-simple] server died:"; tail -40 "$LOG"; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "[gpqa-simple] readiness timeout"; exit 1; }
echo "[gpqa-simple] server up after ~$((i * 5))s"

python3 "$(dirname "$0")/gpqa_simple.py" \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --model "$MODEL_PATH" \
  --csv "$GPQA_CSV" \
  --seed "$SEED" \
  --temperature "$TEMP" \
  --top-p 0.95 \
  --max-tokens "$MAXTOK" \
  --concurrency 32 \
  --out "$RESULTS/gpqa_simple.json"

echo "[gpqa-simple] all done -> $RESULTS"
