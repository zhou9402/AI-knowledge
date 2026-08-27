#!/usr/bin/env bash
# In-container runner: vLLM (upstream, ac750 image) serves MXFP8 MiniMax-M3,
# then lm_eval GSM8K 5-shot per recipes/MiniMax-M3.md (validated ATOM MXFP8
# reference: flexible-extract 0.9484 / strict-match 0.9477).
set -euo pipefail
cd "$(dirname "$0")"
source ./common.env

PORT=$VLLM_PORT
LOG="$WORK_DIR/vllm_accuracy_server.log"
RESULTS="$RESULT_DIR/gsm8k-clean-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS"

echo "[acc] serving $MODEL_PATH on :$PORT (TP=$TP)"
vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --trust-remote-code \
  --block-size "$BLOCK_SIZE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
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

echo "[acc] waiting for server..."
for i in $(seq 1 360); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[acc] server died:"; tail -40 "$LOG"; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "[acc] readiness timeout"; exit 1; }
echo "[acc] server up after ~$((i * 5))s"

# lm_eval is not in the image; pull it into an ephemeral uv environment.
echo "[acc] starting lm_eval gsm8k (5-shot, recipe settings)"
/usr/local/bin/uv run --no-project --python /usr/bin/python3 \
  --with "lm_eval[math,api]>=0.4.7" -- \
  lm_eval \
  --model local-chat-completions \
  --model_args "model=$MODEL_PATH,base_url=http://127.0.0.1:$PORT/v1/chat/completions,num_concurrent=32,max_gen_toks=16384" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size 65 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path "$RESULTS" \
  --log_samples \
  2>&1 | tee "$RESULTS/gsm8k_console.log"

echo "[acc] done -> $RESULTS"
