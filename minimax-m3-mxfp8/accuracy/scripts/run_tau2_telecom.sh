#!/usr/bin/env bash
# In-container runner: tau2-bench telecom domain with BOTH endpoints local.
#   - agent under test: MiniMax-M3 MXFP8 (vLLM, TP=4, port 8330)
#   - user simulator:   Qwen3-235B-A22B-Instruct-2507-FP8 (vLLM, TP=4, port 8331)
# Grading is programmatic (DB end-state assertions) — no judge model needed.
# NV-card sampling for the agent: temperature=1.0, top_p=0.95.
set -euo pipefail
cd "$(dirname "$0")"
source ./common.env

M3_PORT=8330
SIM_PORT=8331
SIM_MODEL="${SIM_MODEL_PATH:-/mnt/vfs/homes/peiyuanz/models/Qwen3-235B-A22B-Instruct-2507-FP8}"
RESULTS="$RESULT_DIR/tau2-telecom-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS"
echo "[tau2] results -> $RESULTS"

serve() { # serve <gpu_ids> <model> <port> <log> <maxlen> <extra...>
  local gpus=$1 model=$2 port=$3 log=$4 maxlen=$5; shift 5
  CUDA_VISIBLE_DEVICES="$gpus" vllm serve "$model" --host 127.0.0.1 --port "$port" \
    --tensor-parallel-size "$TP" --gpu-memory-utilization 0.85 \
    --max-model-len "$maxlen" --no-enable-prefix-caching "$@" \
    >"$log" 2>&1 &
  echo $!
}

# M3 gets the exact eval flag set; Qwen is a stock serve.
# Disjoint GPU halves on the 8-GPU node: M3 on 0-3, user-sim on 4-7.
M3_PID=$(serve 0,1,2,3 "$MODEL_PATH" "$M3_PORT" "$WORK_DIR/vllm_tau2_m3.log" 32768 \
  --trust-remote-code --block-size "$BLOCK_SIZE" --enable-chunked-prefill \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" --max-num-seqs 64 \
  --language-model-only --attention-backend TRITON_ATTN --moe-backend aiter \
  --kv-cache-dtype fp8 --served-model-name minimax-m3 \
  --enable-auto-tool-choice --tool-call-parser minimax_m3)
SIM_PID=$(serve 4,5,6,7 "$SIM_MODEL" "$SIM_PORT" "$WORK_DIR/vllm_tau2_sim.log" 32768 \
  --served-model-name user-sim \
  --enable-auto-tool-choice --tool-call-parser hermes)
cleanup() { kill "$M3_PID" "$SIM_PID" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT

for port in $M3_PORT $SIM_PORT; do
  echo "[tau2] waiting for :$port..."
  for i in $(seq 1 720); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
    sleep 5
  done
  curl -sf "http://127.0.0.1:$port/health" >/dev/null || { echo "[tau2] :$port readiness timeout"; exit 1; }
done
echo "[tau2] both servers up"

TAU2_DIR="$WORK_DIR/extern/tau2-bench"
cd "$TAU2_DIR"
export OPENAI_API_KEY=dummy
export LITELLM_DROP_PARAMS=1
uv run tau2 run \
  --domain telecom \
  --task-split-name base \
  --agent-llm openai/minimax-m3 \
  --agent-llm-args "{\"api_base\": \"http://127.0.0.1:$M3_PORT/v1\", \"temperature\": 1.0, \"top_p\": 0.95, \"max_tokens\": 4096}" \
  --user-llm openai/user-sim \
  --user-llm-args "{\"api_base\": \"http://127.0.0.1:$SIM_PORT/v1\", \"temperature\": 0.0}" \
  --num-trials "${TAU2_TRIALS:-1}" \
  --max-steps 200 \
  --save-to "$RESULTS/tau2_runs" \
  2>&1 | tee "$RESULTS/tau2.console.log"

echo "[tau2] all done -> $RESULTS"
