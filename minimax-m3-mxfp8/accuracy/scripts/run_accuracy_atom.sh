#!/usr/bin/env bash
# In-container runner: ATOM native server (nightly image) + GSM8K 5-shot.
# Env: ATOM_QR=INT4 enables AITER_QUICK_REDUCE_QUANTIZATION=INT4 (quick-reduce
# INT4 all-reduce); unset = baseline. Results tagged accordingly.
set -euo pipefail
cd "$(dirname "$0")"
source ./common.env

TAG=atom-base
if [ "${ATOM_QR:-}" = "INT4" ]; then
  export AITER_QUICK_REDUCE_QUANTIZATION=INT4
  TAG=atom-qrint4
fi
PORT=$ATOM_PORT
LOG="$WORK_DIR/atom_accuracy_$TAG.log"
RESULTS="$RESULT_DIR/gsm8k-$TAG-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS"
echo "[acc-atom] tag=$TAG serving $MODEL_PATH on :$PORT (TP=$TP)"

# Guard against stray listeners answering our health checks (burned once by a
# leftover ATOM server on the same port).
if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "[acc-atom] FATAL: port $PORT already answers /health before we serve"
  exit 1
fi

export ATOM_FORCE_ATTN_TRITON=1
python3 -m atom.entrypoints.openai_server \
  --model "$MODEL_PATH" \
  --tensor-parallel-size "$TP" \
  --server-port "$PORT" \
  --trust-remote-code \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --block-size "$BLOCK_SIZE" \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --kv_cache_dtype fp8 \
  --index-cache-dtype fp8 \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
  --no-enable_prefix_caching \
  >"$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "[acc-atom] waiting for server..."
for i in $(seq 1 720); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[acc-atom] server died:"; tail -30 "$LOG"; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "[acc-atom] readiness timeout"; exit 1; }
# Verify we're talking to OUR server (model id must match).
python3 - "$PORT" "$MODEL_PATH" <<'EOF'
import json, sys, urllib.request
port, want = sys.argv[1], sys.argv[2].rstrip("/")
with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30) as r:
    served = [m.get("id", "").rstrip("/") for m in json.loads(r.read()).get("data", [])]
if not any(want == s or want.endswith(s) or s.endswith(want) for s in served):
    raise SystemExit(f"[acc-atom] FATAL: port {port} serves {served}, not {want}")
print(f"[acc-atom] serving model verified: {served}")
EOF
echo "[acc-atom] server up after ~$((i * 5))s"

# lm_eval lives in an ephemeral uv env (not in the ATOM image either).
PY=$(command -v python3)
# note: bare `VAR=$(failing-cmd)` trips set -e before we can fall back
UV=$(command -v uv || ls /usr/local/bin/uv /usr/bin/uv /opt/venv/bin/uv 2>/dev/null | head -1) || true
if [ -z "$UV" ]; then
  echo "[acc-atom] uv not in image, installing to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  UV="$HOME/.local/bin/uv"
fi
[ -x "$UV" ] || { echo "[acc-atom] FATAL: uv unavailable"; exit 1; }
"$UV" run --no-project --python "$PY" \
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

echo "[acc-atom] done -> $RESULTS"
