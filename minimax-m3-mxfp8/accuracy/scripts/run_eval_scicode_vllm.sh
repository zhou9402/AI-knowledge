#!/usr/bin/env bash
# In-container runner: serve MXFP8 MiniMax-M3 with upstream vLLM (ac750 image)
# on :8125, then run the OFFICIAL SciCode benchmark
# (https://github.com/scicode-bench/SciCode, eval/inspect_ai integration)
# against the OpenAI-compatible endpoint.
#
# Config (matches the NVIDIA card sampling settings):
#   split=test (65 main problems / 338 subproblems), with_background=True (NV-comparable),
#   mode=normal, temperature=1.0, top_p=0.95, max output tokens 65536.
#   Single epoch: the upstream integration keys results by step_id, so
#   multi-epoch runs collapse — run separate jobs per repeat instead.
#   SCICODE_PORT env picks the server port (default 8125).
set -euo pipefail
cd "$(dirname "$0")"
source ./common.env

PORT="${SCICODE_PORT:-8125}"
LOG="$WORK_DIR/vllm_scicode_server_${PORT}.log"
SCICODE_REPO="$WORK_DIR/extern/SciCode"
VENV="$WORK_DIR/venvs/scicode"
RESULTS="$RESULT_DIR/scicode-vllm-$(date +%Y%m%d-%H%M%S)-${SCICODE_TAG:-$$}"
mkdir -p "$RESULTS"
# Keep HF datasets cache on the shared, writable fs (container HOME may differ).
export HF_HOME="$WORK_DIR/.cache/hf"
echo "[scicode] results -> $RESULTS"

echo "[scicode] serving $MODEL_PATH on :$PORT (TP=$TP)"
vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name minimax-m3 \
  --tensor-parallel-size "$TP" \
  --trust-remote-code \
  --block-size "$BLOCK_SIZE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len 131072 \
  --enable-chunked-prefill \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs 128 \
  --language-model-only --reasoning-parser minimax_m3 \
  --attention-backend TRITON_ATTN \
  --moe-backend aiter \
  --kv-cache-dtype fp8 \
  --no-enable-prefix-caching \
  >"$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "[scicode] waiting for server..."
# Guard against stray listeners: nothing may answer before our server is up.
if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "[scicode] FATAL: port $PORT already answers before we serve"; exit 1
fi
for i in $(seq 1 720); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[scicode] server died:"; tail -40 "$LOG"; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "[scicode] readiness timeout"; exit 1; }
# Identity check: the answering server must be OUR process, alive right now.
kill -0 "$SERVER_PID" 2>/dev/null || { echo "[scicode] FATAL: our server died but :$PORT answers (stale listener)"; exit 1; }
python3 - "$PORT" <<'EOF'
import json, sys, urllib.request
with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/models", timeout=30) as r:
    ids = [m.get("id", "") for m in json.loads(r.read()).get("data", [])]
assert any("minimax-m3" in i for i in ids), f"wrong server on port: {ids}"
print(f"[scicode] serving identity verified: {ids}")
EOF
echo "[scicode] server up after ~$((i * 5))s"

# --- SciCode repo + eval env ------------------------------------------------
if [ ! -d "$SCICODE_REPO/.git" ]; then
  git clone --depth 1 https://github.com/scicode-bench/SciCode.git "$SCICODE_REPO"
fi
if [ ! -x "$VENV/bin/python" ]; then
  echo "[scicode] creating venv $VENV"
  /usr/local/bin/uv venv --python /usr/bin/python3 "$VENV"
  # datasets>=3.0: the resolver otherwise picks datasets 2.14.4, which is
  # broken with modern pyarrow (pa.PyExtensionType removed).
  /usr/local/bin/uv pip install --python "$VENV/bin/python" -e "$SCICODE_REPO" gdown "datasets>=3.0"
fi
source "$VENV/bin/activate"
python -c "import inspect_ai, scicode, h5py, scipy, numpy; print('[scicode] env OK')"

# --- numeric test targets (test_data.h5, ~1GB) -------------------------------
H5="$SCICODE_REPO/eval/data/test_data.h5"
if [ ! -s "$H5" ]; then
  echo "[scicode] downloading test_data.h5 from Google Drive"
  gdown 17G_k65N_6yFFZ2O-jQH00Lh6iaw3z-AW -O "$H5"
fi
ls -la "$H5"

# --- run the eval -------------------------------------------------------------
# scicode.py reads prompt templates via relative Path("../data", ...), so the
# working directory must be eval/inspect_ai.
cd "$SCICODE_REPO/eval/inspect_ai"
export VLLM_BASE_URL="http://127.0.0.1:$PORT/v1"
export VLLM_API_KEY=dummy
# inspect-ai resolved to 0.3.69 (era-matching for this repo), which has no
# "openai-api/<service>/<model>" provider; use the "openai" provider pointed
# at the vLLM endpoint via OPENAI_BASE_URL instead.
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export OPENAI_API_KEY=dummy
inspect eval scicode.py \
  --model openai/minimax-m3 \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-tokens 65536 \
  --max-connections 32 \
  -T split=test \
  -T output_dir="$RESULTS/scicode_output" \
  -T h5py_file="$H5" \
  -T with_background=True \
  -T mode=normal \
  --log-dir "$RESULTS/inspect_logs" \
  2>&1 | tee "$RESULTS/inspect.console.log"

echo "[scicode] all done -> $RESULTS"
