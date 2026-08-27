#!/usr/bin/env bash
# Collect all available eval scores into one table. Run on the login node:
#   bash 50_collect_results.sh
set -uo pipefail
WORK=/mnt/vfs/homes/peiyuanz/m3-compare

echo "# MiniMax-M3 MXFP8 on MI355X (vLLM ac750 + repo@417debf overlay) — NV-card eval set"
echo
printf '%-28s %-22s %s\n' "Benchmark" "Score" "Source"
printf '%-28s %-22s %s\n' "---------" "-----" "------"

# GSM8K (job 2637, earlier accuracy run)
g=$(find "$WORK/results" -path "*gsm8k-vllm*" -name "gsm8k_console.log" 2>/dev/null | sort | tail -1)
if [ -n "$g" ]; then
  flex=$(grep -oP 'flexible-extract\|.*?\|  \|0\.\d+' "$g" | grep -oP '0\.\d+' | head -1)
  strict=$(grep -oP 'strict-match\|.*?\|  \|0\.\d+' "$g" | grep -oP '0\.\d+' | head -1)
  printf '%-28s %-22s %s\n' "GSM8K (5-shot)" "flex=${flex:-?} strict=${strict:-?}" "$g"
fi

# GPQA-Diamond (lm_eval results json)
for r in $(find "$WORK/results" -path "*gpqa_diamond*" -name "results.json" 2>/dev/null | sort); do
  python3 - "$r" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
for task, res in d.get("results", {}).items():
    flex = res.get("exact_match,flexible-extract")
    strict = res.get("exact_match,strict-match")
    def pct(x): return f"{x*100:.2f}" if x is not None else "?"
    print(f"{'GPQA-Diamond (cot_0shot)':28s} flex={pct(flex)} strict={pct(strict)}  {sys.argv[1]}")
EOF
done

# AA-LCR (judged output of aa_lcr.py judge; otherwise generation progress)
for j in $(find "$WORK/results" -path "*aa-lcr*" -name "judged.jsonl" 2>/dev/null | sort); do
  python3 - "$j" <<'EOF'
import json, sys
n = c = 0
for line in open(sys.argv[1]):
    r = json.loads(line)
    v = r.get("verdicts", [])
    c += sum(v); n += len(v)
print(f"{'AA-LCR (judged)':28s} {100*c/max(n,1):.2f} ({c}/{n})  {sys.argv[1]}")
EOF
done
for j in $(find "$WORK/results" -path "*aa-lcr*" -name "responses.jsonl" 2>/dev/null | sort); do
  n=$(wc -l < "$j")
  printf '%-28s %-22s %s\n' "AA-LCR (generation)" "${n}/100 done, unjudged" "$j"
done

# SciCode (inspect_ai .eval logs)
for e in $(find "$WORK/results" -path "*scicode*" -name "*.eval" 2>/dev/null | sort); do
  python3 - "$e" <<'EOF' 2>/dev/null
import json, sys, zipfile
try:
    with zipfile.ZipFile(sys.argv[1]) as z:
        hdr = json.loads(z.read("header.json"))
    res = hdr.get("results") or {}
    scores = res.get("scores", [{}])[0].get("metrics", {})
    s = {k: f"{v['value']*100:.2f}" for k, v in scores.items()}
    print(f"{'SciCode':28s} {s}  {sys.argv[1]}")
except Exception as e:
    print(f"{'SciCode':28s} (unparsed: {e})  {sys.argv[1]}")
EOF
done
echo
