# MiniMax-M3 MXFP8: ATOM P vs vLLM P

## SLA boundary

Both measurements use TP4, 60K--120K input tokens, eight prefix sessions,
approximately 90% prefix reuse, OSL=1, and a 16,384-token batch limit. The SLA
is TTFT P50 < 3 seconds.

| Runtime | Highest measured SLA pass | TTFT P50 | Fresh prefill tok/s |
|---|---:|---:|---:|
| ATOM | 5.0 offered QPS (4.959 achieved) | 2.274 s | 39,497.8 |
| vLLM current final stack | 3.45 offered QPS (3.429 achieved) | 2.492 s | 27,317.3 |

ATOM's measured SLA boundary is about 45% higher. The earlier comparison with
vLLM's C8 closed-loop result (3.069 QPS) overstated this gap because the arrival
models differed.

## What the traces show

The available vLLM trace predates the current INT4 QuickReduce final stack, so
its 31.1% RCCL share is not the current communication cost. The current 3.45-QPS
vLLM result explicitly dispatches INT4 QuickReduce. ATOM also uses INT4
QuickReduce.

ATOM enables `use_index_cache=true,index_topk_freq=4`; the current vLLM result
does not. This recomputes sparse index scores and top-k selection on one of every
four sparse layers. vLLM now has an equivalent opt-in implementation, but it
still computes the fused index Q/K projections on skipped layers, so this alone
cannot explain the full gap and requires quality validation.

A first matched vLLM performance run with that opt-in enabled at 3.45 offered
QPS passed the workload checks and reduced TTFT P50 from 1.888 s to 1.152 s
(39.0%) while fresh prefill throughput changed from 26,117 to 26,455 tok/s.
The SLA boundary is being re-scanned; output/quality validation remains a gate
because cross-layer reuse intentionally changes sparse block selection.

The profiled A/B confirms the mechanism: `index_block_score` fell from 1,539
calls / 847 ms to 405 calls / 224 ms, and `topk_index` from 1,539 calls / 381 ms
to 405 calls / 99 ms. Estimated total rank-0 kernel time fell about 9%. A small
logprob probe is not quality-equivalent, however: only 1/10 prompts retained the
same top-1 token, with overall mean absolute delta 1.96. A task-level model eval
is required before production enablement.

Boundary diagnostics reinforce the distinction between performance and quality.
Frequency 4 at 4.5 offered QPS saturated at 3.897 achieved QPS and TTFT P50
5.845 s, so the SLA boundary is below 4.5. Two attempts at 3.75 QPS were stopped
by the default smoke gate because the chat endpoint returned a one-token empty
string. A separately labeled performance-only retry at 3.75 achieved 3.666 QPS,
29,207 fresh tok/s, and TTFT P50 1.934 s, but is not quality-qualified.
The corresponding 4.0 offered-QPS point also met the latency target at 3.802
achieved QPS, 30,284 fresh tok/s, and TTFT P50 2.458 s. This is a 10.9%
performance-boundary gain over the prior 3.429-QPS vLLM result, but still not a
production claim without model evaluation.
The final 4.1 point remained just inside SLA at 3.831 achieved QPS, 30,516 fresh
tok/s, and TTFT P50 2.851 s. It is the highest measured performance-only pass;
4.5 is the measured fail boundary.
Frequency 2 at 3.6 offered QPS passed the smoke and workload checks, achieving
3.482 QPS, 27,742 fresh tok/s, and TTFT P50 1.857 s.

ATOM's P command uses `cudagraph_mode=FULL`, but this does **not** graph prefill.
The implementation's `run_model()` explicitly routes prefill through the eager
branch and only replays graphs for decode. vLLM reports
`AttentionCGSupport.UNIFORM_BATCH` for the M3 sparse-attention and indexer
backends and downgrades `FULL` to `FULL_DECODE_ONLY`; for this P-only comparison,
that matches ATOM's effective behavior and is not a source of the performance
gap.

ATOM's profile attributes 19.9% to attention/indexer and 17.3% to MoE. The old
vLLM profile attributes 16.2% to attention/indexer and 14.7% to MoE, but the
profiles have different request schedules and profiler perturbation. These
shares identify targets; they are not an apples-to-apples speedup measurement.

The native vLLM MXFP8 dense path should be retained. ATOM uses PTPC-FP8 for
dense layers and spends a visible share in conversion/copy kernels, so porting
that dense path is not a likely catch-up lever.

## Prioritized vLLM work

1. Run a quality-gated P-only A/B with `index_topk_freq=4` under the same
   open-loop workload.
2. Capture the current final stack, then compare matched attention/indexer and
   collective shapes rather than aggregate profiler percentages.
3. Optimize sparse attention and indexer kernels at the matched shapes.
4. Keep fused shared expert enabled for D only. It improves D C22 by 6.0%, but
   regresses large-M prefill and is not ATOM P's advantage.

## New kernel experiments

At production prefill shapes, an ATOM-style dense-attention pipeline (paged-FP8
KV gather/dequantize, BF16 FMHA, and prefix-chunk merge) was correct within
relative L2 `0.002867` and took 0.305--0.333x the latency of the current AITER
Unified pipeline in an isolated pipeline benchmark. This did not translate to
an end-to-end win: at 3.45 offered QPS it delivered 26,098 fresh tok/s with
TTFT P50 1.961 s, versus 26,117 fresh tok/s and 1.888 s for the matched current
backend run. Do not replace the production dense backend with this path.

Fusing sparse page-table/context-length materialization into the Top-K output
kernel passed exact-output checks and reduced the combined microbenchmark from
191.682 to 173.602 us (1 request), 110.160 to 100.041 us (2 requests), and
80.961 to 72.241 us (4 requests): a 10.1--12.1% subpath gain. Whole-model
impact has not yet been measured.
