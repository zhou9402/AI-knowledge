# MiniMax-M3 MXFP8: MI355X versus GB300

## Scope

This record compares the TP4 MXFP8 production paths on MI355X and GB300.
Both sides use MXFP8 target weights, FP8 main/index KV, and the same logical
MiniMax-M3 P-only and D-only workloads. GPU events are preferred for critical
path attribution; profiler totals are retained only where they help identify
kernel families.

## P-only result

Matched workload: C8, 60K--120K ISL, about 90% prefix reuse, OSL1, two client
warm-ups, and 48 measured requests.

| Platform | Instrumented fresh tok/s | Request/s | TTFT P50 |
|---|---:|---:|---:|
| MI355X | 21,315.20 | 2.6757 | 3,033.02 ms |
| MI355X SOTA (2026-08-27, integration/p1) | 29,446.58 | 3.6965 | 2,183.68 ms |
| GB300 | 44,135.29 | 5.5404 | 1,415.08 ms |
| GB300 / MI355X | **2.071x** | **2.071x** | -- |
| GB300 / MI355X SOTA | **1.499x** | **1.499x** | -- |

The uninstrumented GB300 control reached 46,527.69 fresh tok/s. Event
instrumentation reduced GB300 throughput by 5.14%, so it did not inflate the
reported NVIDIA advantage.

### P kernel/stage comparison

Times are mean microseconds per call. Stages are nested and must not be added.

| Stage | MI355X us/call | MI355X SOTA 08-27 us/call | GB300 us/call | MI / GB |
|---|---:|---:|---:|---:|
| Routed MoE | 3,475.273 | 3,390.5 | 1,912.037 | 1.818x |
| Shared-expert MLP | 648.470 | 653.5 | 437.064 | 1.484x |
| Sparse QKV projection | 502.578 | 498.5 | 306.279 | 1.641x |
| Attention output projection | 371.800 | 364.3 | 181.330 | 2.050x |
| Indexer | 843.501 | 865.5 | 543.389 | 1.552x |
| AllReduce + norm | 805.301 | 808.5 | 682.786 | 1.179x |
| Sparse attention total | 2,041.734 | 1,529.1 | 1,312.677 | 1.555x |
| Dense attention core | 38,219.372 | 14,843.8 | 3,945.283 | 9.687x |

Dense attention appears in only three layers; routed MoE and sparse attention
repeat across 57 layers. Expanded by architectural multiplicity, the MI355X
P critical-path budget was:

| Top-level bucket | MI share | MI ms/pass | MI SOTA 08-27 ms/pass | GB300 ms/pass |
|---|---:|---:|---:|---:|
| Routed MoE | 37.40% | 198.091 | 193.3 | 108.986 |
| Dense attention | 22.03% | 116.721 | 44.5 | 13.640 |
| Sparse attention | 21.97% | 116.379 | 87.2 | 74.823 |
| AllReduce + norm | 18.24% | 96.636 | 97.0 | 81.934 |
| Dense-layer MLP | 0.36% | 1.885 | 2.0 | 2.596 |

### P kernel-level event detail

This table expands each event-timed stage by its architectural call count.
These rows are nested details, so they must not be added to the top-level
budget above. Unlike the D profiler table discussed later, both P columns were
measured with matching low-overhead device events.

| Kernel/stage | Calls/pass | MI355X ms/pass | MI SOTA 08-27 ms/pass | GB300 ms/pass | MI / GB |
|---|---:|---:|---:|---:|---:|
| Routed MXFP8 MoE | 57 | 198.091 | 193.3 | 108.986 | 1.818x |
| Dense attention core | 3 | 114.658 | 44.5 | 11.836 | 9.687x |
| Sparse attention core/path | 57 | 116.379 | 87.2 | 74.823 | 1.555x |
| INT4/fused AllReduce + norm | 120 | 96.636 | 97.0 | 81.934 | 1.179x |
| Sparse indexer | 57 | 48.080 | 49.3 | 30.973 | 1.552x |
| Shared-expert MLP | 57 | 36.963 | 37.2 | 24.913 | 1.484x |
| Sparse QKV projection | 57 | 28.647 | 28.4 | 17.458 | 1.641x |
| Attention output projection | 60 | 22.308 | 21.9 | 10.880 | 2.050x |
| MLP gate/up projection | 60 | 21.293 | — | 13.799 | 1.543x |
| MLP down projection | 60 | 16.413 | — | 8.164 | 2.010x |
| MoE router | 57 | 3.554 | — | 16.537 | 0.215x |
| Dense QKV projection | 3 | 0.948 | — | 0.812 | 1.167x |

The largest removable P budgets are routed MoE (89.104 ms/pass), dense
attention core (102.822 ms/pass despite appearing in only three layers), and
sparse attention (41.556 ms/pass).

## D-only result

Matched service workload: TP4, DecodeBenchConnector, EAGLE3 draft-3 synthetic
acceptance, 60K--120K token-ID input, exact OSL600, and two client warm-ups.

*(This D table is the pre-optimization MI355X record; the 2026-08-27
rematch with the optimized MI355X stack is in the next section. Read its
residency reconciliation before comparing tok/s across the two.)*

| C | MI355X out tok/s | GB300 out tok/s | GB / MI | MI TPOT P50 | GB TPOT P50 |
|---:|---:|---:|---:|---:|---:|
| 48 | 1,596.94 | 3,032.48 | 1.899x | 29.025 ms | 14.283 ms |
| 56 | 1,740.76 | 3,466.67 | 1.991x | 31.344 ms | 15.049 ms |
| 64 | 1,832.65 | 3,617.74 | 1.974x | 33.212 ms | 16.632 ms |
| 72 | 1,798.61 | 3,991.05 | 2.219x | 36.224 ms | 16.393 ms |

C72 is a GB300 scheduler boundary: only 64 requests were resident and up to
eight were waiting. The like-for-like resident C48--C64 range shows GB300 at
1.90--1.99x MI355X throughput with roughly half the TPOT.

### D trace validity correction

The earlier Kineto trace reported 51.41% of summed MI355X GPU time as
collective/synchronization. That number is invalid for kernel-share or
critical-path comparison: profiling reduced MI355X C16 throughput by about
70% and greatly amplified device-side synchronization waits. It is therefore
removed from the comparison table and must not be used to diagnose D.

The NVIDIA and AMD profiler-family percentages are also not directly
comparable under this asymmetric perturbation. A matched low-overhead event
run on both platforms would be required for a valid family-by-family D
comparison.

### Authoritative MI355X D event breakdown

Low-overhead HIP events later measured the complete target iteration at
55.053 ms:

| Execution boundary | Time | Share |
|---|---:|---:|
| Dense-attention path | 9.088 ms | 16.51% |
| Sparse-attention path | 16.622 ms | 30.19% |
| MoE | 13.490 ms | 24.50% |
| AllReduce + norm | 1.240 ms | 2.25% |
| Dense-layer MLP | 0.273 ms | 0.50% |
| Other captured-graph work | 14.340 ms | 26.05% |

#### D nested event targets

These event boundaries are contained in the rows above and must not be added
again. They are the finest trustworthy D critical-path attribution currently
available.

| Kernel/stage boundary | MI355X ms/iteration | Iteration share |
|---|---:|---:|
| Routed-expert / routing residual | 9.890 | 17.96% |
| Dense-attention backend core | 8.868 | 16.11% |
| Sparse-attention body, excluding indexer | 7.854 | 14.27% |
| Indexer | 4.943 | 8.98% |
| Sparse projections / cache / output | 3.825 | 6.95% |
| Shared-expert MLP | 2.868 | 5.21% |
| MoE router gate | 0.732 | 1.33% |

#### D exact trace kernel names (diagnostic only)

The trace still identifies which concrete kernels execute, but its shares and
collective latency are distorted by profiling. This table is therefore for
kernel mapping only; it must not replace the event table above or be used for
an Amdahl calculation.

| Platform | Trace kernel | Mean us/call | Interpretation |
|---|---|---:|---|
| MI355X | `kernel_unified_attention` | 2,130.446 | Dense attention candidate |
| MI355X | `decode_index_score_topk_partial_fp8` | 156.652 | Fused index-score/partial Top-K |
| MI355X | AITER MoE1 + SwiGLU | 75.379 | Routed expert gate/up + activation |
| MI355X | AITER MoE2 | 41.264 | Routed expert down projection |
| MI355X | `mxfp8_linear_kernel` | 11.271 | MXFP8 dense/projection GEMM |
| MI355X | `gqa_sparse_decode_kernel` | 14.928 | Sparse decode attention core |
| GB300 | `kernel_unified_attention` | 1,516.991 | Dense attention candidate |
| GB300 | FlashInfer sparse indexer Top-K | 21.386 | Top-K portion; not one-to-one with AMD fused kernel |
| GB300 | MiniMax index-decode score | 18.586 | Score portion; not one-to-one with AMD fused kernel |
| GB300 | TRT-LLM MXFP8 MoE gate/up BMM | 45.278 | Dominant gate/up shape |
| GB300 | TRT-LLM MXFP8 MoE down BMM | 28.692 | Dominant down shape |
| GB300 | FlashInfer MXFP8 dense GEMM | 9.496 | MXFP8 dense/projection GEMM |
| GB300 | `gqa_sparse_decode_kernel` | 16.251 | Sparse decode attention core |

The two platforms do not always use one-to-one kernel decomposition. For
example, AMD fuses index score and partial Top-K while GB300 reports separate
score and Top-K kernels. Direct ratios should only be computed for equivalent
boundaries under matched event timing.

The supported conclusion is that MI355X D is primarily limited by
attention/indexer and remaining graph work, not communication bandwidth. The
older profiler suggests that MoE is unlikely to explain the gap, but that
cross-platform family comparison remains directional rather than an
authoritative event-timed result.

## D-only rematch (2026-08-27): optimized MI355X stack

Same C64 D-only workload (TP4, DecodeBenchConnector fill_mean=0.015, EAGLE3
draft-3 synthetic acceptance [0.7, 0.5, 0.4], 60K--120K token-ID input,
OSL600, 192 prompts = 3 waves, seed 42). The MI355X side now runs the
optimized stack: vLLM 0.27.2rc1 (image ac750) + the integrated ws1/ws2/ws3
kernel ports — aiter PTPC (`gemm_a8w8_bpreshuffle`) for all dense MXFP8
linears by default, aiter fused AR+GemmaRMSNorm in `post_attn` mode (FFN-AR
deferral is intentionally disabled under EAGLE3), and the decode-tier MXFP8
fmoe tuned CSV on aiter 0.1.19. The ws4 gluon sparse-PA path stayed off: while
the attend path runs eagerly (`eager_break_during_capture`) it is a net drag.
GSM8K 5-shot (limit 100) on this stack: 0.96 strict-match. The GB300 side is
unchanged (vLLM 0.17.2rc1.dev4448; job 10377 anchor, event job 10388).

### Capacity (instrumentation off)

| Platform | out tok/s | TPOT p50 | step (ITL p50) | TTFT p50 | EAGLE3 acc. len |
|---|---:|---:|---:|---:|---:|
| MI355X (optimized) | 1,724.35 | 23.434 ms | 60.12 ms | 2,125 ms | 2.599 (measured) |
| GB300 | 3,620.99 | 16.032 ms | 36.57 ms | 412 ms | ~2.3--2.4 (ITL/TPOT) |
| GB300 / MI355X | **2.100x** | **1.462x** (AMD/GB) | 1.64x | -- | -- |

GB300's 2.100x measured-throughput lead decomposes exactly into decode speed
times decode residency:

    tok/s ratio = (TPOT_MI / TPOT_GB) x (residency_GB / residency_MI)
    2.100       = 1.462             x 1.437

where residency = measured tok/s / (C / TPOT p50) = 0.907 (GB300) vs 0.631
(MI355X): the fraction of slot-time actually spent decoding. Each 60K--120K
token request spends the rest in connector fill/queueing; the MI355X run's
TTFT (p50 2.13 s, mean 5.5 s) is much heavier than GB300's (p50 0.41 s) on
the same DecodeBenchConnector workload, so more of its 64 slots are
non-decoding at any moment. TPOT is the apples-to-apples decode-speed metric;
on tok/s, part of NV's lead is fill/queue speed, not decode kernels.

### Reconciliation with the earlier record (1,832.65 tok/s / 33.212 ms)

The old C64 row is the pre-optimization stack. Decode speed genuinely
improved: TPOT 33.212 -> 23.434 ms (-29.4%), i.e. the pure-decode ceiling
C/TPOT rose 1,927 -> 2,731 tok/s (+41.7%). Measured tok/s moved the other way
(1,832.65 -> 1,724.35, -5.9%) purely through residency accounting: the old
run implied ~95% decode residency (E2E 20.95 s = 19.93 s decode + ~1.0 s
fill/queue per request), while the new run sits at 63% (E2E 22.27 s = 14.06 s
decode + ~8.2 s fill/queue; TTFT p50 2.13 s). The new stack fills/queues the
60K--120K prompts more slowly per request than the old record's stack did,
which masks the decode win in aggregate throughput. Not a decode regression.

### Per-decode-step stage budget (event-timed legs)

Step identity: with EAGLE3 draft-3, every engine step is one target verify
pass (4 positions/seq) plus three eager draft passes and the sampler. The hot
C64 cudagraph is exactly the verify pass on both platforms (moe x57 +
dense_attn x3 = one 60-layer target-model pass; MI355X graph ...626448 at
55.01 ms/replay, GB300 graph ...723344 at 38.18 ms/replay). Per-replay stage
time = p50 call time x calls per replay. Both legs are instrumented (MI355X
event leg: -50.7% tok/s, +18.1% TPOT; GB300: -14.8% tok/s, +11.2% TPOT), so
absolute stage times are inflated; use them for structure and shares. The
verify-graph totals (55.01 vs 38.18 ms = 1.44x) track the clean TPOT ratio
(1.462x), which cross-checks the attribution. Nesting: dense_attn_core is
inside dense_attn; on the AMD model the MoE block time includes the router
and the shared-expert MLP (MoE.forward calls them), while on GB300
shared_expert_mlp is a sibling call (its stage times sum to the graph total
without nesting).

| Bucket | MI355X ms/step | GB300 ms/step | MI / GB |
|---|---:|---:|---:|
| Routed MoE block (incl. router) | 17.67 | 12.64 | 1.40x |
| Shared-expert MLP | 2.93 | 10.66 | **0.27x (MI faster)** |
| Dense attention (3 full-context layers) | 9.40 | 1.00 | 9.44x |
| Sparse attn + indexer (57 layers) | 7.91 | n/a (uninstrumented, in step-graph gap) | -- |
| AllReduce + norm | 2.02 | 4.08 | **0.49x (MI faster)** |
| Projections / dense GEMMs | 4.77 | 7.99 | **0.60x (MI faster)** |
| Sum of instrumented stages | 44.69 | 36.36 | -- |
| Graph residual (see notes) | 10.32 | 1.82 | -- |
| **Verify graph total** | **55.01** | **38.18** | **1.44x** |
| Step, instrumented leg (ITL p50) | 68.02 | 42.31 | 1.61x |
| Step, clean leg (ITL p50) | 60.12 | 36.57 | 1.64x |

Notes on the buckets:

- Sparse attn + indexer, MI355X: sparse decode 43.1 us x57 + merge; index
  score+topk ~300 us x15 per step (index_topk_freq=4 cross-layer reuse: a
  quarter of the 57 sparse layers refresh per step).
- Sparse attn + indexer, GB300: not instrumented inside the verify graph (its
  stages sum to the graph total without it); from the prior trace it is
  ~0.9 ms sparse decode + ~1.3 ms indexer at this geometry, inside the
  step-minus-graph gap (<=4.1 ms with draft/sampler).
- AllReduce + norm: MI355X runs 60 fused calls/step (ws3 `post_attn`: one
  fused AR+norm per layer; the FFN keeps its internal AR under EAGLE3).
  GB300 runs 120: its 0.17 model defers the FFN AR into the next layer's
  norm — structurally the ws3 `all` mode that EAGLE3 blocks on AMD.
- Graph residual: MI355X 10.32 ms (logits/sampler at 256 verify tokens,
  EAGLE3 aux-hidden capture, in-graph event-record overhead, other
  uninstrumented work) vs GB300 1.82 ms. This is the least-characterized
  MI355X bucket.
- Non-graph work per step (3 draft passes, sampler, host) is ~5 ms on
  MI355X (clean-leg step minus instrumented graph) and within noise on
  GB300 (the on-leg graph time is inflated ~10%).

### Where the remaining GB300 decode advantage comes from

Verify-graph delta is 55.01 - 38.18 = 16.8 ms/step (1.44x, matching the
clean TPOT ratio 1.462x):

- Dense attention over the full 60K--120K context (3 layers): +8.4 ms for
  MI355X (9.44x) — the single largest kernel-level gap.
- Graph residual / uninstrumented work: +8.5 ms for MI355X.
- Routed MoE block: +5.0 ms (1.40x).
- Offset by MI355X wins from the ws1/ws2/ws3 ports: shared-expert MLP
  -7.7 ms (3.6x faster), projections -3.2 ms (1.7x), AR+norm -2.1 ms (2x).

The 2.100x measured-tok/s lead additionally carries the 1.437x residency
factor (connector fill/queue overhead per request, not decode kernels).

### What the optimization bought (same D-only workload, prior record -> now)

| C64 | TPOT p50 | Decode ceiling (C/TPOT) | Measured tok/s |
|---|---:|---:|---:|
| Pre-optimization record | 33.212 ms | 1,927 tok/s | 1,832.65 |
| Optimized stack | 23.434 ms (-29.4%) | 2,731 tok/s (+41.7%) | 1,724.35 (see residency note) |

## D-only rematch at AMD sweet spot, C64 (2026-08-27, event-timed, optimized MI355X stack)

> *Editor's note (2026-08-28 merge): this section was written by the
> parallel analysis pass and accidentally dropped by a concurrent edit;
> restored here. See the section above for the normalized per-step
> budget and the exact residency decomposition (2.100x = 1.462x decode
> speed x 1.437x residency). One correction to the per-call table
> below: the MI355X AR+norm row is 60 calls/step in the stage TSV
> (ws3 `post_attn` mode under EAGLE3), not x120; per-call parity
> (35.2 vs 34.0 us) stands, and per-step MI355X is 2x faster there.

Post-kernel-port MI355X stack (ws1 MoE tuned-CSV + ws2 PTPC aiter GEMM + ws3 fused
AR+norm; ws4 gluon OFF — eager-path regression). GB300 unchanged (canonical config).
Protocol: TP4, DecodeBenchConnector (fill 0.015), EAGLE3 draft-3 synthetic
acceptance [0.7,0.5,0.4], 60K-120K token-id ISL, OSL 600, C64 closed loop.
Timing: HIP/CUDA-event stage timing, OFF leg for capacity, ON leg for stage shares
(instrumentation cost: GB300 ~11-15%, MI355X ~17.5% TPOT — stage tables are
relative shares only).

### Capacity (timing OFF, zero errors both sides)

| Platform | out tok/s | TPOT P50 | vs |
|---|---:|---:|---|
| GB300 (job 10377, repro of canonical) | **3,620.99** | **16.03 ms** | 1.00x |
| MI355X optimized (this work) | 1,724.4 | 23.43 ms | NV = **2.10x** tok/s, 1.46x TPOT |
| MI355X pre-optimization (earlier record) | 1,832.65 | 33.21 ms | — |

MI355X optimization moved TPOT 33.2 -> 23.4 ms (-29%). Raw tok/s is dominated by
the synchronous DecodeBenchConnector KV-fill TTFT at this run length (MI TTFT p90
13.7 s), so TPOT is the clean decode metric; the tok/s dip vs the old record is
wave/accounting noise, not a regression.

### Per-stage decode-step comparison (verify cudagraph, per-call us, relative shares)

| stage (per call) | MI355X opt. | GB300 | MI / GB |
|---|---:|---:|---:|
| Routed MoE (per layer, x57) | 312.7 | 220.9 | **1.42x** |
| Dense attention (per layer, x3) | 3,135 | 331.8 | **9.45x** |
| Index score+topk (x15) | 301.8 | — | n/a |
| Shared-expert MLP (x57) | 51.8 | 185.2 | 0.28x (MI faster) |
| Sparse decode kernel (x57) | 43.6 | — | n/a |
| AR+norm fused (x120) | 35.2 | 34.0 | 1.04x (parity) |
| sparse QKV proj (x57) | 18.7 | 17.7 | 1.06x |
| attn o_proj (x60) | 15.5 | 14.7 | 1.05x |

MI355X verify-graph step budget: MoE 41.0%, dense attn 20.6%, index topk 9.9%,
shared-expert MLP 6.8%, sparse decode 5.6%, AR+norm 5.0%; ~15 ms/step residual =
EAGLE3 draft forwards + sampler + scheduler (not instrumented).

### Read

- GB300's 2.1x lead at C64 is led by **dense attention (9.45x; 3 layers at 60-120K
  ctx)** and routed MoE (1.42x); AR/norm and the projection GEMMs are at parity
  after the kernel ports.
- ws3 (fused AR+norm) holds parity with trtllm allreduce fusion — the glue gap is
  closed. The remaining MI355X decode gap is dense-attention-bound.

### Artifacts

- MI355X: `vultr-mi355x:/mnt/vfs/homes/peiyuanz/m3-compare/results-nv-amd/amd/event-c64/{off,on,on-gated}/` (SUMMARY.md, stage TSVs); optimized-stack decode record in `records/DECODE_OPTIMIZATION_RECORD.md`.
- GB300: repro `results/mxfp8/decode-only/c64-10377/`, event `results/mxfp8-event-stage6/decode-only/c64-10388/` (exit-2 is a summary-jq bug; measurement complete); copied to `vultr:.../results-nv-amd/gb300/`.

## Optimization priorities

1. P: dense attention, routed MoE, and sparse attention.
2. D (updated 2026-08-27): dense attention remains the top kernel target
   (9.4x, ~8.4 ms/step at C64); second is the verify-graph residual on
   MI355X (~10 ms/step: logits/sampler, EAGLE3 aux capture, in-graph event
   overhead — needs direct instrumentation); routed MoE is 1.40x; the
   connector fill/queue path drives the residency gap (1.44x of the 2.10x
   tok/s lead) and is a scheduling/connector problem, not a kernel one.
3. Treat the profiler-only 51.41% collective share as invalidated by
   instrumentation interference.
4. Use clean runs for capacity and low-overhead events for attribution.

## Provenance

- MI355X P event job: 2177.
- GB300 P event job: 9695.
- MI355X SOTA (integration/p1) jobs: 3155 (control), 3156 (event); kernel
  profile job: 3173.
- MI355X D profiler job: 2221.
- GB300 D profiler job: 9724.
- Corrected MI355X D event job: 2239.
- 2026-08-27 D rematch: MI355X event legs `results-nv-amd/amd/event-c64/`
  (off = clean capacity, on = event timing; vultr-mi355x, vllm-bench-ac750,
  192 prompts, seed 42, hot graph 13885675626448); GB300 repro job 10377 and
  event job 10388 (hot graph 266659190723344). Analysis scripts and tables:
  `tools/m3_compare/nv-amd-compare/` and remote `results-nv-amd/compare/`.
- Original working-tree report:
  `benchmarks/kernels/minimax_m3/MXFP8_PONLY_C8_MI355X_VS_GB300_KERNEL_TIMING.md`.

---

## D-only at the AMD sweet point, C24 (2026-08-28, event-timed, optimized MI355X stack)

C64 turned out admission-bound on MI355X (heavy queueing), so the C64 throughput
comparison was invalid. This is the rematch at **C24**, the AMD sweet-spot region.
Same D-only protocol as before (TP4, DecodeBenchConnector, EAGLE3 draft-3
synthetic [0.7,0.5,0.4], 60-120K ISL, OSL 600). MI355X runs the optimized stack
(ws1+ws2+ws3, ws4 off, aiter 0.1.19). Zero errors on all runs. Event legs carry
~18-21% TPOT instrumentation overhead on both sides — stage tables are relative
shares / per-call times only; capacity numbers come from the timing-OFF legs.

### Capacity at C24 (timing OFF)

| Platform | out tok/s | TPOT P50 | TPOT P99 | ITL P50 | TTFT P50 |
|---|---:|---:|---:|---:|---:|
| GB300 (job 10545) | **2,212.4** | **9.86 ms** | 11.09 ms | 24.17 ms | 320.8 ms |
| MI355X optimized | 1,259.4 | 17.58 ms | 19.14 ms | 45.50 ms | 434.5 ms |
| **NV / AMD** | **1.76x** | **1.78x** | 1.73x | 1.88x | — |

The gap narrows vs C64 (2.10x) because C64 on MI355X was queue-bound. At C24 both
sides run queue-free (AMD TTFT p50 435 ms vs 2,125 ms at C64).

### Per-kernel/stage comparison at C24 (verify cudagraph, per-call p50 us)

| stage (calls/step) | MI355X | GB300 | MI / GB |
|---|---:|---:|---:|
| Routed MoE (x57) | 262.0 | 174.8 | **1.50x** |
| Dense attention (x3) | 3,060 | 169.0 | **18.1x** |
| Shared-expert MLP (x57) | 49.1 | 143.4 | 0.34x (MI faster) |
| AR+norm (x60 MI / x120 GB) | 24.5 | 23.1 | parity per call; MI does half the calls (ws3 deferral) |
| moe_router (x57) | 12.2 | 17.2 | 0.71x (MI faster) |
| sparse QKV proj (x57) | 18.4 | 16.7 | 1.10x |
| attn o_proj (x60) | 14.2 | 14.0 | ~1.0x |
| Sparse decode + indexer (MI only) | 25.6 + 30.2 | not instrumented | — |

Per-step budgets: MI355X verify graph 37.7 ms (MoE 39.6%, dense attn 24.3%);
GB300 verify graph ~29.6 ms instrumented (MoE 9.96 ms, shared-expert 8.17 ms,
dense attn 0.51 ms, AR+norm 2.77 ms).

**Dense attention got relatively WORSE at C24 (18.1x vs 9.4x at C64)**: the MI
Triton unified-attention kernel is latency-bound and flat vs batch (9.2 ms/step
at both C24 and C64), while GB300's FMHA scales down with batch. It is the #1
kernel gap at every concurrency measured.

### AMD sweet spot (steady-state, SLA TPOT p50 <= 16.6 ms)

| C | out tok/s | TPOT p50 | SLA |
|---:|---:|---:|---|
| C18 | 1,026.2 | 16.39 ms | PASS (sweet spot) |
| C19 | 1,068.8 | 16.76 ms | miss (plateau edge) |
| C20 | 1,116.3 | 16.75 ms | miss by 0.15 |
| C22 | 1,206.2 | 17.13 ms | miss |
| C24 | 1,259.4 | 17.59 ms | miss |

Flat C18->C20, then ~+0.4 ms/step beyond C20. If the SLA relaxes to ~16.8 ms,
C20 is the efficient plateau. Note: an earlier single-wave C24 reference
(15.46 ms) was reproduced by our warmup wave (15.63 ms); steady state drifts to
17.58 ms — short-window numbers flatter the sweet point.

### Artifacts

- MI355X: `results-nv-amd/amd/event-c{18,19,20,22}/` and `event-c24/` (SUMMARY.md,
  stage-table.txt, TSVs); sweet-spot scan: `results-nv-amd/amd/sweetspot.md`.
- GB300: jobs 10545 (clean) / 10546 (event); copied to
  `results-nv-amd/gb300-c24/` on the vultr cluster.
