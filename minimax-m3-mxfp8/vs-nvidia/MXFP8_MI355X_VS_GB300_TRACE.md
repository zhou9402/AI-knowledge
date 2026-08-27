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
| GB300 | 44,135.29 | 5.5404 | 1,415.08 ms |
| GB300 / MI355X | **2.071x** | **2.071x** | -- |

The uninstrumented GB300 control reached 46,527.69 fresh tok/s. Event
instrumentation reduced GB300 throughput by 5.14%, so it did not inflate the
reported NVIDIA advantage.

### P kernel/stage comparison

Times are mean microseconds per call. Stages are nested and must not be added.

| Stage | MI355X us/call | GB300 us/call | MI / GB |
|---|---:|---:|---:|
| Routed MoE | 3,475.273 | 1,912.037 | 1.818x |
| Shared-expert MLP | 648.470 | 437.064 | 1.484x |
| Sparse QKV projection | 502.578 | 306.279 | 1.641x |
| Attention output projection | 371.800 | 181.330 | 2.050x |
| Indexer | 843.501 | 543.389 | 1.552x |
| AllReduce + norm | 805.301 | 682.786 | 1.179x |
| Sparse attention total | 2,041.734 | 1,312.677 | 1.555x |
| Dense attention core | 38,219.372 | 3,945.283 | 9.687x |

Dense attention appears in only three layers; routed MoE and sparse attention
repeat across 57 layers. Expanded by architectural multiplicity, the MI355X
P critical-path budget was:

| Top-level bucket | MI share | MI ms/pass | GB300 ms/pass |
|---|---:|---:|---:|
| Routed MoE | 37.40% | 198.091 | 108.986 |
| Dense attention | 22.03% | 116.721 | 13.640 |
| Sparse attention | 21.97% | 116.379 | 74.823 |
| AllReduce + norm | 18.24% | 96.636 | 81.934 |
| Dense-layer MLP | 0.36% | 1.885 | 2.596 |

### P kernel-level event detail

This table expands each event-timed stage by its architectural call count.
These rows are nested details, so they must not be added to the top-level
budget above. Unlike the D profiler table discussed later, both P columns were
measured with matching low-overhead device events.

| Kernel/stage | Calls/pass | MI355X ms/pass | GB300 ms/pass | MI / GB |
|---|---:|---:|---:|---:|
| Routed MXFP8 MoE | 57 | 198.091 | 108.986 | 1.818x |
| Dense attention core | 3 | 114.658 | 11.836 | 9.687x |
| Sparse attention core/path | 57 | 116.379 | 74.823 | 1.555x |
| INT4/fused AllReduce + norm | 120 | 96.636 | 81.934 | 1.179x |
| Sparse indexer | 57 | 48.080 | 30.973 | 1.552x |
| Shared-expert MLP | 57 | 36.963 | 24.913 | 1.484x |
| Sparse QKV projection | 57 | 28.647 | 17.458 | 1.641x |
| Attention output projection | 60 | 22.308 | 10.880 | 2.050x |
| MLP gate/up projection | 60 | 21.293 | 13.799 | 1.543x |
| MLP down projection | 60 | 16.413 | 8.164 | 2.010x |
| MoE router | 57 | 3.554 | 16.537 | 0.215x |
| Dense QKV projection | 3 | 0.948 | 0.812 | 1.167x |

The largest removable P budgets are routed MoE (89.104 ms/pass), dense
attention core (102.822 ms/pass despite appearing in only three layers), and
sparse attention (41.556 ms/pass).

## D-only result

Matched service workload: TP4, DecodeBenchConnector, EAGLE3 draft-3 synthetic
acceptance, 60K--120K token-ID input, exact OSL600, and two client warm-ups.

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

## Optimization priorities

1. P: dense attention, routed MoE, and sparse attention.
2. D: dense+sparse attention and indexer; then unexplained graph work.
3. Treat the profiler-only 51.41% collective share as invalidated by
   instrumentation interference.
4. Use clean runs for capacity and low-overhead events for attribution.

## Provenance

- MI355X P event job: 2177.
- GB300 P event job: 9695.
- MI355X D profiler job: 2221.
- GB300 D profiler job: 9724.
- Corrected MI355X D event job: 2239.
- Original working-tree report:
  `benchmarks/kernels/minimax_m3/MXFP8_PONLY_C8_MI355X_VS_GB300_KERNEL_TIMING.md`.
