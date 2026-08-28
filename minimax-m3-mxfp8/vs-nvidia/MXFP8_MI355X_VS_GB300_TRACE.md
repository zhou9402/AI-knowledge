## Latest (2026-08-28) — current best under SLA

SLA: TPOT p50 <= 16.6 ms. All numbers zero-error, warm, steady-state.

### D-only (latest)

Protocol: TP4, DecodeBenchConnector (fill 0.015), EAGLE3 draft-3 synthetic
acceptance [0.7,0.5,0.4], 60-120K token-id ISL, exact OSL 600, closed loop
unless noted. AMD = phase4 merged stack (Codex phase3 + ws1 decode MoE CSV +
ws3 fused AR+norm; PTPC OFF; aiter 0.1.19). NV = canonical GB300 production
(FLASHINFER+trtllm attn, trtllm MoE, EAGLE3).

| Platform | Best under SLA | out tok/s | TPOT p50 | QPS |
|---|---|---:|---:|---:|
| GB300 (NV) | C72 | **3,991.1** | 16.39 ms | ~6.7 |
| MI355X phase4 (AMD) | C28 sweet spot (closed loop) | 1,559.7 | 16.35 ms | 2.60 |
| MI355X phase4 open-loop | 2.4 QPS x 667 s, 1584/1584 ok | 1,423.6 | **15.72 ms** | 2.37 ach., PASS |

**NV/AMD at best-under-SLA: 2.56x throughput.** AMD D TPOT went 33.2 -> 16.35 ms
(-51%) over the campaign (pre-opt record -> phase4). NV leads on raw throughput;
per-token latency is nearly equal (16.39 vs 16.35 ms).

#### D kernel breakdown (event-timed, C24 geometry, per-call p50 us)

| stage (calls/step) | MI355X phase4 | GB300 | MI / GB |
|---|---:|---:|---:|
| Routed MoE (x57) | 262.0 | 174.8 | 1.50x |
| **Dense attention (x3)** | **3,060** | **169.0** | **18.1x** |
| Shared-expert MLP (x57) | 49.1 | 143.4 | 0.34x (MI faster) |
| AR+norm | 24.5 (x60, ws3-fused) | 23.1 (x120) | parity/call, MI half the calls |
| moe_router (x57) | 12.2 | 17.2 | 0.71x (MI faster) |
| sparse QKV proj (x57) | 18.4 | 16.7 | 1.10x |
| attn o_proj (x60) | 14.2 | 14.0 | ~1.0x |
| Sparse decode + indexer (MI) | 25.6 + 30.2 | not instrumented | — |

Why dense attention is 18x and not a typo: MI runs vLLM Triton
`unified_attention` for the 3 full-context layers — latency-bound (no
split-KV), flat vs batch (3.06 ms at both C24 and C64). GB300's FMHA sm100
splits KV and scales with batch (331.8 us at C64 -> 169 us at C24). Fix on MI:
switch the 3 dense layers to aiter ASM/FMHA (Codex `opt/dense-attention-fmha`
path). Per-step shares at C24: MI verify graph 37.7 ms — MoE 39.6%, dense attn
24.3%, indexer 7.0%; GB300 verify graph ~29.6 ms — dense attn only 0.51 ms.

### P-only (latest)

Protocol: C8, 60-120K ISL, ~90% prefix reuse, OSL 1, open-loop SLA
(TTFT p50 < 3 s; fresh prefill tok/s at the SLA sweet point).
The MI355X number is the production-legal SOTA: no PTPC dense-GEMM route and
no index top-k sharing (both are numerics/policy-restricted).

| Platform | SLA sweet spot |
|---|---:|
| GB300 | **44,135 fresh tok/s** (QPS 5.54) |
| MI355X SOTA (integration/p1, 2026-08-27) | **29,446.6 tok/s @ QPS 3.70**, TTFT p50 2,183.7 ms |
| ATOM reference | 39.5K tok/s |

NV/AMD P: **1.50x**. (The 39.7K = 100.6%-of-ATOM figure used PTPC + top-k
sharing and is not production-legal under the current accuracy policy; it lives
in the historical chapter.)

## Kernel breakdown (event-timed, AMD MI355X phase4 vs NV GB300)

### D (verify cudagraph, C24 geometry, per-call p50 us)

| stage (calls/step) | MI355X | GB300 | MI / GB |
|---|---:|---:|---:|
| Routed MoE (x57) | 262.0 | 174.8 | 1.50x |
| **Dense attention (x3)** | **3,060** | **169.0** | **18.1x** |
| Shared-expert MLP (x57) | 49.1 | 143.4 | 0.34x (MI faster) |
| AR+norm | 24.5 (x60, ws3-fused) | 23.1 (x120) | parity/call, MI half the calls |
| moe_router (x57) | 12.2 | 17.2 | 0.71x (MI faster) |
| sparse QKV proj (x57) | 18.4 | 16.7 | 1.10x |
| attn o_proj (x60) | 14.2 | 14.0 | ~1.0x |
| Sparse decode + indexer (MI) | 25.6 + 30.2 | not instrumented | — |

Dense-attention 18.1x is real, not a typo: MI runs vLLM Triton
`unified_attention` for the 3 full-context layers — latency-bound (no
split-KV), flat vs batch (3.06 ms at both C24 and C64). GB300's FMHA sm100
splits KV and scales (331.8 us at C64 -> 169 us at C24). Fix on MI: switch the
3 dense layers to aiter ASM/FMHA (`opt/dense-attention-fmha` path).
Per-step shares at C24: MI verify graph 37.7 ms — MoE 39.6%, dense attn 24.3%,
indexer 7.0%; GB300 verify graph ~29.6 ms — dense attn only 0.51 ms.

### P kernel/stage comparison (C8; MI SOTA = production-legal stack)

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



## Historical records (all pre-2026-08-28 data below; kept for reference)

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

> Update 2026-08-28: the C48-C72 numbers below are superseded for AMD by the
> C24 sweet-spot rematch further down (the C64 AMD arm was later found
> admission-bound; use the C24 section for current AMD data).

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
2. D (updated 2026-08-28, C24 sweet-spot data): dense attention remains the
   top kernel target (18.1x per-call at C24, ~9.2 ms/step; MI Triton unified
   attention is latency-bound with no split-KV — fix = aiter ASM/FMHA path);
   routed MoE is 1.50x; AR+norm and projection GEMMs are at parity (or MI
   faster after the kernel ports). The connector fill/queue path drives the
   residency gap (scheduling/connector problem, not a kernel one). AMD sweet
   spot on the merged stack: C28 (1,559.7 tok/s, TPOT p50 16.35 ms).
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
