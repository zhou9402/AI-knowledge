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

### Matched C16 profiler comparison

The following totals sum GPU kernels across four TP ranks and streams. They
are useful for locating kernel families, but are not wall-clock shares because
overlap and synchronization waits are included.

| Kernel family | MI355X share | GB300 share | MI / GB total time |
|---|---:|---:|---:|
| Collective / synchronization | 51.41% | 34.54% | 2.909x |
| Attention + sparse indexer | 25.89% | 28.83% | 1.756x |
| MoE, including routed BMMs | 9.11% | 18.11% | 0.984x |
| MXFP8 linear + quantization | 3.10% | 10.13% | 0.599x |
| Norm / activation / RoPE | 3.04% | 1.85% | 3.223x |
| All summed kernels | 100% | 100% | 1.954x |

This profiler reduced MI355X C16 throughput by about 70%. Therefore the 51.41%
collective number must not be interpreted as the production critical path.
It mainly reflects synchronization waiting amplified by Kineto.

### Corrected MI355X D event breakdown

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

The corrected conclusion is that D is primarily limited by attention/indexer
and remaining graph work, not communication bandwidth. MoE was already at
parity in the matched trace, and the aggregate MI355X MXFP8 linear path was
faster than GB300.

## Optimization priorities

1. P: dense attention, routed MoE, and sparse attention.
2. D: dense+sparse attention and indexer; then unexplained graph work.
3. Do not use the profiler-only 51% collective share as a production claim.
4. Use clean runs for capacity and low-overhead events for attribution.

## Provenance

- MI355X P event job: 2177.
- GB300 P event job: 9695.
- MI355X D profiler job: 2221.
- GB300 D profiler job: 9724.
- Corrected MI355X D event job: 2239.
- Original working-tree report:
  `benchmarks/kernels/minimax_m3/MXFP8_PONLY_C8_MI355X_VS_GB300_KERNEL_TIMING.md`.

