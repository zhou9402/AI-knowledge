# MiniMax-M3 MXFP8: INT4 event breakdown and real-P-shape attention UT

## Conclusions

Three checks were run in parallel against the current TP4 MXFP8 P stack:

1. P-side INT4 QuickReduce is active. Communication is 18.636% of measured
   target-model GPU time, not the obsolete profiler result of roughly 40%.
2. On real production P shapes, ATOM's sparse-attention core is 2.076x faster
   than vLLM's current Triton core. This is the largest confirmed attention
   opportunity.
3. The dense-attention candidate did dispatch in the previous E2E test.
   Gather/dequant and merge do not erase the gain; random-data UT retains a
   2.12--2.16x complete-path speedup. The flat E2E test was offered-rate
   limited, and dense attention exists in only 3 of 60 M3 layers.

## 1. Current P event distribution with INT4 QuickReduce

Job 3016 used GPU events, not Kineto/torch profiler: TP4, MXFP8, 60K--120K
input, eight prefix sessions, about 90% prefix reuse, OSL=1, and 3.45 offered
QPS. It completed 96/96 requests with TTFT P50 1.952 s.

| Exclusive category | Rank-MAX GPU share |
|---|---:|
| Sparse-attention module compute | 39.484% |
| MoE compute | 31.668% |
| INT4 QuickReduce communication | 18.061% |
| Dense-attention compute | 6.712% |
| Other norm/model compute | 2.422% |
| Dense-layer MLP compute | 1.078% |
| Other all-reduce | 0.575% |
| **Total** | **100.000%** |

INT4 QuickReduce is 96.91% of measured all-reduce time, and every measured
QuickReduce event used the INT4 codec. The sparse module further divides into
Indexer/Top-K 10.946%, sparse core 14.141%, QKV 6.254%, QK-norm/cache 3.257%,
and output projection/other 4.886%. MoE divides into routed experts 23.439%,
shared expert 7.434%, and router 0.795%.

Event instrumentation reduced achieved QPS from the uninstrumented 3.429 to
2.987 (-12.89%). These percentages are attribution only; the clean run remains
the capacity result.

## 2. Real production P-shape attention UT

The exact ragged batch was taken from the ATOM production dispatch log:
query lengths `[4161, 7695, 4480]`, cached prefixes
`[73856, 68352, 67072]`, and 16,336 total fresh query tokens.

| Subpath | vLLM | ATOM | Result |
|---|---:|---:|---:|
| Sparse-attention core | 1227.890 us | 591.404 us | ATOM 2.076x faster |
| Indexer score | 459.644 us | 469.604 us | vLLM 2.17% faster |
| Top-K + sparse metadata | 298.202 us | 276.842 us | ATOM 1.077x faster |
| Indexer + Top-K combined | 757.846 us | 746.446 us | ATOM 1.015x faster |

Sparse correctness passed (`relL2=0.025928`, cosine `0.999664`). Indexer
outputs match after scale normalization, and final Top-K/block-table/context
outputs are exact. Cross-layer `index_topk_freq=4` reuse is separate from this
kernel-only comparison and remains a quality-changing optimization.

## 3. Dense-attention dispatch and overhead ablation

The previous E2E candidate log contains both `cp_mha_gather_cache_kernel` and
`merge_attn_states_kernel`, proving that the candidate prefix-hit path ran.

| Shape | Gather/dequant | FMHA | Merge | Candidate total | Unified | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| b1, q8192, prefix73728 | 0.060 ms | 4.153 ms | 0.232 ms | 4.431 ms | 9.549 ms | 2.155x |
| b2, q4096, prefix36864 | 0.060 ms | 2.104 ms | 0.159 ms | 2.293 ms | 4.869 ms | 2.123x |
| b4, q2048, prefix18432 | 0.057 ms | 1.067 ms | 0.081 ms | 1.178 ms | 2.509 ms | 2.130x |

Gather plus merge costs only 0.14--0.29 ms. The former 3.0--3.3x result used
zero-filled timing inputs and overstated the gain; 2.12--2.16x on seeded
random production-like data is authoritative. Correctness passed with
relative L2 0.00285--0.00398 and cosine 0.999992--0.999996.

## Priority

1. Integrate or reproduce the ATOM sparse-attention core on vLLM's production
   FP8 page-128 contract.
2. Measure a clean boundary A/B for dense attention; do not infer capacity from
   the prior offered-3.45-QPS run.
3. Keep fused Top-K/metadata as a smaller opportunity. Do not attribute ATOM's
   cross-layer reuse gain to the kernel itself.

