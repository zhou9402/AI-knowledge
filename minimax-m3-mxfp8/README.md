# MiniMax-M3 MXFP8 Optimization Summary

## Cross-platform comparison

- [MI355X versus GB300 MXFP8 trace and event comparison](vs-nvidia/MXFP8_MI355X_VS_GB300_TRACE.md)

## Executive summary

This work optimized MiniMax-M3 MXFP8 serving on MI355X with vLLM, focusing on
the production TP4 prefill and decode paths.

The current **deploy-safe** component limits are:

| Role | Best qualified rate | SLA metric | Result |
|---|---:|---:|:---:|
| P-only | 3.45 QPS | TTFT P50 2.492 s | Pass |
| D-only | 2.20 QPS | TPOT P50 16.439 ms | Pass |

With 18 TP4 engine slots on 72 GPUs, the optimal integer packing is **7P +
11D**:

```text
effective QPS = min(7 * 3.45, 11 * 2.20) = 24.15
total TPM = 24.15 * 86,656.702 * 60 = 125.566M
```

This is:

- **+4.55%** versus the same final stack with static sparse-cache fusion off
  (`120.106M TPM`).
- **+6.68%** versus the previously recalled `117.7M TPM` result. This second
  comparison is a reference comparison, not a strictly matched kernel-only
  A/B.

The deploy-safe production stack reaches 3.429--3.45 P-only QPS. The separate
`opt/sparse-attention-paged` path already implements and validates AITER Gluon
sparse paged attention and raises the P-only SLA boundary to 3.75 QPS. ATOM
still reaches 4.959 QPS under the same SLA workload. A real-shape UT shows only
a 1.53% combined Indexer/Top-K difference, so the indexer kernel itself is not
the principal remaining gap.

## Current decision snapshot

| Optimization | Measured result | Status |
|---|---:|---|
| AITER sparse paged attention | Same-load fresh tok/s +3.4%; P SLA boundary 3.45 to 3.75 QPS (+8.9%) | P-only complete; PD layout integration pending |
| Exact-shape MXFP8 dense GEMM tuning | P +0.84%; D +1.53--2.12% | Deploy-safe |
| D Top-K/index-score bucket | D +4.02% | Deploy-safe |
| BF16 EAGLE GEMM tuning | Kernel +3.04--25.81% | Deploy-safe |
| Sparse QK-norm/cache fusion | D +4.10%; limit 2.10 to 2.20 QPS | Deploy-safe |
| Fused routed + shared expert | D +6.02% at C22 | Requalify with final stack |
| Top-K + metadata fusion | Real P shape: subpath +7.16%; Indexer + Top-K combined +1.53% | UT-qualified only |
| Cross-layer Top-K reuse | P 3.45 to 3.83 QPS | Rejected: quality drift |
| ATOM-style dense attention | Random-data full path 2.12--2.16x; prior E2E was offered-rate limited | Boundary A/B needed |
| Shuffled KV/native-zero path | P benefit | Rejected: PD output corruption |
| AITER sparse attention replacement | About 2.7x slower than Triton | Rejected |

## Current INT4 event distribution

The current P stack was measured with synchronized GPU events and P-side INT4
QuickReduce, not profiler attribution. Sparse attention is 39.484%, MoE
31.668%, INT4 QuickReduce 18.061%, dense attention 6.712%, and other work
4.075%. Other all-reduce is 0.575%, so total communication is 18.636%.
QuickReduce events were all INT4 and account for 96.91% of measured all-reduce
time. This supersedes the obsolete roughly 40% communication trace, which did
not use P-side INT4 QuickReduce.

These event shares are not directly comparable to ATOM's profiler shares.
Instrumentation reduced achieved QPS by 12.89%, so clean uninstrumented runs
remain authoritative for capacity. This event run selected the Triton sparse
kernel; it did not enable the already-validated AITER sparse-paged path.

The remaining blocker is specifically PD layout compatibility. The failed
job-2390 integration used shuffled page-16 storage on P but unshuffled
page-128 storage on D, so raw MoRI KV transfer copied bytes between different
layouts and corrupted output. This does not invalidate the P-only optimization.

## Why the dense-attention UT gain did not translate to E2E

The production-shape UT compared the current AITER Unified direct
paged-FP8 path with an ATOM-style FP8 KV gather/dequant + BF16 FMHA +
prefix-chunk merge pipeline:

| UT shape | Current | ATOM-style | Speedup |
|---|---:|---:|---:|
| q8192, prefix73728 | 9.549 ms | 4.431 ms | 2.155x |
| 2 x q4096, prefix36864 | 4.869 ms | 2.293 ms | 2.123x |
| 4 x q2048, prefix18432 | 2.509 ms | 1.178 ms | 2.130x |

Correctness passed (`relL2=0.00285--0.00398`, cosine at least `0.999992`).
However, the short E2E A/B was flat: the control achieved 3.2785
QPS and the candidate achieved 3.2762 QPS at the same offered rate.

The dispatch question is now closed: the E2E log contains both the candidate
gather and merge kernels. Gather plus merge costs only 0.14--0.29 ms, so it
does not erase the gain. The old 3.0--3.3x estimate was inflated by zero-filled
timing data; the random-data full-path result above is authoritative. The flat
E2E result is explained by the offered-rate ceiling plus limited coverage:
only 3 of M3's 60 layers use dense attention. A clean boundary A/B is needed.

## P-only attention implementation matrix

This matrix refers to the compared TP4 MXFP8 P-only results. ATOM dispatch was
confirmed with the exact 5-QPS production configuration in Slurm job 2956.

The ATOM column corresponds to the run that reached 5.0 offered QPS / 4.959
achieved QPS: TP4, 60K--120K ISL, eight prefix sessions, about 90% prefix
reuse, OSL=1, MBT=16384, block size 128, FP8 main/index KV, prefix caching,
`cudagraph_mode=FULL`, `ATOM_FORCE_ATTN_TRITON=1`, and
`use_index_cache=true,index_topk_freq=4`. Its MXFP8 MoE weights were retained
while dense linear layers used the PTPC-FP8 online-quant configuration.

| Stage | ATOM P | vLLM deploy-safe P | Important difference |
|---|---|---|---|
| Dense attention selection | Generic `Attention`; prefix-hit P path runs `_gather_prefix_and_concat_kv` followed by AITER FlashAttention | `ROCM_AITER_UNIFIED_ATTN` | ATOM does not use Unified Attention for this run |
| Dense attention core | `cp_mha_gather_cache_kernel` then `aiter::fmha_fwd_hd128_bf16_causal_group`; 0.78% + 11.07% of rank-0 trace GPU time | `aiter.ops.triton.unified_attention.unified_attention`, directly consuming paged FP8 KV; trace kernels `kernel_unified_attention` and `kernel_unified_attention_2d` | Gather/dequant + BF16 FMHA versus direct FP8 paged attention |
| Sparse index cache | FP8 | FP8 | Same dtype, different surrounding layout/dispatch |
| Sparse score/Top-K | ATOM M3 Triton indexer; `use_index_cache=true,index_topk_freq=4` | AMD Triton `_index_block_score_kernel` + `_topk_index_kernel`; production-safe config recomputes every sparse layer | ATOM recomputes once per four sparse layers and reuses selected blocks; this is an algorithmic quality trade-off, not a kernel-only speedup |
| Sparse metadata/layout | Top-K path emits the shuffled physical page-16 sparse table directly | Top-K ids stay in the shared buffer and feed the logical page-128 path | ATOM fuses table emission; vLLM avoids the shuffled page-16 contract |
| Sparse attention core | `_run_prefill_fp8_gluon` -> `pa_decode_gluon`; main trace symbol `paged_attention_decode_sliding_window_head_1`; physical page-16 SHUFFLE FP8 KV; about 8.72% of rank-0 trace GPU time | AMD Triton `_gqa_sparse_fwd_kernel`, reading packed logical page-128 FP8 K/V directly | Gluon decode-like per-query kernel versus Triton prefill kernel |
| Prefill graph behavior | Eager prefill despite `cudagraph_mode=FULL` | `FULL_DECODE_ONLY`, so P is eager | Effectively aligned; not the gap |

`PTPC-FP8` in the ATOM run describes online quantization of dense linear
layers/projections. It must not be confused with the attention core itself.

The exact ATOM P dispatch is now closed. A matched production-shape P-side UT
also shows the ATOM sparse core is 2.076x faster than vLLM's current
Triton core, excluding score, Top-K, metadata construction, and cross-layer
reuse. The trace percentages are attribution evidence only because profiling
perturbs timing; the matched GPU-event UT is the performance comparison.

## Update policy

This repository is the canonical record for subsequent MiniMax-M3 MXFP8
optimization results. New qualified measurements, rejected candidates, and
changes to the production recommendation should be appended here together
with their workload contract and evidence.

## Benchmark contract

- Hardware: MI355X, TP4.
- Precision: official MiniMax-M3 MXFP8 weights with FP8 main/index KV cache.
- P-only: 60K--120K ISL, about 90% prefix reuse, OSL=1.
- D-only: `DecodeBenchConnector`, OSL=600, EAGLE3 synthetic acceptance.
- Acceptance gate: 51.3--55.3%.
- SLA: TTFT P50 below 3 s; TPOT P50 below 16.6 ms.
- Clean E2E runs determine throughput and SLA. HIP events and profiler traces
  are used only for attribution because instrumentation changes performance.
- Optimized dispatch is checked fail-closed through server logs. Unmeasured
  shapes retain the production default.

## Validated optimizations

### 1. Exact-shape MXFP8 dense GEMM selection

Production shapes were searched and independently validated under interleaved
HIP Graph replay. Only exact measured M/N/K buckets are selected; adjacent or
unknown shapes fall back to the heuristic.

Representative graph-level gains:

| Dense family | Validated gain |
|---|---:|
| Shared gate/up | 11.42--28.64% |
| Sparse QKV | 30.21--57.09% |
| Attention O projection | 10.60--13.66% |

The first matched three-trial component A/B, before the later final stack, gave:

| Workload | Throughput gain |
|---|---:|
| P C8 | +0.84% |
| D C16 | +2.12% |
| D C22 | +1.53% |

This confirms that large isolated kernel gains translate only partially to
whole-service gains.

### 2. Exact Top-K/index-score sweet-point bucket

The production FP8 index-score/partial-Top-K path was tuned for the actual D
sweet-point shapes. Correctness retained the exact FP8 output set, BF16
recall@16 above 95%, local/init blocks, and HIP Graph capture.

| Workload | Throughput gain | TPOT P50 change |
|---|---:|---:|
| MXFP8 D C22 | +4.02% | -4.78% |
| MXFP4 D C24 | +6.86% | -6.32% |

The benefit applies to both weight formats because they share the FP8 index
cache path.

### 3. BF16 EAGLE exact GEMM rows

Exact production BF16 shapes were tuned for the EAGLE LM head, QKV, and
auxiliary FC paths. Retained graph gains are:

| Family | Retained graph gain |
|---|---:|
| LM head | 22.59--25.81% |
| QKV | 17.90--23.16% |
| Auxiliary FC | 3.04--8.36% |

All selected solution IDs were observed in runtime validation; unmeasured
shapes preserved the official dispatch.

### 4. Static-scale sparse QK-norm/cache fusion

A three-launch Triton candidate fuses the sparse QK-norm/RoPE and main/index
FP8 cache insertion path for the exact D shapes M96 and M104.

Kernel result:

| Raw M | Candidate | Baseline stage | Stage reduction |
|---:|---:|---:|---:|
| 96 | 13.300 us | about 34.3 us | 61.23% |
| 104 | 13.148 us | about 34.3 us | 61.67% |

The candidate preserved the packed page-128 cache contract and exact FP8 cache
bytes. In the matched component A/B:

| D point | Output tok/s | TPOT P50 | Result |
|---|---:|---:|:---:|
| C24 control | 1,411.999 | 15.922 ms | Pass |
| C24 fused | 1,469.878 | 15.464 ms | Pass |
| C26 fused | 1,455.526 | 16.774 ms | Fail |

At C24 this is **+4.10% throughput** and **-2.88% TPOT P50**. In the 10-minute
open-loop sweep it moved the D limit from 2.10 to 2.20 QPS, producing the
matched **+4.55% cluster-capacity gain** used in the final TPM calculation.

### 5. Fused routed and shared expert for decode

The ATOM-style E129/top-5 path fuses the four routed experts and the shared
expert into one MoE execution. It was enabled for ModelOpt MXFP8 only when all
expert projections are quantization-compatible.

Exact M64 UT:

| Path | Median |
|---|---:|
| E128/top-4 + separate MXFP8 shared MLP | 464.043 us |
| E129/top-5 fused | 429.963 us |

The fused path is **7.93% faster**, with relative L2 `0.005119` and cosine
`0.999987`.

Same-node paired D C22 result:

| Metric | Baseline | Fused | Delta |
|---|---:|---:|---:|
| Request QPS | 2.0732 | 2.1980 | +6.02% |
| Output tok/s | 1,243.94 | 1,318.78 | +6.02% |
| TPOT P50 | 16.228 ms | 15.530 ms | -4.30% |

This is a valid D-only candidate, but it has not been combined with and
requalified against the complete deploy-safe final stack. It is therefore not
included in the 125.566M TPM claim. The same path is rejected for P: at
M16384 it is about 27.3% slower than the tuned routed-prefill baseline.

### 6. Fused Top-K and sparse-attention metadata

The Top-K output kernel was extended to materialize the sparse page table and
context lengths directly, avoiding a separate metadata path.

| Requests | Baseline | Fused | Gain |
|---:|---:|---:|---:|
| 1 | 191.682 us | 173.602 us | 10.4% |
| 2 | 110.160 us | 100.041 us | 10.1% |
| 4 | 80.961 us | 72.241 us | 12.1% |

Outputs matched exactly. Whole-model impact has not yet been measured, so this
remains a kernel-qualified candidate rather than a production TPM gain.
This 10.1--12.1% is a same-call kernel/path improvement: Top-K output and
page-table/context metadata generation are fused. It does not include ATOM's
cross-layer `index_topk_freq=4` reuse, which is a separate algorithmic change
and failed the current quality gate.

## Performance-only or rejected experiments

### Cross-layer Top-K reuse

ATOM recomputes sparse index score and Top-K once every four sparse layers.
The equivalent vLLM option reduced rank-0 kernel time by about 9%:

- index score: 1,539 calls / 847 ms to 405 calls / 224 ms;
- Top-K merge: 1,539 calls / 381 ms to 405 calls / 99 ms.

It raised the measured performance-only P boundary from 3.429 to 3.831 QPS
(+11.7%), reaching 77.3% of ATOM's 4.959 QPS. It is **not quality-qualified**:
the small logprob probe showed substantial output drift. It is excluded from
the deploy-safe capacity result.

### ATOM-style dense attention

An isolated complete pipeline compared the current AITER Unified paged-FP8 KV
path with ATOM-style FP8 KV gather/dequantization, BF16 FMHA, and prefix-chunk
merge.

| Production-like shape | Current | ATOM-style | Kernel speedup |
|---|---:|---:|---:|
| q8192 / prefix73728 | 9.513 ms | 2.900 ms | 3.28x |
| 2 x q4096 / prefix36864 | 4.857 ms | 1.545 ms | 3.14x |
| 4 x q2048 / prefix18432 | 2.490 ms | 0.830 ms | 3.00x |

Correctness passed with relative L2 `0.002867`. The E2E test did not reproduce
the gain:

| P at 3.45 offered QPS | Current | Candidate |
|---|---:|---:|
| Achieved QPS | 3.2785 | 3.2762 |
| Fresh prefill tok/s | 26,117 | 26,098 |
| TTFT P50 | 1.888 s | 1.961 s |

The throughput is effectively identical because the run is offered-rate
limited, and the 73 ms TTFT movement came from one short run. Nevertheless,
there is no demonstrated E2E benefit, so the production dense backend is
retained. The likely causes are dynamic E2E shapes and the additional metadata,
KV gather/dequantization, workspace, and scheduling costs not represented by
the fixed-shape UT.

### Rejected sparse-attention alternatives

- AITER Gluon sparse paged attention was correct but 2.70--2.73x slower than
  the current Triton implementation at the production shape.
- Changing sparse-decode split count from the production value of two was
  neutral or slower.
- Head tiles 8 and 4 measured 80.64 and 83.64 us versus the production
  51.40 us and were rejected.
- AITER OPUS and CK VSA are not drop-in M3 replacements today because their
  page layout, score contract, dtype, or causal-GQA semantics do not yet match
  the production page-128 FP8 KV path.

### Rejected MoE/config candidates

- D raw M88 routed-FMoE tuning improved graph replay by only 2.12--2.67%
  across neighboring shapes, below the stable 3% integration threshold.
- The ATOM E129/top-5 fused-expert path regressed large-M prefill.
- P M16384 routed-FMoE showed a 10.91% graph gain in isolation, but the exact
  row did not execute in the integrated production workload; no E2E gain is
  claimed.

### Rejected shuffled-KV integration

Unified attention plus shuffled KV/native zero showed a large component-level
P improvement, but the shuffled storage alias was not compatible with MoRI
transfer offsets and failed the integrated output-quality gate. Shuffle,
native zero, and the MoRI registration overlay remain disabled in the
deploy-safe result.

## Current bottleneck and ATOM gap

Matched P-only SLA measurements are:

| Runtime | Highest measured pass | Fresh prefill tok/s | TTFT P50 |
|---|---:|---:|---:|
| ATOM | 4.959 QPS | 39,497.8 | 2.274 s |
| vLLM, quality-qualified | 3.429--3.45 QPS | 27,317.3 at 3.429 | 2.492 s at 3.45 |
| vLLM, Top-K reuse performance-only | 3.831 QPS | 30,516 | 2.851 s |

The current evidence points to attention as the main remaining catch-up area,
but it must be split into distinct paths:

1. **Sparse attention and indexer:** the largest actionable target. The matched
   P UT confirms ATOM's page-16 Gluon sparse core is 1.98--2.02x faster than
   vLLM's page-128 Triton core. Cross-layer reuse is separate and still fails
   the quality gate.
2. **Dense attention:** ATOM's exact P path is now confirmed as gather/dequant
   plus BF16 AITER FMHA. The fixed-shape candidate has theoretical headroom,
   but the attempted backend substitution did not improve vLLM E2E.
3. **MoE and linear kernels:** several useful exact-shape gains are already in
   the candidate stack, but they do not explain the full ATOM P advantage.
4. **Graph mode:** ATOM's P path is eager even when configured with `FULL`;
   vLLM's `FULL_DECODE_ONLY` behavior is therefore not the P-only gap.

## Code and artifact status

The working tree contains opt-in implementations and tests for:

- exact MXFP8 dense GEMM selection;
- ModelOpt MXFP8 fused-shared-expert compatibility;
- cross-layer index Top-K reuse and logging;
- fused Top-K sparse metadata output.

These changes are not all equivalent in maturity. Dense exact selectors and
fused-shared-expert compatibility have correctness tests; cross-layer reuse is
performance-only pending model evaluation; fused sparse metadata is UT-only;
shuffled KV and the dense-attention backend replacement are rejected.

Primary supporting reports:

- [INT4 event breakdown and real-P-shape attention UT](reports/M3_MXFP8_REAL_P_EVENT_AND_UT_20260827.md)
- [Sweet-point event timing and detailed optimization log](reports/MXFP4_MXFP8_SWEETPOINT_EVENT_TIMING_2302_2305.md)
- [MXFP8 tuned P/D matched A/B](reports/MXFP8_TUNED_P_D_AB_2292_2297.md)
- [ATOM fused-shared-expert screening](reports/ATOM_MXFP8_FSE_KERNEL_UT_2626.md)
- [ATOM P versus vLLM MXFP8 gap analysis](reports/ATOM_P_VLLM_MXFP8_GAP_20260826.md)
- [ATOM P attention dispatch and sparse UT](reports/ATOM_P_ATTENTION_DISPATCH_AND_SPARSE_UT_2956_2960.md)

## Recommended next steps

1. Do not repeat sparse kernel UT or P-only sweet-point testing. Integrate the
   existing AITER sparse-paged path into the deploy-safe PD stack by making P/D
   transfer layouts compatible, then rerun the output-quality gate.
2. Keep Top-K/metadata fusion separate from cross-layer reuse and qualify its
   whole-model effect.
3. Run a clean service-boundary A/B for the confirmed dense gather + BF16 FMHA
   path; the previous fixed offered-rate test could not expose capacity.
4. Combine fused shared expert with the deploy-safe D stack and re-run the
   component boundary; only then update the production TPM calculation.
5. Run a model/task quality evaluation before considering Top-K frequency 4.

## Reports

- [Accuracy validation vs NVIDIA's published numbers](accuracy/README.md): GSM8K 94.77, GPQA-D 92.12, AA-LCR 74.00, SciCode 48.9, tau2-Telecom 92.98 — all aligned; index-cache long-context bug found (37→72).
