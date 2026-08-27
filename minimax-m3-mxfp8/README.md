# MiniMax-M3 MXFP8 Optimization Summary

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

The remaining vLLM-versus-ATOM P-only gap is likely concentrated in attention,
especially sparse attention and the indexer. ATOM reaches 4.959 QPS under the
same P-only SLA workload, versus vLLM's current quality-qualified 3.429--3.45
QPS.

## Current decision snapshot

| Optimization | Measured result | Status |
|---|---:|---|
| Exact-shape MXFP8 dense GEMM tuning | P +0.84%; D +1.53--2.12% | Deploy-safe |
| D Top-K/index-score bucket | D +4.02% | Deploy-safe |
| BF16 EAGLE GEMM tuning | Kernel +3.04--25.81% | Deploy-safe |
| Sparse QK-norm/cache fusion | D +4.10%; limit 2.10 to 2.20 QPS | Deploy-safe |
| Fused routed + shared expert | D +6.02% at C22 | Requalify with final stack |
| Top-K + metadata fusion | Subpath +10.1--12.1% | UT-qualified only |
| Cross-layer Top-K reuse | P 3.45 to 3.83 QPS | Rejected: quality drift |
| ATOM-style dense attention | UT about 3.0--3.3x; no E2E gain | Integration investigation |
| Shuffled KV/native-zero path | P benefit | Rejected: PD output corruption |
| AITER sparse attention replacement | About 2.7x slower than Triton | Rejected |

## MoE status versus ATOM

The latest traces attribute nearly the same fraction of GPU kernel time to
MoE:

| Runtime | MoE share of GPU kernel time |
|---|---:|
| ATOM P trace | 17.3% |
| vLLM current P trace | 17.39% |

This supports the conclusion that MoE is no longer the dominant source of the
vLLM-versus-ATOM P gap. It does **not** prove equal absolute MoE performance:
the traces use different schedules and total execution times, so only a
matched-shape absolute kernel comparison can establish parity. MoE still
accounts for about 17% of vLLM GPU time and the fused shared-expert candidate
shows a remaining D-only opportunity, but attention/indexer is now the higher
priority.

## Why the dense-attention UT gain did not translate to E2E

The isolated production-like UT compared the current AITER Unified direct
paged-FP8 path with an ATOM-style FP8 KV gather/dequant + BF16 FMHA +
prefix-chunk merge pipeline:

| UT shape | Current | ATOM-style | Speedup |
|---|---:|---:|---:|
| q8192, prefix73728 | 9.513 ms | 2.900 ms | 3.28x |
| 2 x q4096, prefix36864 | 4.857 ms | 1.545 ms | 3.14x |
| 4 x q2048, prefix18432 | 2.490 ms | 0.830 ms | 3.00x |

Correctness was acceptable (`relL2=0.002867`, max absolute error
`0.0009766`). However, the short E2E A/B was flat: the control achieved 3.2785
QPS and the candidate achieved 3.2762 QPS at the same offered rate.

The UT result therefore proves a kernel opportunity, not an integrated gain.
The leading explanations are:

1. The E2E experiment selected vLLM's standard `ROCM_AITER_FA` backend and
   forced block size 128; it did not prove that every dense layer dispatched
   through the exact ATOM-style pipeline measured by the UT.
2. Production uses dynamic query/chunk/prefix shapes. The UT used a small set
   of fixed, cache-hot shapes, while observed E2E shapes vary substantially.
3. KV gather/dequant, block-table and metadata preparation, workspace setup,
   prefix-chunk merge, and extra launches can consume the FMHA-core saving.
4. Most M3 layers use sparse attention. Replacing dense attention does not
   improve sparse GQA, index-score, or Top-K kernels.
5. The short open-loop run was offered-rate limited, so nearly identical
   achieved QPS is insufficient to expose a higher service boundary.

The decisive next measurement is a matched-request event breakdown around KV
gather/dequant, FMHA core, chunk merge, metadata preparation, and total dense
attention, together with dispatch and actual-shape logging. If the fast core is
not selected, fix dispatch; if conversion dominates, fuse it or retain an
FP8-native path; if only dynamic shapes regress, tune the E2E-observed buckets.

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

1. **Sparse attention and indexer:** the largest actionable target. Existing
   traces show `_gqa_sparse_fwd_kernel`, index score, and Top-K as major costs.
   Cross-layer reuse proves the available headroom but currently fails the
   quality gate.
2. **Dense attention:** fixed-shape UT has large theoretical headroom, but the
   attempted backend substitution did not improve E2E. The exact ATOM runtime
   path and dynamic production shapes must be reproduced before another
   integration attempt.
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

- [Sweet-point event timing and detailed optimization log](reports/MXFP4_MXFP8_SWEETPOINT_EVENT_TIMING_2302_2305.md)
- [MXFP8 tuned P/D matched A/B](reports/MXFP8_TUNED_P_D_AB_2292_2297.md)
- [ATOM fused-shared-expert screening](reports/ATOM_MXFP8_FSE_KERNEL_UT_2626.md)
- [ATOM P versus vLLM MXFP8 gap analysis](reports/ATOM_P_VLLM_MXFP8_GAP_20260826.md)

## Recommended next steps

1. Capture matched event-level traces for vLLM and ATOM using the same P-only
   request replay and compare sparse attention, indexer, and dense attention by
   exact shape.
2. Prioritize a semantically exact page-128 FP8 sparse-attention/indexer
   implementation; do not replace the current Triton sparse path with the
   slower Gluon candidate.
3. Reproduce the exact ATOM dense-attention runtime path, including dynamic
   metadata and KV conversion overhead, before another E2E A/B.
4. Combine fused shared expert with the deploy-safe D stack and re-run the
   component boundary; only then update the production TPM calculation.
5. Run a model/task quality evaluation before considering Top-K frequency 4.
