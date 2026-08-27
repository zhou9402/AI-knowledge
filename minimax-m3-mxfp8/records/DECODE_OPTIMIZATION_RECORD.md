# MiniMax-M3 MXFP8 Decode Optimization Record

## Objective and contract

- Hardware/runtime: MI355X, vLLM, TP4, MiniMax-M3 MXFP8.
- Workload: DecodeBenchConnector, EAGLE3 draft length 3, synthetic acceptance
  `[0.7, 0.5, 0.4]`, exact OSL=600.
- Production SLA: TPOT P50 below 16.6 ms.
- Current deploy-safe boundary: 2.20 QPS, TPOT P50 16.439 ms.
- Clean open-loop runs determine capacity. GPU events are attribution only.
- Record every correctness-qualified gain here. Retain rejected candidates
  briefly so they are not repeated.

## Current event attribution

The representative event-instrumented graph is 39.195 ms. Nested stage events
give the following optimization ranking; they are attribution estimates rather
than additive capacity measurements.

| Area | Approximate graph share | Current status |
|---|---:|---|
| Routed MoE + separate shared expert | 44.6% | Largest remaining target |
| Indexer + sparse-decode kernels | 17.3% | Top-K bucket deployed; split/merge remains |
| Sparse QKV, QK-norm/cache, projections | about 10--11% | Major exact buckets deployed |
| AllReduce/norm | about 3.3% | Low priority for D |
| Dense attention | about 2.9% | Only three layers; low priority |

## Qualified and deployed

| Change | Evidence | Capacity effect |
|---|---|---|
| Exact D Top-K/index-score bucket | D C22 throughput +4.02%, TPOT P50 -4.78% | Deploy-safe |
| Static-scale sparse QK-norm/cache fusion | D C24 throughput +4.10%, TPOT P50 -2.88% | Boundary 2.10 to 2.20 QPS |
| Exact-shape MXFP8 dense GEMM selection | D component +1.53--2.12% | Deploy-safe |
| BF16 EAGLE exact GEMM rows | Kernel +3.04--25.81% | Dispatch-qualified |

## Active candidates

### Fused routed and shared expert — highest priority

- Kernel UT: E128/top-4 plus separate shared expert 464.043 us; fused
  E129/top-5 429.963 us, a 7.93% reduction.
- Correctness: relative L2 0.005119, cosine 0.999987.
- Same-node D C22 A/B: output throughput +6.02%; TPOT P50 -4.30%.
- Missing gate: combine with the complete deploy-safe D stack, then search the
  2.20-QPS boundary with clean open-loop runs.

### Sparse decode split-K + merge

- Current per-layer events: split-K 43.281 us and merge 23.820 us.
- Across 57 sparse layers these account for about 3.82 ms, or 9.8% of the
  representative graph.
- Next gate: correctness-first UT for reducing split/merge intermediate traffic
  or launch count, followed by graph replay and only then D-only E2E.

## Rejected or exhausted

| Candidate | Result | Decision |
|---|---|---|
| AITER Gluon sparse PA for D | 138.72 us versus current Triton 51.32 us | Reject: 2.70x slower |
| Additional sparse-decode split count | Neutral or slower | Do not repeat |
| Sparse-decode head tiles 8/4 | 80.64/83.64 us versus 51.40 us | Reject |
| Generic routed-FMoE CSV tuning | Best graph gain about 1.57% | Exhausted; kernel change required |
| D INT4 QuickReduce | Communication is small and added precision/launch risk is not justified | Low priority |

## Change log

- 2026-08-27: Established the record and prioritized final-stack fused-expert
  qualification before sparse-decode kernel work.
