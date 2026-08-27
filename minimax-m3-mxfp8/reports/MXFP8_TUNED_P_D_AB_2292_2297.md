# MiniMax-M3 MXFP8 tuned-kernel P-only / D-only A/B

## Contract

- MI355X, TP4, official MXFP8 target, FP8 main/index KV.
- Baseline and candidate ran sequentially on the same nodes.
- P-only: C8, OSL1, 60K--120K ISL, 90% prefix reuse, two client
  warmups, 6C formal requests, three trials.
- D-only: C16 and C22 interleaved, DecodeBenchConnector, OSL600,
  synthetic EAGLE3 acceptance, 6C formal requests, three trials.
- Candidate: exact-bucket dense MXFP8 GEMM selector plus FMoE CSV rows
  M=2/4/8/16/16384. M8192 prefill FMoE is intentionally excluded.

## Correctness and validity

- Dense GEMM candidates passed the dequantized-BF16 reference gate with
  relative error about 0.0266, below the 0.05 threshold.
- FMoE candidates passed the official AITER cosine/Status gate; stage-1
  cosine error is 4.2%--5.1%, stage-2 error is 0%.
- Graph validation produced finite outputs and eager/graph cosine error at
  most 3.5e-6.
- Serving logs prove tuned dense GEMM and FMoE kernels actually dispatched.
- P: all 144 formal requests passed; prefix hit 90.275%; OSL1.
- D: all 684 formal requests passed; OSL600; acceptance 52.89%--53.49%.
- No request failures or forbidden MXFP8 fallback occurred.
- D-only uses DecodeBenchConnector and synthetic acceptance, so it qualifies
  performance integrity, not semantic model-output correctness.

## Three-trial median results

| Workload | Metric | Baseline | Tuned | Delta |
|---|---|---:|---:|---:|
| P C8 | QPS | 3.0724 | 3.0981 | +0.84% |
| P C8 | Uncached prefill tok/s | 24,475.34 | 24,679.90 | +0.84% |
| P C8 | TTFT P50 | 2,630.25 ms | 2,615.31 ms | -0.57% |
| P C8 | TTFT P90 | 2,825.12 ms | 2,701.50 ms | -4.38% |
| D C16 | Output tok/s | 1,077.33 | 1,100.21 | +2.12% |
| D C16 | TTFT P50 | 279.51 ms | 263.08 ms | -5.88% |
| D C16 | TPOT P50 | 13.914 ms | 13.835 ms | -0.56% |
| D C16 | TPOT P90 | 14.707 ms | 14.595 ms | -0.77% |
| D C22 | Output tok/s | 1,266.29 | 1,285.67 | +1.53% |
| D C22 | TTFT P50 | 332.06 ms | 296.43 ms | -10.73% |
| D C22 | TPOT P50 | 16.411 ms | 16.263 ms | -0.90% |
| D C22 | TPOT P90 | 17.602 ms | 17.233 ms | -2.09% |

The tuned stack improves both components, but the whole-service gain is much
smaller than isolated hot-kernel gains because the optimized kernels account
for only part of each iteration. C22 remains within the 16.6 ms TPOT P50 SLA.

Artifacts are under
`/mnt/vfs/homes/peiyuanz/vllm-minimax-kernel-next/benchmarks/kernels/minimax_m3/results/mxfp8-tuned-ab/`
for jobs 2292, 2293, 2296, and 2297. Jobs 2293 and 2297 completed all formal
measurements but their wrappers exited 1 afterward because an evidence grep
expected a setup log string that this AITER build does not emit. Actual tuned
kernel dispatch is present in both server logs; the wrapper gate has been
corrected to match that evidence.
