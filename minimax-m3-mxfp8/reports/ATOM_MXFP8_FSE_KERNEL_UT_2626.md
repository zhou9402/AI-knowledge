# ATOM MXFP8 fused-shared-expert screening UT

## Result

The ATOM-style E129/top-5 AITER FlyDSL path is a promising **decode-only**
candidate, but it is not a prefill candidate with the current tuned vLLM
baseline.

| Production bucket | Current routed E128/top-4 | ATOM E129/top-5 | Delta |
|---|---:|---:|---:|
| D, M=64, full `fused_moe` | 279.646 us | 282.147 us | +0.89% |
| D, M=64, stage1+stage2 only | 242.278 us | 254.846 us | +5.19% |
| P, M=16384, full `fused_moe` | 1887.40 us | 2159.66 us | +14.42% |
| P, M=16384, stage1+stage2 only | 1724.64 us | 2064.11 us | +19.69% |

The M=16384 E128 numbers above used AITER's heuristic row. Production vLLM
already has a better explicitly tuned E128/top-4 row at 1621.71 us for
stage1+stage2, making E129/top-5 approximately 27.3% slower than the actual
routed-prefill baseline before accounting for the separate shared MLP.

For decode, adding the fifth expert costs only 0.89% in the full kernel probe.
The current vLLM event trace attributes substantially more time to the separate
shared-expert MLP, so fusion has enough headroom to be useful. This is a
screening result: the next UT must compare E129/top-5 directly against the exact
same E128/top-4 output plus the same quantized shared MLP weights.

That exact D-side follow-up completed in job 2642:

| M=64 path | Median | P90 |
|---|---:|---:|
| E128/top-4 + production MXFP8 shared MLP | 464.043 us | 470.604 us |
| E129/top-5 fused | 429.963 us | 436.163 us |

The exact fused path is **1.0793x faster** (7.93% lower median time). It uses
the same hidden states, routed ids/weights, quantized routed weights, and
quantized shared-expert weights. Output relative L2 is 0.005119, cosine is
0.999987, and both outputs are finite. This passes the D-only integration gate.
It does not change the prefill rejection above.

## D-only integration smoke

The minimal ModelOpt MXFP8 compatibility change was exercised in a production
DecodeBenchConnector run at C=22. The authoritative comparison is a sequential
baseline/candidate pair on the same node and allocation. It uses the same tuned
MXFP8 dense-GEMM/FMoE stack, input distribution, 132 requests per side, and
exact OSL=600.

| C=22 D-only metric | Paired baseline | Fused E129/top-5 | Delta |
|---|---:|---:|---:|
| Request QPS | 2.0732 | 2.1980 | +6.02% |
| Output tok/s | 1243.94 | 1318.78 | +6.02% |
| Total tok/s | 187544.73 | 198827.25 | +6.02% |
| P50 TPOT | 16.228 ms | 15.530 ms | -4.30% |
| Mean TPOT | 15.811 ms | 15.350 ms | -2.92% |
| P50 TTFT | 407.083 ms | 324.163 ms | -20.37% |

All 264 paired requests completed, every output was 600 tokens, and synthetic
EAGLE3 acceptance was aligned at 53.36% versus 53.43%. The server logs prove
E128/top-4 baseline and E129/top-5 candidate dispatch. Both paired runs pass the
16.6 ms P50 TPOT SLA; the candidate adds 6.02% throughput headroom. Treat the
TTFT movement as run variance rather than an FSE effect because this is a
decode-side kernel change.

Job 2658's two benchmark halves completed successfully. Slurm reports a final
shell failure because the old tuned-run postcondition expected the E128/top-4
FMoE log signature after the candidate run; the postcondition has been updated
to recognize the verified E129/top-5 signature.

An earlier cross-run comparison showed +19.28%, but its baseline came from a
different run and node and is not used as the claimed gain. Two additional
candidate-only runs produced P50 TPOT of 16.41 and 15.83 ms, confirming dispatch
and correctness while also demonstrating why a same-node paired comparison is
necessary.

## Correctness

Both variants completed AITER's strict MXFP8 reference gate without NaN/Inf:

| M | E128/top-4 logits diff | E129/top-5 logits diff |
|---:|---:|---:|
| 64 | 0.00641474 | 0.00650250 |
| 16384 | 0.00634095 | 0.00633636 |

The elementwise `atol=rtol=0.01` check is too strict for the accumulated MXFP8
MoE output and reports a mismatch for both paths. AITER's strict gate accepts
the run because the normalized logits difference remains below 0.01. The
candidate does not worsen that metric materially.

## Dispatch

- E128/top-4 M=64 used the heuristic two-stage FlyDSL path.
- E129/top-5 selected the shipped MiniMax-M3 tuned kernels at both M=64 and
  M=16384.
- The production vLLM logs remain E128/top-4, proving fused shared expert is not
  currently active for ModelOpt MXFP8.

## Provenance

- Slurm job: 2626, `vultr-mi355x-04`, completed successfully in 90 seconds.
- Image: `atom-dev-nightly-202608191459-f86e2bb2.sqsh` (SHA recorded with the
  run).
- Raw run directory:
  `/mnt/vfs/homes/peiyuanz/vllm-minimax-kernel-next/benchmarks/kernels/minimax_m3/results/atom-mxfp8-fse-probe-2626/`
- Exact combined-path run:
  `/mnt/vfs/homes/peiyuanz/vllm-minimax-kernel-next/benchmarks/kernels/minimax_m3/results/mxfp8-fse-exact-2642/`
- D-only integration smoke:
  `/mnt/vfs/homes/peiyuanz/vllm-minimax-kernel-next/benchmarks/kernels/minimax_m3/results/mxfp8-tuned-ab/d-tuned-2651/2651/`
- Same-node paired D-only A/B:
  `/mnt/vfs/homes/peiyuanz/vllm-minimax-kernel-next/benchmarks/kernels/minimax_m3/results/mxfp8-tuned-ab/d-tuned-fse-{base,on}-2658/2658/`
