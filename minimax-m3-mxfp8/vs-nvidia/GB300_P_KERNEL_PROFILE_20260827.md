# MiniMax-M3 MXFP8 P-only on GB300: reproduction + kernel-level profile

Date: 2026-08-27. Companion to `MXFP8_MI355X_VS_GB300_TRACE.md` (event-level
record; GB300 numbers there came from event job 9695) and
`MXFP8_P_SOTA_VS_GB300_20260827.md` (MI355X p1 SOTA re-measure). This record
(a) re-runs the GB300 P-only C8 point to verify the published numbers and
(b) adds the missing **kernel-level** breakdown from a torch-profiler capture.

## 1. Reproduction (job 10376)

Harness `scripts/gb300_component_point.sbatch` (M3_ROLE=prefill, C8), same
logical workload as the published point: 60-120K ISL
(`p-only-zijing-random60k120k`), ~90% prefix reuse, OSL1, 2 client warm-ups,
48 measured requests. Server: image
`vllm-m3-gb300-arm64-cu130-ae2b41b.sqsh` (sha-pinned), TP4, FP8 KV,
FlashInfer backend, `moe_backend=flashinfer_trtllm`, MBT **32768** (GB300 uses
2x the MI355X MBT of 16384), `enforce_eager=True`, no speculative config.
Node `pod4-gb300-4-tray16-f3`. No profiler, no event instrumentation (clean
control-style run).

| GB300 P-only C8 | fresh tok/s | req/s | TTFT P50 | TTFT P90 | prefix hit |
|---|---:|---:|---:|---:|---:|
| Published control (job 9695-era) | 46,527.69 | 5.84 | 1,415.08 ms | — | ~0.90 |
| Published instrumented (job 9695) | 44,135.29 | 5.54 | — | — | ~0.90 |
| **Repro job 10376 (clean)** | **48,313.79** | 6.06 | **1,299.54 ms** | 1,408.99 ms | 0.9044 |

Verdict: **the published GB300 numbers reproduce** — the clean re-run lands
+3.8% above the published control (44.1-46.5K bracket) and TTFT P50 is 8%
better (1,300 vs 1,415 ms). Differences are within node/run variance
(different tray node, same image + workload). The old data is accurate; if
anything it was slightly conservative.

Artifacts:
`m3-gb300-components/results/mxfp8-repro-20260827/prefill-only/c8-10376/`
(tray-summary.json, engine-0/bench/vllm-bench-result.json).

## 2. Kernel-level profile (job 10387)

Same harness + `profiler_config` (torch profiler, `record_shapes`, gzip) and
client-driven `/start_profile` `/stop_profile`. The captured window on each
rank covers **24 engine steps / 637,785 scheduled tokens** (one full
cache-cold pass over the 8-prompt 60-120K set: 13x 32768-token steps + 11
tail steps; 1368 = 57 sparse layers x 24 steps confirms the layer math).
Profiled throughput 40,224 tok/s (-17% vs the clean 10376 run) — profiler
overhead is real, so the table below is for **kernel identity and family
composition only**; absolute budgets stay with the event record (job 9695).

Rank 0 trace:
`results/mxfp8-kernel-profile/prefill-only/c8-10387/server-0/torch-traces/dp0_pp0_tp0_dcp0_ep0_rank0.*.pt.trace.json.gz`
(17 MB gz; ranks 1-3 alongside). Total kernel time in window 11,870 ms over
39.1 s wall.

### Kernel family table (rank 0, whole window)

| Kernel family | Total ms | Calls | Mean us | % kernel | ms / pass (24) |
|---|---:|---:|---:|---:|---:|
| NCCL AllReduce bf16 RING_LL | 2,509.0 | 2,567 | 977.4 | 21.1% | 104.5 |
| Dense blockscaled MXFP8 GEMM (`mm_mxfp8`, cutlass sm100) | 1,791.8 | 5,760 | 311.1 | 15.1% | 74.7 |
| Sparse attention fwd (`fwdSparseAttentionForwardSm100`) | 1,149.5 | 1,368 | 840.3 | 9.7% | 47.9 |
| Routed MoE gate/up BMM (MxE4m3 x MxE4m3, swiGlu) | 978.6 | 1,311 | 746.5 | 8.2% | 40.8 |
| FlashInfer TRT-LLM fused AllReduce+norm (twoshot) | 923.5 | 360 | 2,565.4 | 7.8% | 38.5 |
| Elementwise copy/add (aten, see notes) | 751.9 | 3,002 | 250.5 | 6.3% | 31.3 |
| Routed MoE down BMM (bf16 x MxE4m3) | 622.0 | 1,368 | 454.7 | 5.2% | 25.9 |
| Fused add+RMSNorm (flashinfer) | 527.2 | 2,520 | 209.2 | 4.4% | 22.0 |
| Sparse attention combine (`SparseAttentionForwardCombine`) | 423.3 | 1,368 | 309.4 | 3.6% | 17.6 |
| MXFP8 quantize (swizzled + linear) | 378.7 | 7,128 | 53.1 | 3.2% | 15.8 |
| MoE finalize (vec-load) | 355.2 | 1,368 | 259.6 | 3.0% | 14.8 |
| Indexer block-score FMHA (`Sm100FmhaFwdKernel`, SparseAttnMode=1) | 354.0 | 1,368 | 258.8 | 3.0% | 14.7 |
| Indexer top-k (`IndexerTopKWithSortKernel<16>`) | 310.7 | 1,368 | 227.1 | 2.6% | 12.9 |
| Dense attention (`fmhaSm103aKernel_QkvE4m3...PagedVarSeq`) | 241.4 | 72 | 3,352.4 | 2.0% | 10.1 |
| Fused QK-norm+RoPE+KV insert (`fusedMiniMaxM3QNormRopeKVInsertKernel`) | 187.0 | 1,440 | 129.9 | 1.6% | 7.8 |
| cutlass tf32 GEMM (s128x256, bias+relu) | 159.9 | 1,254 | 127.5 | 1.3% | 6.7 |
| silu_and_mul (shared expert) | 49.6 | 1,440 | 34.5 | 0.4% | 2.1 |
| Sparse k2q metadata kernels | 45.5 | 6,840 | 6.7 | 0.4% | 1.9 |
| MoE router gating + routing | 50.8 | 3,933 | 12.9 | 0.4% | 2.1 |
| flashinfer plan / misc | ~30 | — | — | 0.3% | ~1.3 |

Call-count checksums per pass: 1368 = 57 x 24 (sparse-layer kernels), 72 =
3 x 24 (dense attention), 5760 = 240 x 24 (4 dense GEMMs x 60 layers:
qkv/o/gate_up/down), 2567+360 ~= 120 x 24 (two AllReduce+norm sites per
layer), 2520 = 105 x 24 (fused add+norm sites).

### Alignment with the event-level stage table (job 9695)

Event stages are nested wall windows; trace numbers are pure GPU kernel
durations (profiler-inflated run). Ratio = trace-derived per-call / event
us-per-call.

| Event stage (GB300 us/call) | Constituent kernels (trace) | Trace us/call | Ratio |
|---|---|---:|---:|
| Sparse attention total (1,312.7) | fwd 840.3 + combine 309.4 (+k2q 33.2) | ~1,183 | 0.90x |
| Indexer (543.4) | score FMHA 258.8 + topk+sort 227.1 + k2q misc | ~519 | 0.96x |
| Routed MoE (1,912.0) | gate/up 746.5 + down 454.7 + finalize 259.6 + router/routing/silu ~100 | ~1,561 | 0.82x |
| Dense attention core (3,945.3) | fmhaSm103a paged varlen | 3,352.4 | 0.85x |
| AllReduce + norm (682.8) | NCCL ring 977.4 (107/pass) + flashinfer fused twoshot 2,565.4 (15/pass) + fused add+RMSNorm 209.2 (105/pass) | ~1,375 (165 ms/pass vs event 81.9) | ~2.0x |

- The indexer, sparse-attention, routed-MoE and dense-attention event stages
  all decompose into the listed kernels at 0.82-0.96x of the event window
  (event windows include CPU/launch gaps, so sub-1.0x ratios are expected).
  Kernel identity is confirmed one-to-one.
- **AllReduce+norm is the one misaligned row** (~2x). The NCCL ring kernels
  embed peer-wait time; in the profiler-slowed run (17% throughput loss,
  eager mode) the 4 TP ranks skew against each other and the wait is billed
  to the AR kernel. Treat the AR split between genuine comms and wait as
  unresolved; the event number (682.8 us/call) remains authoritative.
- The `Sm100FmhaFwdKernelTmaWarpspecialized` with `SparseAttnMode=1` (1368
  calls) is attributed to the **indexer block-score** stage, not to sparse
  attention proper: adding it to the indexer matches the event stage to 96%,
  while the sparse-attention event stage is already covered to 90% by
  fwd+combine alone.
- 6.3% of kernel time is plain `aten::copy_`/add elementwise work
  (1,392 large direct_copy calls at ~404 us, ~58/pass ≈ 1 per sparse layer)
  — the single largest "unlabeled" family; candidate for fusion/elimination.

## 3. Notes and leftovers

- The profiler capture is a cache-cold full pass (all 637K input tokens
  scheduled); per-token kernel composition is identical to the 90%-hit steady
  state, but per-pass token counts differ (32768-token chunks throughout).
- MI355X side comparison for the same families lives in
  `MXFP8_P_SOTA_VS_GB300_20260827.md` (p1 stack) — e.g. the sparse-attention
  core on MI355X is the AITER gluon paged-decode kernel (1,529 us/call event)
  vs GB300's fwd+combine pair (~1,150 us/call trace / 1,313 event).
- Decode-side kernel mapping remains as in the old record's D table; this
  record is P-only.
- If a cleaner AR split is needed, re-profile with `active_iterations` small
  and compare across ranks, or use nsys with NVTX ranges around the
  collective only.

## Provenance

- Repro (clean): Slurm job 10376, node pod4-gb300-4-tray16-f3,
  `results/mxfp8-repro-20260827/prefill-only/c8-10376/`.
- Profiled: Slurm job 10387, node pod4-gb300-4-tray18-f3,
  `results/mxfp8-kernel-profile/prefill-only/c8-10387/`
  (4 rank traces + profiler_out_*.txt). Two earlier attempts (10378 FAILED,
  10381 NODE_FAIL) produced no data.
- Published reference numbers: `MXFP8_MI355X_VS_GB300_TRACE.md` (GB300 event
  job 9695: instrumented 44,135.29 / control 46,527.69 fresh tok/s).
