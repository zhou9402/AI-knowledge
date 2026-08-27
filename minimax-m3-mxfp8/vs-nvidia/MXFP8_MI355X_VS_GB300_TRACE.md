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

## AMD SOTA kernel-level breakdown (2026-08-27)

torch-profiler kernel-level decomposition of the current AMD SOTA stack —
the stage table above is event-window based; this section gives the raw
per-kernel accounting (names, calls, durations, shares) for the same P-only
workload on the p1 stack.

- Job: Slurm **3173**, node `vultr-mi355x-04`, artifacts under
  `vllm-m3-wt-p1/benchmarks/kernels/minimax_m3/results/p1-mxfp8-p-c8-torch-profile-fmha/sweep-3173/`
  (rank-0 trace `dp0_pp0_tp0...rank0.*.pt.trace.json.gz`; per-kernel TSV
  alongside it as `*.kernels.tsv`).
- Stack: `integration/p1` @ `fb5d693`, source overlay from
  `/mnt/vfs/homes/peiyuanz/vllm-m3-wt-p1`; toggles: fp8q MoE CSV
  (`minimax_m3_mxfp8_fp8q_prefill_recommended.csv`), dense GEMM sweetpoint
  overlay (sha `37fb8890...`), `M3_USE_ROCM_AITER_UNIFIED_ATTN_OVERLAY=1` +
  `VLLM_ROCM_AITER_UNIFIED_ATTN_FMHA_PREFILL=1` (CK FMHA varlen dense
  prefill), shuffle KV layout + native KV zero, QuickReduce INT4, TP4, MXFP8
  target `MiniMax-M3-MXFP8-c5454eb0-NV-KV`, EAGLE3 MXFP4 draft
  (TRITON_ATTN), synthetic spec. PTPC and `index_topk_freq>1` OFF per the
  standing precision decision. **Note:** the unified-attention overlay is
  load-bearing for FMHA — the image's `rocm_aiter_unified_attn.py` predates
  it, so the env var alone is a no-op (first attempt, job 3171, lacked the
  overlay and silently ran the old Triton dense path; 3171 is kept as the
  FMHA-off A/B reference below).
- Run health: all sbatch gates passed; profiled throughput 31,303.8 fresh
  tok/s vs the 3155 uninstrumented control 31,636.2 (**1.05%** profiler
  overhead — the kernel mix is production-representative). This trace is for
  kernel attribution only; throughput conclusions stay with the
  uninstrumented runs.
- Window: 26 target-model steps (counted via `vllm::moe_forward_shared` =
  57 x 26), wall span 12,700.8 ms, GPU kernel-busy 12,486.3 ms = **98.3% of
  wall, single stream** (stream 1 has 26 calls, negligible). Each step is a
  mixed batch: one 16,384-token prefill chunk + decode/verify tokens + 3
  EAGLE3 draft forwards. Per-step values below divide by 26; shares are of
  total kernel-busy time. `execute_context_*` rows in the profiler table are
  `user_annotation` markers (not kernels) and are excluded.

### Top kernels (rank 0, full names)

| Kernel | calls | /step | total ms | ms/step | mean us | share |
|---|---:|---:|---:|---:|---:|---:|
| `quickreduce::allreduce_prototype_twoshot<AllReduceTwoshot<__half, CodecQ4<__half,4>, true>, __half>` | 3,100 | 119.2 | 2,139.8 | 82.30 | 690.3 | 17.14% |
| `kernel_unified_attention.kd` (EAGLE3 draft attn, TRITON_ATTN) | 78 | 3.0 | 1,554.3 | 59.78 | 19,927.3 | 12.45% |
| `_mxfp8_linear_kernel.kd` (Triton dot_scaled MXFP8 dense GEMM) | 6,240 | 240.0 | 1,887.2 | 72.58 | 302.4 | 15.11% |
| `mfma_moe1_silu_mul_afp8_wfp8_fp8_t128x128x256_pm1_fp8q_sort_async_gui_xcd4_swiglu_v32.kd` (routed MoE stage 1, fp8q) | 1,368 | 52.6 | 1,197.7 | 46.07 | 875.5 | 9.59% |
| `mfma_moe2_afp8_wfp8_bf16_cshuffle_t128x128x256_vscale_fix3_fp4opt_v1_pm1.kd` (routed MoE stage 2) | 1,425 | 54.8 | 975.0 | 37.50 | 684.2 | 7.81% |
| `_index_block_score_kernel.kd` (indexer scores) | 1,482 | 57.0 | 871.2 | 33.51 | 587.8 | 6.98% |
| `paged_attention_decode_sliding_window_head_1.kd` (gluon AITER_SPARSE_PA) | 1,482 | 57.0 | 805.7 | 30.99 | 543.6 | 6.45% |
| `fmha_fwd_hd128_bf16_causal_groupE.kd` (CK FMHA varlen, target dense attn) | 78 | 3.0 | 793.6 | 30.52 | 10,174.3 | 6.36% |
| `_topk_index_kernel.kd` (indexer top-k) | 1,482 | 57.0 | 380.8 | 14.65 | 257.0 | 3.05% |
| `_gemma_fused_add_rmsnorm_kernel.kd` (post-AR add+norm) | 3,120 | 120.0 | 359.3 | 13.82 | 115.2 | 2.88% |
| `_kernel.kd` (`vllm::mxfp8_quantize`, dense-GEMM activation quant) | 6,240 | 240.0 | 337.4 | 12.98 | 54.1 | 2.70% |
| `aten::add` big bf16 (57 MoE shared-expert combine + 3 EAGLE3 aux) | 1,560 | 60.0 | 150.8 | 5.80 | 96.7 | 1.21% |
| `__amd_rocclr_copyBuffer.kd` (scheduler/metadata copies) | 2,366 | 91.0 | 103.2 | 3.97 | 43.6 | 0.83% |
| `dynamic_per_group_scaled_quant` (aiter, in-MoE per_1x32 quant) | 1,482 | 57.0 | 96.8 | 3.72 | 65.3 | 0.78% |
| `aten::mul_` bf16x2.0 (MoE `routed_scaling_factor`) | 1,482 | 57.0 | 80.3 | 3.09 | 54.2 | 0.64% |
| `Cijk_Alik_Bljk_BSS_..._MT128x64x128_...` (hipBLAS bf16 GEMM, MoE router gate) | 1,311 | 50.4 | 80.4 | 3.09 | 61.3 | 0.64% |
| `fusedMiniMaxM3QNormRopeKVInsertKernel<bf16,u8,fp8,_,true,...>` (sparse QK-norm+RoPE+KV insert) | 1,482 | 57.0 | 69.8 | 2.68 | 47.1 | 0.56% |
| `opus_moe_sorting_entry` P23+P0+P1+clear (MoE sorting) | 5,700 | 219.2 | 80.6 | 3.10 | ~14 | 0.65% |
| `cross_device_reduce_2stage<bfloat16_t,4>` (aiter custom AR remainder) | 124 | 4.8 | 55.1 | 2.12 | 444.1 | 0.44% |
| `bf16gemm_bf16_tn_256x256` (draft bf16 GEMM) | 25 | 1.0 | 54.8 | 2.11 | 2,192.7 | 0.44% |
| `_swiglu_oai_kernel.kd` (shared-expert + dense-MLP activation) | 1,560 | 60.0 | 34.9 | 1.34 | 22.3 | 0.28% |
| `CatArrayBatchedCopy_contig` (spec-decode concat) | 130 | 5.0 | 28.8 | 1.11 | 221.2 | 0.23% |
| `reshape_and_cache_kernel<...,fp8,true>` (aiter sparse KV insert) | 1,482 | 57.0 | 22.6 | 0.87 | 15.3 | 0.18% |
| `grouped_topk_kernel<float,float4,4,...>` (router top-k) | 1,482 | 57.0 | 22.3 | 0.86 | 15.1 | 0.18% |
| `_build_sparse_block_table_prefill_kernel.kd` | 1,482 | 57.0 | 20.1 | 0.77 | 13.6 | 0.16% |
| `_gemm_afp4wfp4_kernel_...` (draft MXFP4 GEMM) | 50 | 1.9 | 20.7 | 0.80 | 413.3 | 0.17% |
| `mfma_moe1_silu_mul_afp8_wfp8_bf16_t128x128x256_pm1_async_gui_swiglu_v32.kd` (MoE s1, bf16 bucket) | 57 | 2.2 | 21.3 | 0.82 | 374.0 | 0.17% |
| `mxfp4_moe_sort_kernel<256,32,24,32>` | 1,425 | 54.8 | 14.5 | 0.56 | 10.2 | 0.12% |
| `_insert_index_cache_kernel.kd` | 1,482 | 57.0 | 13.1 | 0.50 | 8.9 | 0.10% |
| `cp_mha_gather_cache_kernel.kd` (paged-KV gather feeding FMHA) | 78 | 3.0 | 12.0 | 0.46 | 154.3 | 0.10% |

The 38 listed rows cover 98.7% of kernel-busy time; the tail (draft norms
`add_rmsnorm_quant` 0.53 ms/step, small-M MoE variants, RCCL remainder
`ncclDevKernel_Generic_1` 0.12 ms/step, dense-layer `rotary_embedding` /
`reshape_and_cache_flash` / `scaled_fp8_quant` 0.16 ms/step, indexing glue)
is in the TSV.

### Kernel -> stage mapping

Stage names follow the event-window stage table in this document and its p1
companion (`MXFP8_P_SOTA_VS_GB300_20260827.md`).

| Stage | Kernels (per target step) |
|---|---|
| Routed MoE (`moe`) | `mfma_moe1_*` + `mfma_moe2_*` (all bucket variants) + `opus_moe_sorting_*` + `mxfp4_moe_sort` + in-MoE `dynamic_per_group_scaled_quant` — 2395.3 ms / 92.1 ms/step total; fp8q bucket variants (`..._fp8q_sort_async_...`) carry 52.6/57 calls per step, bf16/t32 variants the small-M steps |
| Shared-expert MLP | 114 of the 240/step `_mxfp8_linear_kernel.kd` (gate_up+down) + 114 quant `_kernel.kd` + 57 of 60 `_swiglu_oai_kernel.kd`; then the `aten::mul_` (routed scale) + 57 of 60 `aten::add` (shared+routed combine) |
| Sparse QKV projection / Attention output projection / Dense MLP proj | the rest of `_mxfp8_linear_kernel.kd` + `_kernel.kd` (240/step each = 60 qkv + 60 o_proj + 57x2 shared + 3x2 dense MLP) |
| Indexer | `_index_block_score_kernel.kd` + `_topk_index_kernel.kd` (+ `_build_sparse_block_table_prefill`, `_insert_index_cache`) |
| Sparse attention total | `paged_attention_decode_sliding_window_head_1.kd` (gluon) + block-table build |
| Dense attention core | `fmha_fwd_hd128_bf16_causal_groupE.kd` + `cp_mha_gather_cache_kernel.kd` (new CK FMHA varlen path; 3 dense layers) |
| AllReduce + norm | `quickreduce::allreduce_prototype_twoshot` (QR INT4, 120/step) + `_gemma_fused_add_rmsnorm_kernel.kd` (120/step) + `cross_device_reduce_*` / `ncclDevKernel_Generic_1` remainders |
| QK-norm+RoPE+KV insert | `fusedMiniMaxM3QNormRopeKVInsertKernel` (57/step fp8-index variant + 3/step dense variant) + aiter `reshape_and_cache` (sparse, fp8) |
| MoE router | `grouped_topk_kernel` + router-gate hipBLAS GEMM (`Cijk_...`, 57/step on full-chunk steps) |
| EAGLE3 draft (not a target-model stage) | `kernel_unified_attention.kd` (3/step, TRITON_ATTN backend), `bf16gemm_bf16_tn_256x256`, `_gemm_afp4wfp4_kernel`, `_dynamic_mxfp4_quant_kernel`, aiter `add_rmsnorm_quant`, `wvSplitK_hf_sml_*`, `CatArrayBatchedCopy`, argmax `reduce_kernel` |

### Self-consistency with the event-stage table (job 3156)

Per-call kernel sums vs the stage-window means (stage windows include eager
CPU launch gaps; kernels are pure GPU busy):

| Stage | stage us/call | kernel us/call | kernel/stage |
|---|---:|---:|---:|
| AllReduce + norm | 808.5 | 690.3 (QR) + 115.2 (norm) = 805.5 | **1.00x** |
| Indexer | 865.5 | 587.8 + 257.0 = 844.8 | **1.02x** |
| Dense attention core | 14,843.8 | 10,174.3 + 154.3 = 10,328.6 | 1.44x |
| Routed MoE | 3,390.5 | 875.5 + 684.2 + ~66 (sort) + 65.3 (quant) = 1,691 | ~2.0x |
| Sparse attention total | 1,529.1 | 543.6 + 13.6 = 557.2 | ~2.7x |

AR+norm and the indexer are kernel-dense (stage ~= kernel); routed MoE and
sparse attention still lose ~2-2.7x of their stage window to eager launch
gaps — consistent with the fusion/launch-overhead directions being pursued
(norm-rope fusion, FMoE bucket tuning, residual-add fusion).

### Notable findings

1. **EAGLE3 draft attention is a first-class cost on P-only long-context:**
   `kernel_unified_attention.kd` (draft, TRITON_ATTN backend, 3 forwards per
   step over 60-120K context) burns 59.8 ms/step = **12.45% of kernel time**
   — ~2x the target model's entire sparse-attention kernel time (31.0
   ms/step). The draft does not use the unified/FMHA path;
   `opt/draft-prefill-attn` (draft FMHA prefill) exists but is not in p1.
2. **FMHA dense prefill confirmed live** (`fmha_fwd_hd128_bf16_causal_groupE`
   3/step + `cp_mha_gather_cache` 3/step), replacing the Triton
   `kernel_unified_attention_2d.kd` (9.98 ms/call) used by the FMHA-off
   control job 3171; at these shapes the CK FMHA kernel itself is at parity
   with unified-2d (10.17 vs 9.98 ms/call) plus a 154 us gather — the
   stage-level dense-attention win (-61% vs the old stack) comes from the
   old stack's much slower path, not from FMHA beating unified-2d here.
3. **Residual elementwise leftovers:** the 60 `aten::add` (5.8 ms/step) + 57
   `aten::mul_` (3.1 ms/step) are exactly the set addressed by
   `opt/residual-add-fusion` (not in p1).
4. **240 standalone activation quants per step** (`_kernel.kd`, 13.0
   ms/step) feeding the dense MXFP8 GEMMs — the AR-epilogue norm+quant
   fusion (`opt/ar-epilogue-fusion`, not in p1) targets ~65 of them.
5. MoE stage-1/stage-2 kernel mix: fp8q variants cover 52.6/57 calls per
   step (full 16,384 chunks); small-M steps fall back to the bf16/t32
   bucket variants (mfma_moe1 bf16 374 us, t32 108/58 us).

## Provenance

- MI355X P event job: 2177.
- GB300 P event job: 9695.
- MI355X D profiler job: 2221.
- GB300 D profiler job: 9724.
- Corrected MI355X D event job: 2239.
- Original working-tree report:
  `benchmarks/kernels/minimax_m3/MXFP8_PONLY_C8_MI355X_VS_GB300_KERNEL_TIMING.md`.

---

## D-only rematch at AMD sweet spot, C64 (2026-08-27, event-timed, optimized MI355X stack)

Post-kernel-port MI355X stack (ws1 MoE tuned-CSV + ws2 PTPC aiter GEMM + ws3 fused
AR+norm; ws4 gluon OFF — eager-path regression). GB300 unchanged (canonical config).
Protocol: TP4, DecodeBenchConnector (fill 0.015), EAGLE3 draft-3 synthetic
acceptance [0.7,0.5,0.4], 60K-120K token-id ISL, OSL 600, C64 closed loop.
Timing: HIP/CUDA-event stage timing, OFF leg for capacity, ON leg for stage shares
(instrumentation cost: GB300 ~11-15%, MI355X ~17.5% TPOT — stage tables are
relative shares only).

### Capacity (timing OFF, zero errors both sides)

| Platform | out tok/s | TPOT P50 | vs |
|---|---:|---:|---|
| GB300 (job 10377, repro of canonical) | **3,620.99** | **16.03 ms** | 1.00x |
| MI355X optimized (this work) | 1,724.4 | 23.43 ms | NV = **2.10x** tok/s, 1.46x TPOT |
| MI355X pre-optimization (earlier record) | 1,832.65 | 33.21 ms | — |

MI355X optimization moved TPOT 33.2 -> 23.4 ms (-29%). Raw tok/s is dominated by
the synchronous DecodeBenchConnector KV-fill TTFT at this run length (MI TTFT p90
13.7 s), so TPOT is the clean decode metric; the tok/s dip vs the old record is
wave/accounting noise, not a regression.

### Per-stage decode-step comparison (verify cudagraph, per-call us, relative shares)

| stage (per call) | MI355X opt. | GB300 | MI / GB |
|---|---:|---:|---:|
| Routed MoE (per layer, x57) | 312.7 | 220.9 | **1.42x** |
| Dense attention (per layer, x3) | 3,135 | 331.8 | **9.45x** |
| Index score+topk (x15) | 301.8 | — | n/a |
| Shared-expert MLP (x57) | 51.8 | 185.2 | 0.28x (MI faster) |
| Sparse decode kernel (x57) | 43.6 | — | n/a |
| AR+norm fused (x120) | 35.2 | 34.0 | 1.04x (parity) |
| sparse QKV proj (x57) | 18.7 | 17.7 | 1.06x |
| attn o_proj (x60) | 15.5 | 14.7 | 1.05x |

MI355X verify-graph step budget: MoE 41.0%, dense attn 20.6%, index topk 9.9%,
shared-expert MLP 6.8%, sparse decode 5.6%, AR+norm 5.0%; ~15 ms/step residual =
EAGLE3 draft forwards + sampler + scheduler (not instrumented).

### Read

- GB300's 2.1x lead at C64 is led by **dense attention (9.45x; 3 layers at 60-120K
  ctx)** and routed MoE (1.42x); AR/norm and the projection GEMMs are at parity
  after the kernel ports.
- ws3 (fused AR+norm) holds parity with trtllm allreduce fusion — the glue gap is
  closed. The remaining MI355X decode gap is dense-attention-bound.

### Artifacts

- MI355X: `vultr-mi355x:/mnt/vfs/homes/peiyuanz/m3-compare/results-nv-amd/amd/event-c64/{off,on,on-gated}/` (SUMMARY.md, stage TSVs); optimized-stack decode record in `records/DECODE_OPTIMIZATION_RECORD.md`.
- GB300: repro `results/mxfp8/decode-only/c64-10377/`, event `results/mxfp8-event-stage6/decode-only/c64-10388/` (exit-2 is a summary-jq bug; measurement complete); copied to `vultr:.../results-nv-amd/gb300/`.
