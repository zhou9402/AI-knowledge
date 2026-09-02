# MI355X decode kernel census (optimized indexer stack) + B300 reference

Date: 2026-09-02. Cluster: vultr-mi355x (gfx950).

Stack: `peiyuanz/indexer-graph-opt-m3amd` @ `1fc829ebf` (cudagraph-safe
materialized-argmax + Q4-MFMA indexer) on the internal-verify image, TP4,
fp8 KV, EAGLE3 GQA-V1 x3, FULL_DECODE_ONLY, at the SLA sweetspot c48
(2683 tok/s output, TPOT P50 15.68 ms, acceptance 2.60).

Every kernel is listed individually. Times are hardware-timed GPU durations
per engine step (4-rank mean); step = 15.68 ms x 2.60 = 40.6 ms (2.6 tokens
per step). GPU busy ~96%.

Caveat: the two aiter collective kernels are spin-wait kernels; raw durations
absorb rank skew under the profiler. Real transfer cost (p50 x call count) is
~2.6-3.7 ms/step; the raw spin-inflated values are shown for completeness.

## Full per-kernel table (>= 10 µs/step)

| kernel | µs/step | %step | calls/step | role |
|---|---:|---:|---:|---|
| `aiter::reduce_scatter_cross_device_store` | 26,091 (raw, spin) | — | 117 | comm |
| `_decode_index_score_fp8_q4_mfma_balanced_kernel` | 7,759 | 19.1% | 57 | indexer score (new) |
| `mfma_moe1_silu_mul_afp8_wfp8_..._swiglu_v32` | 5,987 | 14.7% | 57 | MoE GEMM1+act |
| `_mxfp8_linear_kernel` | 4,512 | 11.1% | 240 | dense GEMM |
| `paged_attention_decode_sliding_window_head_1` | 3,385 | 8.3% | 63 | sparse+dense PA |
| `mfma_moe2_afp8_wfp8_bf16_..._fp4opt_v1` | 3,212 | 7.9% | 57 | MoE GEMM2 |
| `aiter::cross_device_reduce_2stage` | 1,869 (raw, spin) | — | 13 | comm |
| `_kernel.kd` | 1,268 | 3.1% | 237 | MXFP8 activation quant (stripped Triton name; sits between gemma_rmsnorm and mxfp8_linear) |
| `_topk_topp_kernel` | 1,183 | 2.9% | 1 | sampler |
| `aiter::local_device_load_rmsnorm` | 675 | 1.7% | 117 | norm inside fused AR |
| `Cijk_Alik_Bljk_BSS...MT16x16x1024` | 627 | 1.5% | 57 | draft bf16 GEMM (tensile) |
| `_topk_index_repeated_argmax_kernel` | 602 | 1.5% | 57 | indexer TopK (new) |
| `_m3_fused_shuffle_insert_kernel` | 512 | 1.3% | 57 | fused KV/index insert |
| `ncclDevKernel_Generic_1` | 498 | 1.2% | 4 | residual NCCL |
| `fused_mx_quant_moe_sort_kernel` | 492 | 1.2% | 57 | MoE sort |
| `__amd_rocclr_copyBuffer` | 378 | 0.9% | 73.7 | host memcpy in hot path |
| `_swiglu_oai_kernel` | 317 | 0.8% | 60 | MoE act |
| `opus_moe_sorting_entry` (P23) | 311 | 0.8% | 57 | MoE sort |
| `_routed_scale_add_kernel` | 303 | 0.7% | 57 | MoE combine |
| `pa_decode_ps_reduce_hip_kernel` | 300 | 0.7% | 57 | sparse PA reduce |
| `grouped_topk_kernel` | 290 | 0.7% | 57 | router topk |
| `opus_moe_sorting_entry` (P0_v2) | 284 | 0.7% | 57 | MoE sort |
| `Cijk_...MT256x192x64` | 247 | 0.6% | 3 | draft GEMM |
| `Cijk_...MT64x48x256` | 180 | 0.4% | 5 | draft GEMM |
| `at::native::reduce_kernel` | 95 | 0.2% | 3 | |
| `elementwise_kernel_manual_unroll` | 78 | 0.2% | 4 | |
| `hgemm_bf16_32x64x128x3` | 63 | 0.2% | 2 | draft GEMM |
| `_resample_kernel` | 50 | 0.1% | 1 | spec resample |
| `add_rmsnorm_quant_kernel` (x2 variants) | 75 | 0.2% | 15 | norm+quant |
| `_gemm_a16_w16_kernel...` | 45 | 0.1% | 2 | draft GEMM |
| `index_elementwise_kernel` (x2 variants) | 51 | 0.1% | 6 | index bookkeeping |
| `bfloat16tofloat32` cast | 35 | 0.1% | 1 | |
| `Cijk_...MT64x96x256` | 34 | 0.1% | 1 | draft GEMM |
| `CatArrayBatchedCopy_contig` | 32 | 0.1% | 5 | concat |
| `reshape_and_cache_shuffle_kernel` | 31 | 0.1% | 6 | dense KV write |
| `Cijk_SB...PostGSU16` | 30 | 0.1% | 2 | draft GEMM |
| `pa_decode_ps_reduce_hip_kernel` (2nd variant) | 30 | 0.1% | 6 | PA reduce |
| `vectorized_gather_kernel` | 30 | 0.1% | 6 | gather |
| `Cijk_...MT128x192x128` | 26 | 0.1% | 1 | draft GEMM |
| `_compute_local_logits_stats_kernel` | 25 | 0.1% | 1 | sampler prep |
| `elementwise (misc)` | 22 | 0.1% | 4 | |
| `hgemm_bf16_32x64x128x4` | 20 | 0.0% | 2 | draft GEMM |
| `FillFunctor` / `CUDAFunctor_add` / `triton_poi_fused_*` | ~53 | 0.1% | 11 | elementwise |
| `_gemma_add_rmsnorm_mxfp8_quant_kernel` | 17 | 0.0% | 3 | fused norm+quant (only 3 of 60 layers!) |
| `_compute_slot_mappings_kernel` | 16 | 0.0% | 3 | spec bookkeeping |
| `fusedMiniMaxM3QNormRopeKVInsertKernel` | 15 | 0.0% | 3 | dense qknorm/rope/insert |
| `rotary_embedding_kernel` | 15 | 0.0% | 3 | draft rope |
| `act_and_mul_kernel` | 15 | 0.0% | 3 | |
| `hgemm_bf16_96x64x64x4` | 14 | 0.0% | 1 | draft GEMM |
| `elementwise_kernel_with_index` (arange) | 14 | 0.0% | 3 | |
| `__amd_rocclr_fillBufferAligned` | 14 | 0.0% | 3 | fill |
| everything <10 µs (16 kinds) | ~65 | 0.2% | ~20 | tail |

## Small-kernel optimization candidates (ranked)

1. `_kernel.kd` — standalone MXFP8 activation quant (3.1% of step). The fused
   `_gemma_add_rmsnorm_mxfp8_quant` path covers only 3/60 layers; extending
   it removes this kernel entirely.
2. MoE tail (`fused_mx_quant_moe_sort` + 2x `opus_moe_sorting` +
   `grouped_topk` + `_routed_scale_add` + `_swiglu_oai`) ≈ 2.2 ms/step as
   six launches per layer per step; merge candidates.
3. `__amd_rocclr_copyBuffer` 0.9% at 73.7 calls/step — source TBD.
4. Draft's seven untuned tensile `Cijk_*` bf16 GEMM shapes (~1.1 ms) — add
   to the bf16 tuned CSV.
5. `local_device_load_rmsnorm` (1.7%) is the norm part of the fused AR;
   only worth touching together with the AR fusion.

## B300 reference (nsys, same workload)

| bucket | MI355X c48 (this stack) | B300 c72 (its sweetspot) |
|---|---:|---:|
| indexer | ~8,400 µs/step (21%) | 8,000 (24.9%) |
| MoE | ~10,900 (27%) | 10,200 (31.7%) |
| comm (corrected) | ~2,600-3,700 (6-9%) | 2,940 (9.1%) |
| dense GEMM | 4,512 (11.1%) | 1,856 (5.8%) |
| sparse PA | ~3,700 (9.1%) | 1,006 (3.1%) |
| draft | ~1,100 (2.7%) | 872 (2.7%) |
| step | 40.6 ms | 32.2 ms |
| output tok/s | 2,683 | 4,738 |

Indexer has converged across platforms (8.4 vs 8.0 ms/step). Remaining
MI355X gaps: dense GEMM, sparse PA, and the small-kernel tail above.

## Provenance

- Traces: `vigil-results/m3-prof-opt-d-c48-mi355x/traces/` (torch profiler,
  4 ranks, c48 steady state) on vultr-mi355x; B300:
  `verda-b300-01:/mnt/nvme/peiyuanz/vigil-results/m3-prof2-d-b300/`.
- Analysis: step-anchored by kernel-count invariants (mxfp8_linear=240/step,
  index_score=57/step, PA=63/step, topk_topp=1/step); spin-inflated
  collectives corrected with the p50-per-call lower bound.
