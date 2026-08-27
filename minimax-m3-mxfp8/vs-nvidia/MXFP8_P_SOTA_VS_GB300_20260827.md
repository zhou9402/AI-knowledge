# MiniMax-M3 MXFP8 P-only: MI355X SOTA stack (integration/p1) versus GB300

Date: 2026-08-27. Companion to `MXFP8_MI355X_VS_GB300_TRACE.md` (the older
baseline); this record re-measures the MI355X side with the current SOTA
software stack and reuses the previously published GB300 numbers unchanged.

## What changed on the MI355X side

Stack: `integration/p1` @ fe8c926 (= kimi/main-cleanup + opt/moe-fp8q-variant
+ opt/dense-attention-fmha + opt/norm-rope-misc-fusion +
opt/sparse-attention-paged), run from worktree
`/mnt/vfs/homes/peiyuanz/vllm-m3-wt-p1` with the source overlay mounted.

Production toggles (identical in both runs below):

- `M3_AITER_CONFIG_FMOE=<p1 worktree>/benchmarks/kernels/minimax_m3/minimax_m3_mxfp8_fp8q_prefill_recommended.csv`
  (fp8q MoE kernel for the 16384-token bucket)
- `M3_SHUFFLE_KV_CACHE_LAYOUT=1`, `M3_USE_NATIVE_KV_ZERO_OVERLAY=1`
  (AITER_SPARSE_PA paged sparse path + native batched KV zeroing)
- `VLLM_ROCM_AITER_UNIFIED_ATTN_FMHA_PREFILL=1`
  (CK FMHA varlen for the 3 dense prefill layers; confirmed live in
  server.log: "prefill tokens use CK FMHA varlen")
- `M3_USE_MXFP8_TUNED_GEMM_OVERLAY=1` with dense source sha256 pin
  `37fb8890...` (verified); `M3_USE_MINIMAX_TOPK_SWEET_OVERLAY=0`
  (topk patch not compatible with this tree; irrelevant for P-only)
- Quick AllReduce INT4 active
  (`QUICK_REDUCE, AITER_CUSTOM, PYNCCL` dispatch chain), TP4, MXFP8 target
  `MiniMax-M3-MXFP8-c5454eb0-NV-KV`, EAGLE3 MXFP4 draft loaded
- PTPC (`VLLM_ROCM_MXFP8_PTPC_AITER`) and cross-layer indexer reuse
  (`index_topk_freq>1`) deliberately OFF per standing decision

Harness: `mi355x_ac750_p_only_sweep.sbatch`, closed-loop C8 P-only,
60K-120K ISL, ~90% prefix reuse (measured hit rate 0.9028), OSL1,
2 client warm-ups + 48 measured requests. Same logical workload as the old
MI355X and GB300 rows.

## Headline result

| Platform | Instrumented fresh tok/s | Control fresh tok/s | TTFT P50 (control) |
|---|---:|---:|---:|
| MI355X old baseline | 21,315.20 | 21,377.65 | 3,033.02 ms |
| **MI355X p1 SOTA** | **29,446.58** | **31,636.16** | **2,024.29 ms** |
| GB300 (unchanged) | 44,135.29 | 46,527.69 | 1,415.08 ms |

- Instrumentation overhead on the p1 stack: **6.92%**
  ((31,636.16 - 29,446.58) / 31,636.16). GB300's event overhead was 5.14%.
  Caveat: the two MI355X arms ran on different nodes (control node 04,
  instrumented node 01), so a small node-to-node component is included.
- Gap to GB300, instrumented like-for-like: **2.071x -> 1.499x**
  (44,135.29 / 29,446.58).
- Gap to GB300, control like-for-like: **2.177x -> 1.471x**
  (46,527.69 / 31,636.16).
- TTFT P50 ratio (GB300 vs MI355X control): 2.143x -> **1.430x**.
- MI355X gain from the p1 stack: +38.1% instrumented like-for-like
  (21,315.20 -> 29,446.58); +48.0% uninstrumented like-for-like
  (21,377.65 -> 31,636.16).
- Recaptured share of the GB300 gap: instrumented like-for-like the absolute
  gap shrank from 22,820 tok/s to 14,689 tok/s (**35.6% recaptured**);
  control like-for-like from 25,150 to 14,892 tok/s (**40.8% recaptured**).

## Per-stage comparison (HIP events, mean us/call)

Methodology matches the old record: eager HIP-event windows of 100 calls per
rank; the first window per rank is dropped when a stage has more than one;
values are the mean of total_us/calls across the remaining windows and all
four ranks. Stages are nested and must not be added. Dense stages
(dense_attn_core, dense_qkv_proj) have a single window per rank including
warm-up calls, so those rows are statistically weaker.

| Stage | MI355X old | MI355X p1 | GB300 | old MI/GB | new MI/GB | p1 vs old |
|---|---:|---:|---:|---:|---:|---:|
| Routed MoE (`moe`) | 3,475.273 | 3,390.490 | 1,912.037 | 1.818x | **1.773x** | -2.4% |
| Shared-expert MLP | 648.470 | 653.524 | 437.064 | 1.484x | **1.495x** | +0.8% |
| Sparse QKV projection | 502.578 | 498.515 | 306.279 | 1.641x | **1.628x** | -0.8% |
| Attention output projection | 371.800 | 364.297 | 181.330 | 2.050x | **2.009x** | -2.0% |
| Indexer | 843.501 | 865.547 | 543.389 | 1.552x | **1.593x** | +2.6% |
| AllReduce + norm (QR INT4) | 805.301 | 808.530 | 682.786 | 1.179x | **1.184x** | +0.4% |
| Sparse attention total | 2,041.734 | 1,529.065 | 1,312.677 | 1.555x | **1.165x** | **-25.1%** |
| Dense attention core | 38,219.372 | 14,843.830 | 3,945.283 | 9.687x | **3.762x** | **-61.2%** |

Additional p1 stages not in the old cross-vendor table (unchanged vs the old
MXFP8 run within noise): dense QKV projection 326.470, MLP gate/up 354.462,
MLP down 272.444, MoE router 62.068, QK-norm+RoPE+KV insert 50.659.

Stage-name mapping notes for the p1 stack:

- `sparse_attn_total` wraps `MiniMaxM3SparseAttention._run_attention`, which
  now dispatches to the AITER gluon paged-decode kernel (AITER_SPARSE_PA)
  instead of the Triton `_gqa_sparse_fwd_kernel`. Same stage boundary, new
  kernel path.
- The old `sparse_qknorm_cache_insert` stage (Triton fp8-index insert) no
  longer fires: on the shuffle path, KV insert goes through the C++
  `fused_minimax_m3_qknorm_rope_kv_insert` + aiter `reshape_and_cache`
  (asm_layout=True). The remaining `qknorm_rope_kv_insert` stage (50.7 us)
  covers the dense-layer fused op only.

## Critical-path budget, expanded by architectural multiplicity

Routed MoE, sparse attention, and indexer repeat across 57 sparse layers;
dense attention appears in 3 layers; AllReduce+norm fires twice per layer
(120x). Same expansion as the old record.

| Top-level bucket | MI355X old ms/pass | MI355X p1 ms/pass | GB300 ms/pass | p1 share |
|---|---:|---:|---:|---:|
| Routed MoE | 198.091 | 193.258 | 108.986 | 45.6% |
| AllReduce + norm | 96.636 | 97.024 | 81.934 | 22.9% |
| Sparse attention | 116.379 | 87.157 | 74.823 | 20.6% |
| Dense attention | 116.721 | 44.531 | 13.640 | 10.5% |
| Dense-layer MLP | 1.885 | 1.961 | 2.596 | 0.5% |
| **Captured total** | **529.712** | **423.931** | **281.979** | 100% |

Captured-total ratio versus GB300: 1.879x (old) -> **1.503x** (p1), matching
the instrumented E2E ratio of 1.499x.

## Where the remaining gap sits

1. **Routed MoE is now half of the captured critical path** (193.3 of
   423.9 ms) at 1.77x GB300 per call. It barely moved with the p1 stack
   (fp8q bucket variant gave only ~2.4% per call at C8 shapes). It is the
   primary remaining P-side target.
2. **AllReduce + norm (97.0 ms, 22.9%)** is at 1.18x GB300 per call with
   QuickReduce INT4 on both runs; closing it needs either further quantized
   collective gains or overlap, not kernel micro-tuning.
3. **Sparse attention** (87.2 ms) is nearly at parity per call (1.165x); the
   paged-PA rewrite captured most of the available win.
4. **Dense attention** fell 61% per call with CK FMHA but remains 3.76x
   GB300; it is now only ~10% of the captured path, so even a full fix is
   worth at most ~31 ms/pass (~7%).
5. The indexer (~2% slower than the old run, within noise) stays a shared
   headroom item on both vendors.

## Provenance

- MI355X p1 control (uninstrumented): Slurm job 3155, node
  `vultr-mi355x-04`, artifacts under
  `vllm-m3-wt-p1/benchmarks/kernels/minimax_m3/results/p1-mxfp8-p-c8-control/sweep-3155/`.
- MI355X p1 instrumented: Slurm job 3156, node `vultr-mi355x-01`, artifacts
  and per-rank stage TSVs + `summary.json` under
  `.../results/p1-mxfp8-p-c8-stage-profile/sweep-3156/stage-timing/`.
- Old MI355X baseline: job 2177 (21,315.20 instrumented; 21,377.65
  uninstrumented reference from the same study).
- GB300 P event job: 9695 (instrumented 44,135.29; control 46,527.69).
- All sbatch gates passed in both new runs: MXFP8 dispatch
  (`RocmDotScaledMxfp8LinearKernel`, `AITER_MXFP8` MoE), AITER_SPARSE_PA
  selection, native KV zeroing, QuickReduce INT4 dispatch chain, no
  emulation fallbacks.
