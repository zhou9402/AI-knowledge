# ATOM P Attention Dispatch and Sparse Core UT

The exact TP4 MXFP8 ATOM P run at 5.0 offered QPS was traced in Slurm job 2956.
It achieved 4.9589 QPS in the archived production run.

| Stage | Actual ATOM implementation | Trace share |
|---|---|---:|
| Dense KV gather | `_gather_prefix_and_concat_kv` / `cp_mha_gather_cache_kernel` | 0.78% |
| Dense attention | `aiter.flash_attn_varlen_func` / `aiter::fmha_fwd_hd128_bf16_causal_group` | 11.07% |
| Sparse score | `_index_block_score_kernel` | 2.66% |
| Sparse Top-K | `_topk_index_kernel` | 1.03% |
| Sparse attention | `_run_prefill_fp8_gluon` -> `pa_decode_gluon`; main symbol `paged_attention_decode_sliding_window_head_1` | about 8.72% |

ATOM's sparse path reinterprets logical page-128 FP8 KV as physical page-16
SHUFFLE blocks. It is not the Triton sparse-prefill kernel used by vLLM. Trace
shares are attribution evidence only because profiling perturbs timing.

## Matched P sparse-core UT

The GPU-event UT excludes index score, Top-K, metadata construction, and
cross-layer reuse. It uses FP8 KV with ATOM per-token scales.

| Batch | Query tokens | Prefix tokens | vLLM | ATOM | ATOM speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 8192 | 73728 | 661.445 us | 327.003 us | 2.022x |
| 2 | 4096 | 36864 | 656.406 us | 330.923 us | 1.984x |
| 4 | 2048 | 18432 | 658.966 us | 327.723 us | 2.011x |

All cases pass the FP8 correctness gate: cosine similarity
0.999754--0.999764, relative L2 error 2.17--2.22%, and maximum absolute error
0.000563--0.000587.

The earlier 10.1--12.1% Top-K result is a same-call fusion with sparse
page-table/context metadata generation. It does not include cross-layer reuse.
ATOM's `index_topk_freq=4` is a separate, quality-changing optimization and is
currently rejected by the model-quality gate.

The main portable opportunity is therefore the sparse core itself: it is about
2x faster at matched P shapes without changing per-layer Top-K semantics.
