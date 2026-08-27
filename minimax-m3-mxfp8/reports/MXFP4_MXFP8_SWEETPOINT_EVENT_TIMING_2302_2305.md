# MiniMax-M3 MI355X sweetpoint HIP-event timing

## Scope

- Device: MI355X, TP4, production tail-fix image.
- Timing: device-side HIP events inserted at model-stage boundaries; no Kineto
  or PyTorch profiler.
- P-only: C8, 60K--120K ISL, 90% prefix reuse, OSL1, Unified attention,
  shuffled KV layout, and native KV zeroing.
- D-only: DecodeBenchConnector, OSL600, synthetic EAGLE3 acceptance, Unified
  attention, and non-shuffled KV layout.
- MXFP8 includes the exact-bucket dense-GEMM selector and the validated FMoE
  candidate rows used by the current tuned serving stack.
- Distributed values first take the median per rank, then the maximum of the
  four rank medians. Stage boundaries are inclusive and therefore must not be
  summed.

## Decode graph

Values are the summed inclusive stage time in one graph replay. The call count
is 57 layers for routed MoE/shared expert/sparse QKV and 60 for all-reduce and
O-projection.

| Stage | MXFP4 C24 | MXFP8 tuned C22 | MXFP8 delta |
|---|---:|---:|---:|
| Whole graph replay | 38.210 ms | 39.195 ms | +2.6% |
| Routed MoE | 11.783 ms | 14.466 ms | +22.8% |
| Shared expert MLP | 3.136 ms | 3.026 ms | -3.5% |
| Sparse QKV projection | 1.179 ms | 1.697 ms | +43.9% |
| Sparse QK-norm/cache insert | 1.461 ms | 1.519 ms | +4.0% |
| All-reduce + norm | 1.447 ms | 1.293 ms | -10.7% |
| Attention O-projection | 0.828 ms | 1.051 ms | +27.0% |
| Indexer, eager per call | 83.2 us | 81.2 us | -2.4% |

The main MXFP8 decode opportunity is routed MoE. Sparse QKV and O-projection
are secondary exact-shape GEMM opportunities. Indexer and shared expert do not
explain the MXFP8-versus-MXFP4 decode gap.

## Prefill

P runs are eager for the long-prefill stages. Values are the median duration of
one inclusive layer/stage call over the dynamic production workload.

| Stage | MXFP4 C8 | MXFP8 tuned C8 | MXFP8 delta |
|---|---:|---:|---:|
| Dense attention | 7.951 ms | 7.513 ms | -5.5% |
| Dense-attention core | 7.416 ms | 6.972 ms | -6.0% |
| Routed MoE | 3.689 ms | 4.486 ms | +21.6% |
| Dense-layer MLP | 3.045 ms | 3.584 ms | +17.7% |
| All-reduce + norm | 1.967 ms | 1.894 ms | -3.7% |
| Sparse attention total | 1.267 ms | 1.118 ms | -11.8% |
| Indexer | 0.782 ms | 0.693 ms | -11.4% |
| Shared expert MLP | 0.337 ms | 0.634 ms | +87.9% |
| Sparse QKV projection | 0.383 ms | 0.528 ms | +37.9% |
| Dense QKV projection | 0.336 ms | 0.486 ms | +44.9% |
| Attention O-projection | 0.286 ms | 0.399 ms | +39.2% |
| MLP gate/up projection | 0.195 ms | 0.355 ms | +82.1% |
| MLP down projection | 0.106 ms | 0.248 ms | +133.8% |

MXFP8 attention and indexer are already faster than MXFP4. The remaining P
gap is routed MoE plus dense MXFP8 linear kernels, especially shared expert and
the gate/up, down, QKV, and O-projection shapes.

## Validity and overhead

- P event runs: jobs 2304 (MXFP4) and 2305 (MXFP8); all formal requests
  completed, zero failures, prefix hit 90.275%.
- D event runs: jobs 2303 (MXFP4) and 2302 (MXFP8); all formal requests
  completed, zero failures, exact OSL600, acceptance in band.
- Event insertion reduced observed service throughput and raised TPOT. These
  runs are used only for like-for-like stage attribution; uninstrumented runs
  remain the source of TPM and SLA claims.
- Initial P jobs 2300/2301 loaded the native-zero `sitecustomize` before the
  timing `sitecustomize`, so they produced no event files and are excluded from
  stage timing. The combined initialization was fixed before 2304/2305.

## Follow-up

- Job 2308 searched the actual D padded token buckets M=32/64/128 with the
  official AITER FMoE tuner. Production-dispatch eager gains were 2.85%, 5.57%,
  and 1.97%, respectively; all passed the official status/cosine gate.
- Job 2309 then isolated the only material eager candidate, M64, under HIP
  Graph replay. It improved median replay time from 325.923 to 320.883 us
  (+1.57%), with finite output and graph/eager cosine error below 4.4e-6.
  This is below the production-integration threshold, so no serving default or
  model config was changed.
- The next meaningful work is kernel-level MXFP8 routed-MoE and exact-shape
  dense-linear optimization rather than additional CSV tuning. Priority shapes
  are the D sparse QKV/O-projection paths and the P shared-expert, QKV,
  O-projection, gate/up, and down paths.

## Artifacts

- `results/sweetpoint-event/mxfp4-p-r2/sweep-2304/`
- `results/mxfp8-tuned-ab/p-tuned-2305/sweep-2305/`
- `results/sweetpoint-event/mxfp4-d/2303/`
- `results/mxfp8-tuned-ab/d-tuned-2302/2302/`
- `results/mxfp8-fmoe-smallm-tune/2308/`
- `results/mxfp8-fmoe-d-m64-graph/2309/`
- Summarizer: `benchmarks/kernels/minimax_m3/summarize_graph_stage_timing.py`

## Deep sparse-attention kernel events

Jobs 2312 (MXFP4 D C24) and 2313 (MXFP8 D C22) added device events at
the actual Triton JIT launchers used by the production `shuffle=0` path. Both
runs were valid: 144/144 and 132/132 requests completed, respectively, exact
OSL600, zero failures, and 53.56%/53.67% speculative acceptance.

| Kernel event, per sparse layer | MXFP4 C24 | MXFP8 C22 |
|---|---:|---:|
| FP8 index score + partial top-k | 34.920 us | 35.360 us |
| Index top-k merge | 28.141 us | 28.840 us |
| Sparse split-K decode | 43.181 us | 43.281 us |
| Sparse decode merge | 23.340 us | 23.820 us |

The two index kernels account for 55.8%/56.2% of the inclusive
`index_decode_score_topk` device time. The split-K decode and merge kernels
account for 25.2%/25.1% of inclusive `sparse_attn_total`; the remainder is the
other buffer/setup/device work in those parent regions. Scaling the four
one-call medians across 57 sparse layers gives 7.385 ms (17.8%) of the
event-instrumented 41.545 ms MXFP4 graph and 7.485 ms (17.3%) of the 43.371 ms
MXFP8 graph. These whole-graph percentages are attribution estimates: nested
events and event overhead mean they must not replace uninstrumented throughput
or SLA results.

The production sweet point does not call AITER Gluon paged attention: with
`shuffle=0`, it uses vLLM's Triton `_gqa_sparse_decode_kernel` followed by
`_merge_topk_attn_out_kernel`. The index path uses
`_decode_index_score_topk_partial_fp8_kernel` followed by
`_topk_index_merge_kernel`.

Deep-event artifacts:

- `results/sweetpoint-kernel-event/mxfp4-d/2312/`
- `results/sweetpoint-kernel-event/mxfp8-d/2313/`

## Existing-kernel search before optimization

- AITER PR 4373 provides a merged CK VSA block-sparse attention operator, but
  its published validation is FP16/BF16 VSA on gfx942 rather than M3 causal GQA
  with FP8 paged KV. It needs a semantic/shape compatibility gate before any
  performance comparison.
- AITER PR 4511 provides an open gfx950 OPUS MXFP8 paged MQA-logits/indexer
  kernel. It is graph-safe and covers prefill/MTP decode, but the published
  contract is DeepSeek-style with 64-token pages, not M3's 128-token block and
  index-cache contract.
- ATOM's current M3 recipe still forces Triton attention. Its
  `index_topk_freq=4` option is cross-layer algorithmic reuse, not an equivalent
  kernel replacement, and therefore needs separate model-quality validation.

No upstream candidate is a proven drop-in replacement yet. The next bounded
work should be compatibility microbenchmarks—first OPUS indexer if its page-128
contract can be supported, then CK VSA only if causal GQA/FP8 KV semantics can
be matched—before writing a new kernel.

## Existing-library audit and production-shape UT

Job 2314 measured the existing providers at the actual MXFP8 D sweet-point
shape (`M=88`, speculative `q_len=4`, batch 22). Job 2316 repeated sparse PA
with non-zero random Q/K/V, identical FP8 cache contents, selected old and tail
blocks, and the production page-table conversion.

| Candidate | Shape | Median | Correctness | Decision |
|---|---:|---:|---|---|
| Current Triton FP8 index score + partial top-k | 32K / 60K | 91.28 / 141.64 us | BF16 recall@16 95.45% / 95.67%; forced local and graph capture pass | baseline; tune whole path |
| Current Triton sparse decode | 32K / 60K | 51.12 / 51.32 us | non-zero AITER parity cosine 0.99979; mean abs error about 7.2e-5 | retain |
| AITER Gluon sparse PA | 32K / 60K | 139.56 / 138.72 us | same parity and graph gates pass | reject: 2.70--2.73x slower |

The sparse-PA result isolates the production call, including AITER's logical
page-128 to physical page-16 block-table compaction. It therefore rules out the
current Gluon path as a serving integration candidate even though its cache
layout and output are correct.

### Candidate and integration matrix

| Area | Best existing opportunity | Kernel evidence | Integration blocker / next gate |
|---|---|---|---|
| Routed MoE | AITER FlyDSL tuned config | P padded M=16384 graph +10.91%; D M64 graph only +1.57% | config-only; use the exact E128/topk4/M16384 row, not the shipped E129/topk5 rows or the regressing M8192 row |
| Dense MXFP8 linear | current vLLM Triton dot-scaled kernel with exact shape configs | whole quant+GEMM gains: D shared gate 18.4%, sparse QKV 13.8%, O-proj 39.8%; P shared-down 29.3%, O-proj 11.4% | add narrowly tested `_select_cfg` buckets behind opt-in, then run the existing MXFP8 linear correctness suite and E2E |
| Index score + top-k | current fused FP8 score/partial-top-k plus narrow graph-replay configs | four anchor families survive graph replay; broad eager gains were mostly launch-gap noise | restore the opt-in selector that was overwritten by concurrent work; do not use generic AITER `top_k_per_row`, which requires a materialized FP32 score tensor |
| Sparse decode | current Triton split-K + merge | 51 us at M88; AITER Gluon is 2.7x slower | retain Triton; optimize its split-K/merge schedule or use a semantically exact external kernel |
| MAX/Mojo MSA | `msa_amd_decode_dispatch` and AMD sparse indexer on gfx950 | documented page128, head128, FP8 KV, top16 and MTP support; semantically the closest external implementation | no MAX/Mojo runtime on the cluster and no PyTorch/AITER ABI; first obtain a pinned runtime and write a standalone ABI/layout bridge UT |
| AITER OPUS MXFP8 logits PR4511 | schedule-free gfx950 index scorer | upstream reports median decode gain 8.8% vs Triton | open PR; page64, E8M0 scale and DeepSeek score contract differ from M3 page128 block-max/top16 semantics; requires an adapter or upstream M3 variant |
| AITER CK VSA PR4373 | CK thin sparse attention | upstream implementation exists | validated for FP16/BF16 VSA on gfx942, not M3 causal GQA with FP8 paged KV; no performance test until semantic gate exists |
| ATOM M3 | Triton plus `index_topk_freq=4` | public recipe supports M3 | cross-layer index reuse is an algorithmic quality trade-off, not kernel equivalence; needs model/vendor quality validation |

No custom kernel should be started before the MAX ABI feasibility check and an
OPUS page-128/score-contract assessment. The immediate low-risk TPM path is the
validated MoE M16384 row plus exact dense-GEMM and top-k dispatch buckets;
Sparse Decode remains important, but the currently available AITER replacement
is not competitive at the production shape.

Additional artifacts:

- `results/existing-kernel-audit/2314/`
- `results/existing-kernel-audit/2316/`

## Sweet-point Top-K tuning and E2E validation

Jobs 2317, 2318, and 2274--2278 separated production FP8 whole-path timing
from eager launch noise.  The additional D sweet-point bucket is narrowly
defined as speculative `q_len=4`, batch 20--24, at least 384 index blocks.  It
uses 32 exact chunks and leaves score/merge launch parameters at their Triton
defaults.

- Job 2317 found 8--13% event-time gains at batch 22 across 49K, 57K, and 60K
  contexts.
- Job 2318 used separately captured graphs and seven interleaved rounds of 100
  replays.  Batches 20--23 improved by 8.25--14.38% across all three contexts.
- Every point preserved the production FP8 output set exactly, retained BF16
  recall@16 above 95%, forced init/local blocks, non-degenerate score
  distributions, and graph capture.

The serving A/B used the same node and the existing dense-GEMM plus FMoE tuned
stack.  Jobs 2319 and 2323 each ran three complete C22 trials; job 2323 logged
selector hits for actual graph batches 20, 22, and 24.

| MXFP8 D C22, 3-trial median | Current tuned | + Top-K bucket | Delta |
|---|---:|---:|---:|
| Output tok/s | 1,265.18 | 1,316.09 | **+4.02%** |
| Request/s | 2.10863 | 2.19349 | **+4.02%** |
| Total tok/s | 190,746.69 | 198,422.80 | **+4.02%** |
| TPOT P50 | 16.484 ms | 15.696 ms | **-4.78%** |
| TPOT P90 | 17.485 ms | 16.636 ms | **-4.86%** |

All six trials completed 132/132 requests with exact OSL600, zero failures,
and synthetic acceptance between 52.8% and 53.3%.  The candidate is retained
as an opt-in overlay; job 2320 is excluded because its temporary mount source
was placed inside the enroot runtime directory and disappeared before server
startup (zero performance requests).

Artifacts:

- `results/topk-sweetpoint/2317/`
- `results/topk-sweetpoint-validate/2318/`
- `results/mxfp8-tuned-ab/d-tuned-2319/2319/`
- `results/mxfp8-tuned-ab/d-tuned-topk-2323/2323/`

## Sparse decode schedule rejection

Jobs 2321 and 2322 tested split counts 1, 2, 4, and 8 at q_len4 batches
20--24, 32K/60K context, using non-zero random FP8 KV.  All outputs passed the
same AITER parity gate (cosine at least 0.99979), but production's two chunks
were the local optimum.  One chunk regressed 22--25%, four chunks regressed
roughly 1--3%, and eight chunks was noise-level/neutral.  No split-count change
is retained.  Job 2324 additionally swept explicit Triton warp counts; any
provisional result must be rejected unless an interleaved graph replay proves
it differs from Triton's effective default.

- `results/sparse-single-chunk-ab/2321/`
- `results/sparse-single-chunk-ab/2322/`
- `results/sparse-launch-sweep/2324/`

## MXFP4 Top-K component validation

Jobs 2325 and 2326 repeated the same opt-in FP8 index-score/top-k bucket on
MXFP4-weight D-only C24.  Each arm used three trials on the same node and
completed 144/144 requests per trial with exact OSL600 and zero failures.

| MXFP4 D C24, 3-trial median | Current | + Top-K bucket | Delta |
|---|---:|---:|---:|
| Output tok/s | 1,391.66 | 1,487.14 | **+6.86%** |
| Request/s | 2.31943 | 2.47856 | **+6.86%** |
| Total tok/s | 207,966.99 | 222,235.16 | **+6.86%** |
| TPOT P50 | 16.076 ms | 15.061 ms | **-6.32%** |
| TPOT P90 | 17.100 ms | 16.058 ms | **-6.09%** |

This confirms that the index-path optimization is useful with both MXFP4 and
MXFP8 weights because both runs use the same FP8 index-cache path.  TTFT was
noisy across the short decode-only trials and is not used for this decision.

- `results/mxfp4-topk-ab/d-baseline-2325/2325/`
- `results/mxfp4-topk-ab/d-topk-2326/2326/`

## Sparse decode head-tiling rejection

Job 2331 evaluated additional head tiling after the split-count sweep.  The
production path measured 51.40 us, while head tiles of 8 and 4 measured 80.64
and 83.64 us.  Both candidates preserved correctness and graph capture but
were substantially slower, so no production sparse-attention change is
retained.

- `results/sparse-launch-sweep/2331/`

## MXFP8 dense exact-selector final result

Job 2330 searched all 16 production dense shapes at raw M=80, 88, 92, and
96. The independent validation did not turn those anchors into broad ranges:
job 2333 separately captured heuristic and candidate HIP Graphs, alternated
them for 11 rounds of 100 replays, and tested the adjacent M-1/M/M+1 values.
Job 2336 repaired the one false-positive search winner at shared-gate M92.

The final default-off selectors retain only the exact measured M values:

| Dense family (N,K) | Exact retained M | Config | Validated graph gain |
|---|---|---|---:|
| Shared gate/up (1536,6144) | 87,88,89,91,92,93,95,96,97 | `(32,16,1024,2,2)` | +11.42% to +28.64% |
| Sparse QKV (2560,6144) | 79,80,81,87,88,89,91,92,93,95,96,97 | `(32,16,1024,2,2)` | +30.21% to +57.09% |
| O projection (6144,2048) | 79,80,81,87,88,89,91,92,93,95,96,97 | `(64,32,512,2,2)` | +10.60% to +13.66% |
| Shared down (6144,768) | none | heuristic | rejected; search gains below 3% |

The M92 shared-gate search winner with BK512 regressed 19.51% under graph
replay. The replacement BK1024 config improved M91/M92/M93 by
13.72%/13.13%/12.44%. Job 2344 then verified the exact-image overlay: seven
`SELECTOR_HIT` cases, three `DEFAULT_PARITY` misses, and exit `0:0`. The patch
is applied to a job-local source copy only when
`M3_USE_MXFP8_TUNED_GEMM_OVERLAY=1`; production source and defaults remain
unchanged.

- `results/mxfp8-dense-sweetpoint-validate/2333/README.md`
- `results/mxfp8-dense-sweetpoint-validate/2333/exact-selector-recommendation.json`
- `results/mxfp8-dense-overlay-verify/2344/verify.log`

## Routed-MoE raw-M88 bucket128 rejection

The D sweet-point input is raw M=88 (C22 times target plus three draft
positions), while AITER selects the bucket128 config row. These are distinct
from concurrency and from physically padding the benchmark input to M128.
The production heuristic baseline in job 2346 split the full routed-FMoE graph
as follows:

| Raw M / selector bucket | Stage 1 | Stage 2 | Stage share | Full graph |
|---|---:|---:|---:|---:|
| 88 / 128 | 199.921 us | 115.241 us | 63.4% / 36.6% | 327.162 us |

Job 2350 ran the official FlyDSL tuner on the actual raw M=88 input. All
448/448 task groups completed. Its winner was stage1
`flydsl_moe1_afp8_wfp8_bf16_t32x64x256_w3_gui_fp8` at 175.9413 us and stage2
`flydsl_moe2_afp8_wfp8_bf16_t32x128x256_atomic_persist` at 102.7593 us. The
row was remapped to selector bucket128 only after search; it was not a padded
M128 measurement.

Job 2359 cleared both the AITER config cache and metadata LRU before each arm,
proved candidate dispatch, separately captured baseline and candidate HIP
Graphs, and alternated 31 rounds of 10 replays. All points passed finite-output,
candidate/baseline cosine-distance (1.27e-5 to 1.31e-5), and graph/eager parity
(below 3.8e-7) gates.

| Raw M sharing bucket128 | Baseline | Candidate | Median gain | P10 gain |
|---:|---:|---:|---:|---:|
| 80 | 292.910 us | 285.294 us | +2.67% | +2.10% |
| 84 | 296.602 us | 289.390 us | +2.49% | +2.06% |
| 88 | 297.342 us | 291.170 us | +2.12% | +1.75% |
| 92 | 298.246 us | 291.678 us | +2.25% | +1.65% |
| 96 | 304.210 us | 296.578 us | +2.57% | +1.75% |

No adjacent raw-M point reached the required stable 3% threshold, so the
bucket128 candidate is rejected and D must not enable a routed-MoE override.
The already validated P raw/padded M16384 row remains a separate opt-in
candidate (+10.91% graph gain); this rejection does not change it.

- `results/mxfp8-fmoe-m88-bucket128/2350/`
- `results/mxfp8-fmoe-bucket128-validate/2359/`
- `results/mxfp8-moe-backend-eval/README.md`

## Prepared final component A/B stack

`mi355x_mxfp8_final_stack_ab.sbatch` is prepared but was not submitted. Its
default arm is `M3_CANDIDATE=0`, so every new selector/config override remains
off unless the caller explicitly selects candidate arm 1.

| Role | Baseline arm | Candidate arm |
|---|---|---|
| MXFP8 P | Triton attention, production heuristic configs | Unified attention, shuffle=1, native-zero, exact Top-K overlay, exact dense selectors, and only the validated P M16384 routed-FMoE row |
| MXFP8 D | Triton attention, production heuristic configs | Unified attention, exact Top-K overlay, and exact dense selectors; `AITER_CONFIG_FMOE` is explicitly empty |

The exact Top-K and dense guards remain no-ops on unmeasured shapes. A separate
default-off `M3_USE_BF16_EAGLE_CONFIG_OVERLAY` switch accepts one or more
colon-separated, pre-existing AITER BF16 GEMM config files and forwards them as
`AITER_CONFIG_GEMM_BF16`; no BF16 config is selected implicitly. This is runner
plumbing only: no component E2E run was launched for this preparation.

## MXFP4 D C24 final-stack dispatch rejection

Job 2372 ran the prepared same-node sequential A/B. The baseline completed all
three C24 repeats and proved the current Unified-attention path with neither
the exact Top-K overlay nor the BF16 candidate CSV. All 432 requests completed,
all outputs were exactly 600 tokens, failures were zero, and synthetic
acceptance was 52.61--53.21%.

| MXFP4 D C24 | Baseline 3-run median | Candidate diagnostic, 2-run median | Apparent delta |
|---|---:|---:|---:|
| Output tok/s | 1,383.02 | 1,463.22 | +5.80% |
| Request/s | 2.30504 | 2.43870 | +5.80% |
| Total tok/s | 206,676.47 | 218,661.00 | +5.80% |
| TPOT P50 | 16.280 ms | 15.033 ms | -7.66% |
| TPOT P90 | 17.213 ms | 15.939 ms | -7.41% |

The candidate values are **not** a Top-K+BF16 result. Formal traffic confirmed
three Top-K selector-hit messages but zero tuned BF16 hipBLASLt row hits. The
final CSV path was loaded and consulted 1,148 times, yet every lookup used the
default. The two already completed candidate repeats were otherwise valid
(288/288 requests, exact OSL600, zero failures, acceptance 53.00--53.09%); their
gain is a partial Top-K-only diagnostic consistent with jobs 2325/2326, not new
BF16 evidence. Job 2372 was cancelled before the redundant third candidate
repeat and ended `CANCELLED` after 11:13.

The observed BF16 graph/runtime M sets explain the miss:

| BF16 GEMM (N,K) | Observed M values in C24 production capture |
|---|---|
| (50016,6144) | 6,8--10,12--24,32,36,40,48,52,56,60,64,128 |
| (6144,18432) | 8,12,16,24,32,36,40,48,52,56,60,64 |
| (2304,12288) | 6,8--10,12--24,32,104--256 by 8,272--512 by 16,8192 |

The validated CSV contains exact M80/M88/M92/M96 rows for these families, so
its isolated job-2367 correctness/dispatch validation does not establish a C24
serving hit. Do not resubmit this unchanged final-stack runner or claim a BF16
gain. These observed M values are the input to any future exact tuning; no new
MXFP4 tuning was started here.

- `results/mxfp4-d-final-stack-ab/2372/`

## Final BF16 EAGLE kernel table

Jobs 2358, 2362, 2363, and 2365 alternated production and candidate HIP Graph
replays for 11 rounds of 200 replays. The final table retains only stable gains
of at least 3%. Job 2362 was the first aux-FC screen; job 2365 retuned M80/M88
and confirmed that M92 remained below threshold. Aux M96 retains job 2362's
solution 440865. All retained rows use hipBLASLt BF16 and exact M dispatch.

| Family / shape `(M,N,K)` | Solution | Production | Candidate | Graph gain | Source |
|---|---:|---:|---:|---:|---:|
| LM head `(80,50016,6144)` | 439883 | 140.931 us | 114.619 us | +22.96% | 2358 |
| LM head `(88,50016,6144)` | 439747 | 145.246 us | 118.477 us | +22.59% | 2358 |
| LM head `(92,50016,6144)` | 439747 | 145.640 us | 118.646 us | +22.75% | 2358 |
| LM head `(96,50016,6144)` | 439747 | 145.495 us | 115.648 us | +25.81% | 2358 |
| QKV `(80,2304,12288)` | 440119 | 26.322 us | 22.326 us | +17.90% | 2363 |
| QKV `(88,2304,12288)` | 440119 | 27.095 us | 22.625 us | +19.76% | 2363 |
| QKV `(92,2304,12288)` | 440119 | 27.959 us | 22.701 us | +23.16% | 2363 |
| QKV `(96,2304,12288)` | 439921 | 27.852 us | 22.828 us | +22.01% | 2363 |
| Aux FC `(80,6144,18432)` | 440865 | 55.294 us | 51.030 us | +8.36% | 2365 |
| Aux FC `(88,6144,18432)` | 440516 | 53.515 us | 51.936 us | +3.04% | 2365 |
| Aux FC `(96,6144,18432)` | 440865 | 53.391 us | 51.312 us | +4.05% | 2362 |

Aux-FC M92 was rejected at +2.53%. Job 2367 then prepended the 11 candidates
to AITER's base and 13 model config CSVs. Runtime validation observed all 11
expected solution IDs, preserved dispatch parity for 39 exact misses, retained
the official O-projection M64/M128 FlyDSL choices, and passed eager/graph
correctness with maximum relative error `4.8233e-5`. The final candidate CSV
SHA256 is `5253d16af3b4f5dff969a663f00a51b091c7116e88a3daa046813073d4056430`.

- `results/eagle-bf16-interleaved-graph/2358/`
- `results/eagle-bf16-interleaved-graph/2362/`
- `results/eagle-bf16-interleaved-graph/2363/`
- `results/eagle-bf16-interleaved-graph/2365/`
- `results/eagle-bf16-csv-overlay/2367/`

## MXFP8 final component matched A/B

Jobs 2368--2371 ran on four otherwise idle MI355X nodes, one job per node.
Each arm used three formal repeats, the same checkpoint, image, prompt set,
prefix policy, and synthetic EAGLE acceptance. All jobs exited `0:0`.

| Role / arm | Per-round primary throughput | Three-round median | Latency median | Gain |
|---|---|---:|---:|---:|
| P C8 baseline, 2368 | 18463.90 / 18404.62 / 18431.52 fresh tok/s | 18431.52 fresh tok/s | mean TTFT 3255.45 ms | -- |
| P C8 final, 2369 | 25166.31 / 24960.45 / 25074.57 fresh tok/s | 25074.57 fresh tok/s | mean TTFT 2406.99 ms | +36.04%; TTFT -26.06% |
| D C22 baseline, 2370 | 882.03 / 1062.57 / 1047.21 output tok/s | 1047.21 output tok/s | mean TPOT 19.621 ms | -- |
| D C22 final, 2371 | 1245.84 / 1342.25 / 1348.54 output tok/s | 1342.25 output tok/s | mean TPOT 15.094 ms | +28.17%; TPOT -23.07% |

P completed 48/48 requests in every repeat with zero failures, identical mean
fresh tokens of 7966.125, and prefix hit rates of 90.2751--90.2753%. D
completed 132/132 requests in every repeat with zero failures and exact
600-token outputs. Median acceptance was 53.170% for baseline and 53.096% for
final, both inside the 51.3--55.3% gate.

Dispatch was checked fail-closed. Neither baseline worker environment contained
a candidate Top-K, dense, BF16, or FMoE override. P final used Unified
attention, shuffle, and native zero, and all four ranks selected the named
M16384 FMoE candidate. Its C8/M16384 execution did not enter the decode Top-K,
dense M80--96, or BF16 M80--96 sweet branches; those selectors were loaded but
were no-ops for the observed shapes. P baseline used AITER's two-stage default
at M16384, not the named candidate kernels. D final logged three exact Top-K
hits at batch 20/22/24, ten dense sweet-selector hits, and 64 exact BF16 sweet
hits; D FMoE remained unset. The corresponding D baseline hit counts were all
zero.

- `results/mxfp8-final-stack-ab/p-baseline-2368/`
- `results/mxfp8-final-stack-ab/p-final-stack-2369/`
- `results/mxfp8-final-stack-ab/d-baseline-2370/`
- `results/mxfp8-final-stack-ab/d-final-stack-2371/`

## Prepared final MXFP8 1P1D integration

`mi355x_mxfp8_final_stack_openloop_1p1d.sbatch` prepares one matched arm at the
production 1.20 offered QPS, with 23 requests per prefix slot, P-side INT4
QuickReduce, and D-side QuickReduce disabled. `M3_CANDIDATE=0` is the default
and keeps Triton attention plus every new selector/config override off.
`M3_CANDIDATE=1` enables the validated P and D final stacks, with P-only native
zero and M16384 FMoE, D FMoE explicitly empty, and the BF16 candidate prepended
to the complete official config list. Result suffixes contain the arm and QPS.
The runner was subsequently submitted; the qualified result is documented
below.

## Final MXFP8 1P1D open-loop integration

Jobs 2375 and 2391 are the matched production 1.20-QPS arms. Both used the
same 736-request workload, seed 20260819, input-length seed 42, 60K--120K
uniform inputs, 90% prefixes, and 600 output tokens. Job 2375 is the untouched
baseline. Job 2391 is the fail-closed fallback final stack: P/D Unified
attention, exact Top-K and dense selectors, BF16 overlay, and the P M16384
FMoE config loaded, but P/D shuffle, native zero, and the shuffled MoRI
registration overlay all disabled. Both jobs completed `0:0`.

| Metric | Baseline 2375 | Final fallback 2391 | Delta |
|---|---:|---:|---:|
| Completed / failed | 736 / 0 | 736 / 0 | parity |
| Duration | 623.116 s | 621.088 s | -0.33% |
| Request throughput | 1.18116 QPS | 1.18502 QPS | +0.33% |
| Total token throughput | 102355.04 tok/s | 102689.26 tok/s | +0.33% |
| Total TPM | 6.1413M | 6.1614M | +0.33% |
| Output throughput / TPM | 708.70 tok/s / 42.522K | 711.01 tok/s / 42.661K | +0.33% |
| TTFT p50 / p99 | 1074.25 / 3503.21 ms | 837.56 / 2281.54 ms | -22.03% / -34.87% |
| TPOT p50 / p99 | 16.476 / 19.009 ms | 12.672 / 14.892 ms | -23.09% / -21.66% |
| Prefix hit / EAGLE acceptance | 89.8306% / 53.1772% | 89.8306% / 52.9903% | gate parity |
| Failed transfers | 0 | 0 | parity |

Thus the demonstrated sustainable rate at this fixed production point is
6.161M total TPM, or 42.661K output TPM. This is not a saturation sweep: the
offered rate was fixed at 1.20 QPS, so the stronger result is the 22--35%
latency reduction while preserving completion, prefix, acceptance, and
transfer correctness; the 0.33% throughput change mostly reflects the shorter
drain.

Per-minute windows below bucket requests by identical scheduled start times.
The last row is the partial 13-request tail. Columns are requests, TTFT
p50/p99 in ms, and TPOT p50/p99 in ms.

| Minute | Requests | Baseline TTFT | Final TTFT | Baseline TPOT | Final TPOT |
|---:|---:|---:|---:|---:|---:|
| 1 | 72 | 1053 / 2844 | 835 / 1669 | 16.320 / 17.986 | 12.658 / 14.292 |
| 2 | 76 | 1080 / 3361 | 818 / 2270 | 16.641 / 18.910 | 12.832 / 14.912 |
| 3 | 73 | 1082 / 2768 | 842 / 1961 | 16.589 / 18.163 | 12.687 / 14.326 |
| 4 | 65 | 1059 / 2446 | 847 / 1437 | 16.210 / 17.782 | 12.061 / 14.497 |
| 5 | 84 | 1165 / 2145 | 843 / 1577 | 17.107 / 18.791 | 13.377 / 15.071 |
| 6 | 72 | 1011 / 4137 | 801 / 2709 | 16.341 / 17.984 | 12.520 / 14.291 |
| 7 | 61 | 1038 / 1870 | 812 / 1529 | 15.744 / 17.461 | 12.130 / 13.630 |
| 8 | 71 | 1080 / 2320 | 845 / 1489 | 16.633 / 18.955 | 12.631 / 14.271 |
| 9 | 64 | 1092 / 2619 | 836 / 1875 | 15.859 / 17.197 | 12.333 / 13.731 |
| 10 | 85 | 1194 / 3909 | 858 / 2584 | 17.171 / 20.102 | 13.303 / 15.139 |
| 11 tail | 13 | 959 / 1560 | 751 / 1092 | 15.188 / 16.166 | 11.496 / 12.050 |

Dispatch and correctness were fail-closed. Job 2391 logged Unified attention
on both roles, exact Top-K hits for batch 20/22/24, tuned dense MXFP8 hits,
and an exact BF16 LM-head `(32,50016,6144)` hit. The configured P M16384 FMoE
row did not execute in this integrated workload: the observed routed-FMoE
startup shape was M128 and used AITER's heuristic fallback, so no integrated
FMoE gain is claimed. Native/shuffle registration hits were zero as required.
There were no tracebacks, runtime/worker errors, transfer failures, recompute,
NaNs, or index fallback. The single q/prob-scale warning per role was identical
in baseline and candidate and is not introduced by the final stack.

The excluded shuffled-path sequence is preserved as negative evidence. Jobs
2376/2381/2382 failed before formal traffic due respectively to non-contiguous
MoRI registration or incomplete native-op loading. Job 2389 then passed a
four-rank preflight using the exact-image connector: connector import,
`same_data_ptr`, real `IOEngine.register_torch_tensor`, and native zero all
passed. The only exact-image retry, job 2390, proved compute-view identity and
full storage-span coverage but failed the Paris chat-quality gate with corrupt
output. Therefore a contiguous alias of the shuffled storage is memory-safe
but not transfer-offset compatible with MoRI; it must not be published.

- `results/pd-steady/1p1d-packed-local-pint4-dnone-c32-60000-120000i-r0.9-600o-2375-mxfp8-final-integration-baseline-qps1.20/`
- `results/pd-steady/1p1d-packed-local-pint4-dnone-c32-60000-120000i-r0.9-600o-2391-mxfp8-final-integration-final-stack-no-shuffle-qps1.20/`

## MXFP8 production-QPS sweep and SLA semantic recheck

Jobs 2393--2401 swept the qualified job-2391 fallback stack without changing
any kernel, connector, seed, or workload parameter. Shuffle, native-zero, and
the MoRI registration overlay remained permanently disabled. Each point ran
10 minutes open loop plus drain on one exclusive eight-GPU MI355X node. The
production SLA is an alert-style condition: the p50 metric must remain above
`TTFT = 3 s` or `TPOT = 16.6 ms` continuously for 10 minutes. A global p50 over
all samples in the run is not equivalent to that condition.

| Job | Offered QPS | Completed / failed | Achieved QPS | Global TTFT p50 | Global TPOT p50 | 9x1P1D offered-load TPM equivalent | 9x1P1D achieved TPM equivalent | 10-minute alert |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 2393 | 1.50 | 928 / 0 | 1.4812 | 0.859 s | 13.410 ms | 70.192M | 69.313M | not triggered |
| 2394 | 1.70 | 1024 / 0 | 1.6775 | 0.913 s | 13.852 ms | 79.551M | 78.497M | not triggered |
| 2395 | 1.90 | 1152 / 0 | 1.8727 | 0.996 s | 14.404 ms | 88.910M | 87.630M | not triggered |
| 2396 | 2.10 | 1280 / 0 | 2.0732 | 1.152 s | 15.257 ms | 98.269M | 97.016M | not triggered |
| 2397 | 2.20 | 1344 / 0 | 2.1715 | 1.233 s | 15.823 ms | 102.948M | 101.615M | not triggered |
| 2398 | 2.30 | 1408 / 0 | 2.2673 | 1.306 s | 16.466 ms | 107.628M | 106.098M | not triggered |
| 2401 | 2.35 | 1440 / 0 | 2.3158 | 1.370 s | 16.934 ms | 109.967M | 108.366M | not triggered |
| 2399 | 2.40 | 1440 / 0 | 2.3652 | 1.418 s | 17.380 ms | 112.307M | 110.680M | not triggered |
| **2400** | **2.50** | **1504 / 0** | **2.4590** | **1.589 s** | **17.726 ms** | **116.987M** | **115.068M** | **not triggered; highest tested** |

The earlier conclusion that 2.35--2.50 QPS failed, based only on global p50,
is withdrawn. None of the four boundary jobs had 10 consecutive complete
one-minute windows above either threshold. Therefore this sweep did **not**
close the upper boundary and does not establish a final 1P1D sweet point. Job
2400 only proves that one P-to-one D link sustained 2.50 offered QPS without
triggering the continuous-10-minute alert. The last two TPM columns are useful
load equivalents, but multiplying a linked 1P1D pair by nine is **not** the
correct 72-GPU production-capacity calculation because optimal production may
use an unequal integer P/D allocation.

Jobs 2403--2410 subsequently closed the deployable final-stack component
boundaries. All runs forced shuffle, native-zero, and MoRI registration off. P
used Unified attention and INT4 QuickReduce; D used Unified attention, exact
Top-K/dense selectors, the BF16 overlay, and an empty D FMoE config. Quick
points used two client warmups and `6*C` formal requests. Boundary and peak
points then used one discardable round plus three stable rounds on the same
node; the table reports the median of rounds 2--4.

| Role / point | Job | Stable request rate | Primary throughput | Latency p50 | SLA |
|---|---:|---:|---:|---:|:---:|
| P C8 peak | 2410 | **3.630000 req/s** | 28,917.04 fresh tok/s | TTFT 2,221.12 ms | pass |
| P C10 boundary pass | 2409 | 3.586762 req/s | 29,135.62 fresh tok/s | TTFT 2,766.67 ms | pass |
| P C11 boundary fail | 2409 | 3.462823 req/s | 28,854.45 fresh tok/s | TTFT 3,328.55 ms | fail |
| D C23 | 2407 | 2.230340 req/s | 1,338.20 output tok/s | TPOT 16.127 ms | pass |
| D C24 peak / boundary pass | 2407 | **2.321956 req/s** | **1,393.17 output tok/s** | TPOT 16.016 ms | pass |
| D C26 boundary fail | 2407 | 2.296138 req/s | 1,377.68 output tok/s | TPOT 17.809 ms | fail |

The first quick C23 wave in job 2403 produced only 1,059.71 output tok/s. It
was not used: a same-node rerun showed the expected first-wave/cold effect, and
C23 also misses the even exact Top-K/dense buckets that C24 hits. Jobs 2407,
2409, and 2410 completed with zero request failures; all retained prefix and
acceptance gates passed.

The final 72-GPU capacity uses the SLA-valid request-rate peaks, not replicated
1P1D load. With 18 TP4 engine slots, exhaustive integer enumeration of
`NP + ND = 18`, `qP = 3.630000`, and `qD = 2.321956` selects **7P + 11D**:
`min(7 * qP, 11 * qD) = 25.410002 req/s`. At 86,656.702 total tokens/request,
the deployable calculated capacity is therefore
`25.410002 * 86,656.702 * 60 = 132.117M TPM`. This supersedes the provisional
8P+10D / 116.315M estimate above. It is numerically +135.28% over the old
document's 56.153M figure, but the old number used replicated 1P1D rather than
integer component packing, so this must not be labeled a strictly matched
kernel-only gain.

Dispatch was fail-closed. The P M16384 FMoE row was configured but did not
execute: observed fused-MoE shapes used AITER heuristic fallback, so no FMoE
gain is claimed. P QuickReduce and Unified attention were active. D logged the
expected exact Top-K, dense, and BF16 candidate hits. No shuffled, native-zero,
or MoRI-registration candidate was active on either role.

The fixed one-minute recheck is below. Each cell is `TTFT p50 seconds / TPOT
p50 milliseconds`; a TPOT value above 16.6 is marked with `*`.

| Minute | 2398 / 2.30 | 2401 / 2.35 | 2399 / 2.40 | 2400 / 2.50 |
|---:|---:|---:|---:|---:|
| 1 | 1.538 / 17.261* | 1.648 / 17.596* | 1.715 / 17.816* | 1.875 / 18.023* |
| 2 | 1.208 / 17.512* | 1.299 / 17.510* | 1.330 / 17.627* | 1.633 / 17.836* |
| 3 | 1.780 / 17.422* | 1.985 / 17.950* | 2.069 / 18.211* | 2.288 / 18.153* |
| 4 | 1.371 / 16.312 | 1.433 / 16.743* | 1.483 / 16.451 | 1.712 / 16.620* |
| 5 | 1.348 / 15.072 | 1.548 / 15.899 | 1.652 / 17.554* | 2.304 / 18.332* |
| 6 | 1.367 / 17.707* | 1.347 / 17.569* | 1.369 / 17.953* | 1.309 / 18.404* |
| 7 | 1.154 / 16.392 | 1.210 / 17.612* | 1.283 / 17.832* | 1.782 / 18.022* |
| 8 | 1.266 / 16.295 | 1.260 / 16.088 | 1.320 / 16.870* | 1.171 / 16.405 |
| 9 | 1.219 / 15.638 | 1.139 / 15.708 | 1.090 / 15.131 | 1.141 / 15.518 |
| 10 | 1.000 / 14.842 | 1.087 / 15.342 | 1.274 / 16.192 | 1.281 / 17.158* |

No column contains 10 consecutive starred minutes, and every listed TTFT p50
is below 3 seconds.

At 2.30 QPS, the ten complete-minute TPOT p50 values were
17.261, 17.512, 17.422, 16.312, 15.072, 17.707, 16.392, 16.295, 15.638, and
14.842 ms. The corresponding TTFT p50 values were 1.538, 1.208, 1.780, 1.371,
1.348, 1.367, 1.154, 1.266, 1.219, and 1.000 s. The 22-request drain tail was
14.774 ms TPOT p50 and 1.358 s TTFT p50.

The 2.35-QPS run's ten complete-minute TPOT p50 values
were 17.596, 17.510, 17.950, 16.743, 15.899, 17.569, 17.612, 16.088, 15.708,
and 15.342 ms; TTFT p50 ranged from 1.087 to 1.985 s. Its full-run TPOT p50 of
16.934 ms does not trigger the alert because minutes 5 and 8--10 reset the
continuous-over-threshold timer.

At 2.40 QPS, minutes 4, 9, and 10 were below the TPOT threshold (16.451,
15.131, and 16.192 ms). At 2.50 QPS, minutes 8 and 9 were below it (16.405 and
15.518 ms). Every complete-minute TTFT p50 in all four boundary jobs was below
3 seconds. As a sensitivity analysis, a 60-second sliding p50 sampled every
second from request scheduled-start buckets also found no 10-minute breach:
the longest TPOT-over-threshold streaks were 142, 179, 189, and 192 seconds at
2.30, 2.35, 2.40, and 2.50 QPS respectively; TTFT had no over-threshold streak.
This rolling reconstruction is indicative, not identical to the production
monitor: raw data lacks metric scrape timestamps, completion-time buckets,
histogram aggregation rules, and alert evaluation cadence. A run only slightly
longer than 10 minutes also provides little margin for proving a 10-minute
continuous alert, although the observed below-threshold resets are sufficient
to disprove such a breach in these runs under the fixed-minute reconstruction.

Every point completed with zero request failures and zero failed transfers.
Fatal and forbidden-marker counts were zero. Dispatch remained the qualified
job-2391 fallback dispatch: Unified attention on P/D, exact Top-K and dense
selectors, the BF16 overlay, no shuffled/native/MoRI registration hits, and no
claim for the loaded-but-unexecuted P M16384 FMoE row. Prefix hit rate remained
89.815--89.824% and EAGLE acceptance remained 53.046--53.155% across the sweep.
The Slurm queue was empty after job 2401, so all four benchmark nodes were
released.

### MXFP8 static-scale sparse-cache fusion (kernel-only)

HIP-event attribution at C24/C26 isolated the unfused sparse QK-norm/RoPE plus
main/index cache insertion at about 34.3 us/layer. A three-launch Triton
candidate preserves the calibrated scalar K/V scales, FP8 main/index cache
bytes, page-128 layout, TP4-local 16Q/1KV/1-index heads, forced boundary
offsets 0/127/128/255/256/383, and BF16 Q/index-Q outputs. Job 2507 validated
the production packed cache contract under captured, interleaved graph replay:

| raw M | candidate median | vs 34.3 us stage | stage reduction | cache bytes | graph replay |
|---:|---:|---:|---:|:---:|:---:|
| 96 | 13.300 us | 2.579x | 61.23% | exact | stable |
| 104 | 13.148 us | 2.609x | 61.67% | exact | stable |

Q/index-Q reference error was zero at M96; at M104 Q relative L2 was
1.70e-5 and index-Q was zero. The FP8 K/V/index-K dequantized relative L2 was
2.64--2.70%, matching the production FP8 byte representation exactly. The
candidate remains default-off and exact-selected only for M96/M104; M95/97 and
M103/105, non-FP8 caches, non-scalar scales, different page/head geometry, or
shuffled storage all retain the production path. Source application is
fail-closed against exact production SHA `7830e9e5...bd7ee6`. This is a
kernel-only result; no component or SLA gain is claimed here.

The same-node component A/B then quantified the serving effect (jobs 2508 and
2511 on node03, discard r1 and median of stable r2--r4, 6C requests, OSL600):

| arm | output tok/s median | request/s median | TPOT p50 median | SLA |
|---|---:|---:|---:|:---:|
| C24 final-stack control | 1,411.999 | 2.35333 | 15.922 ms | pass |
| C24 static fused M96 | **1,469.878** | **2.44980** | **15.464 ms** | pass |
| C26 static fused M104 | 1,455.526 | 2.42588 | 16.774 ms | fail |

At C24 the fusion raises output/request throughput by **4.10%** and lowers
TPOT p50 by **2.88%**. C26 remains just beyond the strict 16.6 ms SLA and is
also 0.98% below the C24 fused throughput, so no higher-C open-loop boundary
run was started. Every retained round completed all requests with exact OSL600,
zero failures, and 52.63--53.49% synthetic acceptance. The control log had no
static selector hit; candidate startup capture logged both exact M96 and M104
graphs, and formal C24/C26 map to q_len4 raw M96/M104 respectively.

### P batching-token limit at 3.6 QPS

Jobs 2513/2514 held the deploy-safe P stack and 10-minute QPS3.6 workload
constant while changing only `max_num_batched_tokens`. Both used C8 prefix
sessions, max-seqs/max-concurrency 128, the same seed, no shuffle/native/MoRI,
and 89.776% prefix-cache hit rate.

| MBT | job | completed | achieved QPS | fresh tok/s | TTFT p50 / p90 | result |
|---:|---:|---:|---:|---:|---:|:---:|
| 16,384 | 2494 | baseline | -- | -- | 5.527 s / -- | fail |
| 24,576 | 2513 | 2160/2160 | 3.55696 | 28,335 | 4.080 / 6.637 s | fail SLA |
| 32,768 | 2514 | 2159/2160 | 3.54391 | 28,231 | 5.638 / 8.923 s | invalid |

MBT24K reduced TTFT p50 by 26.2% versus the supplied MBT16K baseline, but
remained above the 3-second SLA. MBT32K regressed and had one failed request.
Therefore MBT24K is the best tested batching setting at QPS3.6, but MBT alone
does not make that offered rate sustainable. The improvement motivated the
boundary refinement below; no point above QPS3.6 was started.

A follow-up MBT24K boundary refinement kept the same 10-minute workload and
all other settings fixed:

| offered QPS | job | completed | achieved QPS | fresh tok/s | TTFT p50 / p90 | qualification |
|---:|---:|---:|---:|---:|---:|:---:|
| 3.50 | 2518 | 2103/2104 | 3.47164 | 27,656 | 2.756 / 4.816 s | fail: 1 request |
| 3.55 | 2519 | 2136/2136 | 3.52466 | 28,078 | 3.055 / 5.189 s | fail: TTFT SLA |

The lower point met the latency threshold but failed the zero-request-failure
gate; the upper point completed cleanly but missed the 3-second TTFT p50 SLA by
55 ms. Consequently neither is fully qualified, the previous QPS3.45 baseline
remains the highest pass, and the conditional QPS3.575 run was not launched.

- `results/pd-steady/1p1d-packed-local-pint4-dnone-c32-60000-120000i-r0.9-600o-2393-mxfp8-final-integration-final-stack-no-shuffle-qps1.50/`
- `results/pd-steady/1p1d-packed-local-pint4-dnone-c32-60000-120000i-r0.9-600o-2394-mxfp8-final-integration-final-stack-no-shuffle-qps1.70/`
- `results/pd-steady/1p1d-packed-local-pint4-dnone-c32-60000-120000i-r0.9-600o-2395-mxfp8-final-integration-final-stack-no-shuffle-qps1.90/`
- `results/pd-steady/1p1d-packed-local-pint4-dnone-c32-60000-120000i-r0.9-600o-2396-mxfp8-final-integration-final-stack-no-shuffle-qps2.10/`
- `results/pd-steady/1p1d-packed-local-pint4-dnone-c32-60000-120000i-r0.9-600o-2397-mxfp8-final-integration-final-stack-no-shuffle-qps2.20/`
- `results/pd-steady/1p1d-packed-local-pint4-dnone-c32-60000-120000i-r0.9-600o-2398-mxfp8-final-integration-final-stack-no-shuffle-qps2.30/`
- `results/pd-steady/1p1d-packed-local-pint4-dnone-c32-60000-120000i-r0.9-600o-2401-mxfp8-final-integration-final-stack-no-shuffle-qps2.35/`
- `results/pd-steady/1p1d-packed-local-pint4-dnone-c32-60000-120000i-r0.9-600o-2399-mxfp8-final-integration-final-stack-no-shuffle-qps2.40/`
- `results/pd-steady/1p1d-packed-local-pint4-dnone-c32-60000-120000i-r0.9-600o-2400-mxfp8-final-integration-final-stack-no-shuffle-qps2.50/`

### Deploy-safe component open-loop capacity boundary

Jobs 2494--2520 replaced the closed-loop concurrency-derived capacity with
finite-rate, component-only open-loop measurements. Each retained point used
the production 60K--120K workload and seed, two warmups, at least 600 seconds
of arrivals plus drain, and max-concurrency 128 only as a safety ceiling. P
used eight prefix sessions and OSL1; D used DecodeBench and OSL600. The primary
gate here is the full-run p50 requested for this capacity sweep: P TTFT below
3 seconds and D TPOT below 16.6 ms. Fixed start-time minute windows remain in
each `*-minutes.json` artifact.

P and D were each validated independently with their own 10-minute open-loop
component workload. Earlier 1P1D sustained runs remain transport/link pressure
evidence only; their achieved rate is not used to infer cluster capacity.

| Role / stack | Offered QPS | Job | Arrival valid | Completed / failed | p50 | Result |
|---|---:|---:|:---:|---:|---:|:---:|
| P final | 3.30 | 2505 | yes | 1,984 / 0 | TTFT 1.853 s | pass |
| P final | **3.45** | **2510** | yes | 2,072 / 0 | **TTFT 2.492 s** | **pass** |
| P final | 3.60 | 2494 | yes | 2,160 / 0 | TTFT 5.527 s | fail |
| P final | 3.90 | 2496 | no, ceiling-throttled | 2,344 / 0 | TTFT 30.129 s | invalid / fail |
| D final, fusion off | **2.10** | **2509** | yes | 1,280 / 0 | **TPOT 15.958 ms** | **pass** |
| D final, fusion off | 2.20 | 2512 | yes | 1,408 / 0 | TPOT 16.681 ms | fail |
| D final, fusion off | 2.30 | 2504 | yes | 1,408 / 0 | TPOT 17.540 ms | fail |
| D final, fusion off | 2.50 | 2497 | yes | 1,536 / 0 | TPOT 19.074 ms | fail |
| D final + static fusion | **2.20** | **2517** | yes | 1,408 / 0 | **TPOT 16.439 ms** | **pass** |
| D final + static fusion | 2.25 | 2520 | yes | 1,408 / 0 | TPOT 16.735 ms | fail |
| D final + static fusion | 2.30 | 2515 | yes | 1,408 / 0 | TPOT 17.202 ms | fail |
| D final + static fusion | 2.50 | 2516 | yes | 1,536 / 0 | TPOT 18.872 ms | fail |

The retained P=3.45 run was unthrottled (arrival stretch 0.999629), drained in
3.87 seconds, had max observed concurrency 27, 89.777% prefix-cache hit, zero
failed transfers, and no fatal marker. Its ten minute TTFT p50 values were
2.534, 3.742, 2.151, 4.098, 4.001, 2.065, 1.294, 1.023, 2.442, and 2.427
seconds; seven of ten conservative minute windows passed even though the
specified full-run p50 gate passed.

Static fusion moved the matched 2.20-QPS D point from 16.681 ms in job 2512
to 16.439 ms in job 2517, a 1.45% TPOT reduction that changes the global-p50
result from fail to pass. Job 2517 was unthrottled (arrival stretch 0.999549),
drained in 8.37 seconds, had max observed concurrency 39, 53.069% synthetic
acceptance, and zero failures/fatal markers. Its ten minute TPOT p50 values
were 17.159, 18.141, 14.925, 15.646, 16.252, 17.232, 17.003, 18.356,
15.108, and 16.086 ms. The 2.25-QPS fail closes the fusion boundary to
`[2.20 pass, 2.25 fail]`.

Every fusion job applied the exact-source-SHA overlay and logged one startup
selector hit for each of M96 and M104 with page 128 and TP4-local heads 16/1.
All four completed with zero request failures and no fatal marker. Shuffle,
native-zero, and MoRI remained disabled. Job 2495 is not in the table because
it overlapped job 2493 on node03; it remains diagnostic only and job 2504 is
the uncontended replacement.

Using sustainable offered rates `qP=3.45` and `qD=2.20`, exhaustive integer
enumeration of 18 TP4 engine slots selects **7P + 11D**:
`min(7 * 3.45, 11 * 2.20) = 24.15 req/s`. At 86,656.702 total
tokens/request, deployable candidate capacity is therefore
`24.15 * 86,656.702 * 60 = 125.566M TPM`. The same stack with the static
fusion left at its default-off setting is bounded by `qD=2.10`, or
120.106M TPM, so enabling the exact-selector fusion adds 4.55% calculated
cluster capacity. It is also +6.68% versus the user's recalled 117.7M result.
The 125.566M figure is +123.61% versus the old document's 56.153M number, but
that old number used replicated 1P1D rather than this integer component
packing and is not a strictly matched kernel-only baseline.
This open-loop result supersedes the earlier 132.117M closed-loop estimate;
the latter overstated SLA-valid offered capacity by 4.96%.

The static-fusion switch remains default-off. The open-loop wrapper is
`mi355x_mxfp8_final_stack_component_openloop.sbatch`; its P branch forces the
switch off and its D branch only enables it through
`M3_USE_STATIC_FP8_QKNORM_CACHE_OVERLAY=1`. Raw artifacts are under
`results/mxfp8-final-component-openloop/`. Jobs 2515--2520 released their
nodes after completion.
