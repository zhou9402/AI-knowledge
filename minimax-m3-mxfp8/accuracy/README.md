# MiniMax-M3 MXFP8 accuracy validation on MI355X (vs NVIDIA's published numbers)

Reproducible accuracy evaluation of **MiniMax-M3 (MXFP8)** on 4x MI355X (gfx950,
TP=4) with **upstream vLLM** (`vllm-minimax-m3-ac750-k3base-a3a9c408e1` image +
`vllm-minimax-kernel-next@417debf` boot-required overlay only), benchmarked
against the [nvidia/MiniMax-M3-NVFP4](https://huggingface.co/nvidia/MiniMax-M3-NVFP4)
card's MXFP8 baseline (which follows Artificial Analysis methodology).

## Results

| Benchmark | NV card (MXFP8, B200) | Ours (MI355X) | Verdict |
|---|---:|---:|---|
| GSM8K | 95.30 (full-precision baseline) | 94.77 / 94.84 (flex/strict) | ✅ within noise |
| GPQA-Diamond | 92.53 | **92.12 ± 1.74** (5 repeats) | ✅ parity |
| AA-LCR | 76.62 | **74.00** (3 repeats, Qwen3-235B judge) | ✅ mechanism-aligned (residual = judge model: AA currently uses GPT-5.6 Luna) |
| SciCode | 49.90 | **48.9** (3 repeats: 49.1 / 51.5 / 46.0) | ✅ within noise |
| τ²-Telecom | 92.22 | **92.98** (114/114 tasks, 0 infra errors) | ✅ parity/exceed |
| MMMU-Pro | 71.97 | — | ❌ excluded: multimodal, not comparable on text-only serving |

**INT4 quick all-reduce (`AITER_QUICK_REDUCE_QUANTIZATION=INT4`) accuracy check:**
no measurable impact (ATOM: 94.84/94.92 vs 95.22/95.30 baseline, within noise;
the env is a no-op on vLLM, whose TP groups pick CUSTOM/PYNCCL, not QUICK_REDUCE).

## How to reproduce

Scripts live in `scripts/`; SLURM wrappers in `scripts/slurm/`. Copy
`scripts/slurm/site.env.example` to `scripts/site.env` and set paths. All jobs
run through the enroot harness with the pure image; see each sbatch header.

| Benchmark | Runner | sbatch | Harness |
|---|---|---|---|
| GSM8K | `run_accuracy_clean.sh` (vLLM) / `run_accuracy_atom.sh` (ATOM) | `accuracy_clean.sbatch` / `eval_accuracy_atom.sbatch` | lm_eval 5-shot, `apply_chat_template`, `fewshot_as_multiturn` |
| GPQA-Diamond | `run_gpqa_simple_vllm.sh` + `gpqa_simple.py` | `eval_nv_vllm.sbatch` (set `GPQA_SEED/GPQA_PORT`, 5 runs) | AA-style prompt ("The last line ... 'Answer: A/B/C/D'"), temp=1.0, top_p=0.95, max_tokens=128000 |
| AA-LCR | `run_eval_aalcr_variant.sh` + `aa_lcr.py` | `eval_aalcr_variant.sbatch` | official prompt template + official equality-checker prompt; judge = Qwen3-235B-A22B-Instruct-2507-FP8 (self-hosted) |
| SciCode | `run_eval_scicode_vllm.sh` | `eval_scicode_vllm.sbatch` (1 per node = 1 repeat) | official repo inspect_ai integration, `with_background=True`, single epoch per job |
| τ²-Telecom | `run_tau2_telecom.sh` | `eval_tau2_telecom.sbatch` | sierra tau2-bench, agent=M3, user-sim=Qwen3-235B (tool-calling enabled on both) |

## Pitfalls found (each cost real points)

1. **Index cache kills long context.** `use_index_cache=true,
   index_topk_freq=4` (ATOM-recipe default) drops AA-LCR from 72.0 to 37.0.
   Harmless at short context (GSM8K/GPQA unchanged). Do not enable for
   long-context serving.
2. **Reasoning models need `--reasoning-parser minimax_m3`.** Without it,
   `<mm:think>` leaks into `content` and any harness that takes the *first*
   code/answer block scores drafts: SciCode 32.0 → 48.9.
3. **max output tokens must cover reasoning spirals.** GPQA wrong answers are
   40k+ token deliberations; a 32k cap truncates them (~2 pts on GPQA). AA
   gives reasoning models their max disclosed output budget.
4. **lm_eval's `gpqa_diamond_cot_zeroshot` prompt lacks an end-of-answer
   marker** and its strict regex matches ~19% of responses; the AA prompt
   fixes extraction. `gpqa_local/` is the stock task repointed at the official
   GitHub `dataset.zip` (HF copy is gated; password is in the upstream README).
5. **Verify the answering server is yours.** On shared nodes, stale servers
   and port collisions produced wrong-config measurements twice; all runners
   now check port-free-before-serve and `/v1/models` identity after.
6. Judge-parsing gotcha: `"CORRECT" in "INCORRECT"` is `True` — match the
   first token, not substring.
7. SciCode's inspect integration keys code/pass-logs by step_id: multi-epoch
   runs collapse (epoch 2/3 are unscored). Use separate jobs per repeat.
