# MiniMax-M3 MXFP8 P-only（Prefill）优化记录

日期：2026-08-26 至 2026-08-27。
代码仓库：`github.com/zhou9402/vllm-minimax-kernel-next`（main + opt/* 分支，每个分支含独立 RESULTS.md 与复现 job 号）。

## 目标与口径

- 硬件/运行：MI355X ×4（TP4），vLLM 定制镜像 `vllm-minimax-m3-ac750-k3base-a3a9c408e1-mori-tailfix`（vLLM v0.27.2rc1.dev77+gac7509e2b）+ 运行时 overlay
- 负载：P-only 生产 proxy，随机 60K–120K ISL，prefix 命中 ~90%，OSL=1，C8，EAGLE3 draft=3（synthetic acceptance）
- SLA：TTFT p50 < 3 s；吞吐用 open-loop 供给速率口径（fresh prefill tok/s）
- 精度门：GSM8K 5-shot（参考 0.9484）；profiler 数字不用于性能结论（观察者效应有前科）

## 总览

- 起点：SLA 甜点 3.45 QPS / 26.8K tok/s（ATOM 39.5K 的 67.8%）
- 终点：**SLA 甜点 5.0 QPS / 39.7K tok/s = ATOM 的 100.6%（追平）**，同点 TTFT 更低（1,745ms vs 2,274ms）
- 注意：39.7K 的 parity 组合里包含两条**后因精度被停用**的优化（PTPC、indexer 跨层复用）；停用后的有效组合数字见下表"全部已验证项叠加"行

## 优化项总表

| # | 优化点 | 内容 | 实测收益 | 精度 | 状态 |
|---|---|---|---|---|---|
| 1 | indexer 跨层共享（index_topk_freq=4） | top-k 每 4 层共享一次（ATOM 生产自带配置），顺手修了一个 C++ fused op 丢 fp8 KV scale 的真 bug | indexer score+topk **-73.7%**，E2E **+8.6%** | 语义改动 | **已停用**（精度决定） |
| 2 | 稀疏 attention 换 gluon paged-decode kernel | prefill 拆成 per-token decode（ATOM 同款做法），page-128→16 零拷贝；镜像里躺着的原型被启用 | kernel 1.33-1.67x/层，E2E **+8.9%** | bf16 compute，relerr 0.08% ✓ | 启用 |
| 3 | EAGLE3 draft prefill 换 CK FMHA varlen | draft 的 TRITON_ATTN prefill 形状批次（qlen>8）走 fp8 KV dequant gather + CK FMHA；decode/verify 不动 | kernel 2.14-2.18x，E2E **+10.8%**（SLA 甜点 31.75K→35.2K） | 真实 acceptance 无可归因回归 ✓ | 启用 |
| 4 | PTPC dense GEMM（aiter `gemm_a8w8_bpreshuffle`） | MXFP8(1x32) 权重 load 时转 PTPC e4m3 + preshuffle，activation 走 aiter per-token 量化，替 Triton dot_scaled | kernel 1.4-2.5x，E2E **+6.4%** | GSM8K 0.94 通过，但单层 relerr 0.045 | **已停用**（精度决定；代码保留，env 门控默认关） |
| 5 | dense 3 层 prefill 换 CK FMHA | unified backend 内 env 门控换 prefill kernel | 吞吐持平（SLA 边界两臂同 3.75 QPS，8 job A/B） | prefill 误差 2.6%→0.3% | 启用（精度项） |
| 6 | norm/rope/cache 链融合 | sparse 层 8-kernel 链（q/k/index norm + rope + 双 cache 写入）融合成 3 个 Triton kernel | E2E **+2.35%** | baseline-baseline 对照噪声内 ✓ | 启用 |
| 7 | AR epilogue：norm+quant 融合 | AR 后 GemmaRMSNorm + MXFP8 量化合成一个 Triton kernel，预量化喂 dot_scaled GEMM | TTFT p50 **-12%**，饱和点 +1.4% | UT 逐位一致 ✓ | 与 PTPC 互斥，备用（当前栈 PTPC 停用后可启用） |
| 8 | MoE fp8q CSV（16384 bucket） | stage1 直接出 fp8 中间结果（免 stage2 重量化）+ sort 异步重叠；纯 CSV 零代码 | MoE bucket **-11.5%**，E2E ~+1.3% | cosine 5.7e-5 ✓ | 启用 |
| 9 | gemma fused add+RMSNorm（aiter） | add+norm 单 pass（1+w 语义差已处理） | E2E +0.6% | relerr 0.0015 ✓ | 启用 |

## 验证过的负结果（不要重复尝试）

| 项 | 结论 |
|---|---|
| 换更大的 all-reduce（custom AR/symm-mem 替代 RCCL） | prefill 尺寸全带宽 bound，无差别；QR INT4 已是最优（profier 上的 5,498 ms/Mtok AR 桶经低开销 HIP-event 验证是**真实开销**，不是同步等待） |
| AR+norm+quant 整体换 aiter fused op | aiter 的融合 AR 是 bf16-wire custom AR，比 QR INT4 慢 2.7x，净亏 ~9,500 ms/Mtok |
| RoPE 输出缓存到 USM | 生产 ND-extends 路径绕过了 RoPE（输出未被消费），开了反而 +300us/step |
| MoE v33 单 kernel 融合（aiter 0.1.21 移植/overlay） | 只贵不快：本镜像 fp8q 已覆盖其收益，v33 在 0.1.19 上不可用 |
| CPS=512（sparse PA 长上下文调优） | 100K 段微赚，60K 段亏；默认不开 |
| image 内 ROCM_AITER_FA backend 切换 dense attention | 镜像内该 backend 文件自身无法 import（KVCacheLayout 缺失） |

## 最终生产配置（P-only）

```
M3_SHUFFLE_KV_CACHE_LAYOUT=1 + M3_USE_NATIVE_KV_ZERO_OVERLAY=1   # sparse PA
VLLM_TRITON_ATTN_FMHA_PREFILL=1                                  # draft prefill FMHA
VLLM_MINIMAX_M3_FUSED_QKNORM_ROPE_CACHE=1                        # norm/rope/cache 融合（默认开）
M3_AITER_CONFIG_FMOE=minimax_m3_mxfp8_fp8q_prefill_recommended.csv
VLLM_ROCM_AITER_UNIFIED_ATTN_FMHA_PREFILL=1                      # dense FMHA（精度）
# 停用：VLLM_ROCM_MXFP8_PTPC_AITER（PTPC）、index_topk_freq>1（跨层复用）
```

## 数字轨迹（SLA 甜点口径）

| 阶段 | tok/s | vs ATOM |
|---|---:|---:|
| 起点（旧栈） | 26,792 @ 3.45 QPS | 67.8% |
| 4 项集成（含 indexer 共享） | 31,751 @ 4.0 | 80.4% |
| + draft FMHA | 35,176 @ 4.5 | 89.1% |
| + PTPC（停用前的 parity 组合） | 39,735 @ 5.0 | **100.6%** |
| 停用 PTPC + 跨层复用后 | 重测中（预计 ~33-35K） | — |

## 剩余已知空间（未做）

- MoE v33 单 kernel 融合：需镜像升 aiter 0.1.21
- MoE 内部量化的 prequant 入口（aiter kernel 工作）
- shared-expert gate_up/down 与 residual add 的融合（+1.1% 已验证在分支上，待合）
- 跨层复用若精度问题能解决可重新启用（+8.6% 的最大单项之一）
