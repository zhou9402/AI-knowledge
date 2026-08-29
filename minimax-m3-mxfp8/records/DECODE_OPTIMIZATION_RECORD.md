# MiniMax-M3 MXFP8 Decode 优化记录

日期：2026-08-26 至 2026-08-27。
工作产物仓库：`github.com/zhou9402/ATOM` 的 `tools/m3_compare/`（benchmark harness、trace 分析、4 个 UT + 4 个 patch、本记录的全部原始数据）。

## 目标与口径

- 硬件/运行：MI355X ×4（TP4，gfx950），模型 MiniMax-M3-MXFP8-c5454eb0；vLLM 定制镜像 `vllm-minimax-m3-ac750-k3base-a3a9c408e1.sqsh`（vLLM 0.27.2rc1.dev77+gac7509e2b，aiter 0.1.19）+ 运行时 overlay
- 负载：ISL 4096 / OSL 512，ignore_eos，closed-loop，c8/c32/c64（warmup + measured 各一轮，协议与 phase-1 基线 job 2948 完全一致）
- 精度门：GSM8K 5-shot（基线参考 ~0.94）
- 对照目标：ATOM 同 workload decode 参照 TPOT p50 8.37/13.42/17.43 ms、tok/s 853/1958/2888（c8/c32/c64）
- 注意：本记录包含两个阶段——Codex 的 open-loop EAGLE3 campaign（见文末"Codex 阶段存档"，负载口径不同：DecodeBenchConnector、EAGLE3 draft=3、OSL=600），与本次 closed-loop kernel-port campaign。两阶段数字不可直接互比。

## 总览

- 基线（job 2948）：TPOT p50 10.23/19.86/29.49 ms，tok/s 648/1338/1771（c8/c32/c64）
- **Stack A（ws1+ws2+ws3）**：tok/s 680/1398/1900（**+4.9%/+4.5%/+7.3%**），TPOT p50 −4.6%/−4.1%/−6.2%，TTFT −13~18%，零请求错误，GSM8K 0.96（≥基线 0.94）
- **Leg B（Stack A + atom-nightly aiter overlay）**：tok/s 692.6/1456.8/2032.5，c64 **+14.7% vs 基线**，c8/c32/c64 均零错误；确认 flydsl v33 moe kernel 被选中
- Stack B（+ws4 gluon/shuffle）：负收益（c64 −16.2%），ws4 默认关闭
- 距 ATOM 仍有差距（Leg B c64 2032 vs 2888 tok/s），剩余空间见"剩余差距与下一步"

## E2E 结果表（closed-loop decode；TPOT/ITL 单位 ms；基线 = job 2948）

| c | stack | TPOT p50 | TPOT p99 | output tok/s | ΔTPOT p50 | Δtok/s |
|--:|---|---:|---:|---:|---:|---:|
| 8 | baseline | 10.234 | 14.031 | 648.4 | — | — |
| 8 | **A**（ws1+2+3） | **9.762** | 11.645 | **680.4** | **−4.6%** | **+4.9%** |
| 8 | **Leg B**（A+新 aiter） | — | — | **692.6** | — | **+6.8%** |
| 32 | baseline | 19.859 | 29.142 | 1337.9 | — | — |
| 32 | **A** | **19.035** | 26.985 | **1398.1** | **−4.1%** | **+4.5%** |
| 32 | B（+ws4 knobs） | 22.824 | 30.604 | 1200.5 | +14.9% | −10.3% |
| 32 | **Leg B** | — | — | **1456.8** | — | **+8.9%** |
| 64 | baseline | 29.492 | 39.494 | 1771.4 | — | — |
| 64 | **A** | **27.673** | 33.137 | **1900.0** | **−6.2%** | **+7.3%** |
| 64 | B（+ws4 knobs） | 38.099 | 46.815 | 1484.5 | +29.2% | −16.2% |
| 64 | **Leg B** | — | — | **2032.5** | — | **+14.7%** |

（ATOM 参照 c64：TPOT p50 17.43 ms / 2888 tok/s。）

Jobs：smoke 3109 + 3130；Stack A sweep 3132；Stack B sweep 3139（首次 3134 启动崩溃，见"集成问题"）；精度 3135。Leg B：早期两轮 c32/c64 曾 OOM，已修复后重跑通过。

## 四个 Workstream 结论（UT 收益 + E2E 判决）

| WS | 内容 | UT 收益 | 数值变更 | E2E 判决 |
|---|---|---|---|---|
| **WS1 MoE** | aiter tuned fmoe CSV（把 ATOM 命中的解表行带进 vLLM 的 `AiterMxfp8Experts` 查找路径） | 0.1.19 上 c32 中性；M=8 −11.4%、M=1024 −22.9%；跨镜像（atom-nightly aiter）**−11.5%/层**（M=32 uniform，hot48 −6.9%），输出 rel_l2 0.0025 | 无数值回归（fp8q 变体 rel_l2 0.003 vs 默认 bf16-out 0.016，更准） | 机制有效但**大头绑定 aiter 版本**（~1.65ms/step 的 moe1/moe2 kernel 差需 atom-nightly 的 flydsl `moe2_layout` + v33 fp8q moe1）；Leg B 已兑现 |
| **WS2 GEMM** | dense MXFP8 linear 转 aiter PTPC a8w8（`gemm_a8w8_bpreshuffle`），含 decode M-guard | kernel **1.23–2.42×**（7 个 shape 类），步合计 +10.2%；叠加 ws3 prequant 后步合计 1.86× | **改数值**：MXFP8(1x32) 权重 load 时重量化为 per-channel fp8（relerr 0.046 vs Triton 0.037） | **决策：生产禁止，默认关、opt-in**（GSM8K 虽过，但单层 relerr 超标；ATOM 的 PTPC 是它自己 recipe 的一部分，不构成我们跟进的依据）。Stack A 的 E2E 数字含 ws2，属验证性运行 |
| **WS3 AR+norm** | aiter fused AR+GemmaRMSNorm（复用 vLLM TP group 的 ca_comm）+ FFN-AR 延迟融合 + cudagraph capture patch | **1.20–1.22×**（bf16 AR+norm，35.3→29.1µs/call）；~120 站点/步 ≈ 0.7ms/step | **逐位一致**（max_abs 0.0 vs bf16-rounded fp32 ref），无数值变更 | 安全启用。EAGLE 并存时用 `=post_attn`；PP>1 自动禁用；勿与 EP 组合 |
| **WS4 attn** | sparse decode 换 aiter gluon `pa_decode_gluon`（page-16 SHUFFLE KV）+ host 开销削减 glue | kernel（graph 内）**1.11×/1.55×/1.96×** @b8/32/64；含 block-table build 仍 1.25×/1.63× @c32/64 | allclose 通过（fp8 容差，max\|diff\| 3.5e-3） | **E2E 负收益，默认关闭**：vLLM attend 走 eager（`eager_break_during_capture`），python glue + block-table 构建的 host 开销淹没 kernel 收益（patched gluon eager 53µs vs Triton 21µs @c32）。需 attend 进 cudagraph 后才能解锁 |

**重要更正**：ws4 推翻了 Codex 阶段"AITER gluon sparse PA 慢 2.7×"的结论——那个数字是 eager 下测的（或 geometry 不同）；graph 内 kernel 层面 gluon 在 M3 decode geometry 上稳定赢 Triton split-K（且优势随 batch 增大，Triton 固定 TARGET_GRID=256 在 c64 填不满 GPU）。

## Trace 分析依据（phase 2，job 21/26 采集）

每 decode step 每 rank GPU busy：vLLM 17155µs vs ATOM 11436µs（Δ +5720µs）。最大差距桶（vLLM min-rank vs ATOM）：

| 类别 | vLLM µs/step | ATOM µs/step | Δ | 归属 |
|---|---:|---:|---:|---|
| Dense GEMM + quant-prep | 3716 + 1172(quant) | 1266 | +2450 / +1172 | WS2（+WS3 epilogue） |
| MoE experts | 6542 | 4893 | +1649 | WS1 |
| Norm（未融合） | 608 | 5 | +603 | WS3 |
| Copies/elementwise | 1159 | 70 | +1089 | WS3（registered buffer/cudagraph fix） |
| Sparse attention | 906 | 833 | +73 | WS4（graph 口径下实际 ~300/680µs/step @c32/64） |
| Index top-k | 302 | 473 | **−171** | vLLM 更快，可反向移植给 ATOM |

另：vLLM custom AR spin-wait 造成巨大 rank 间偏斜（min-rank 1513µs vs max-rank 56180µs/step），rollup 用 min-rank 作口径。

## Stack B 判决（ws4）

E2E 下 gluon sparse-decode 路径在 vLLM eager attend 下大败：c32/c64 TPOT p50 +14.9%/+29.2%（tok/s −10.3%/−16.2%）。Stack B 日志确认 gluon 路径真实启用（`AITER_SPARSE_PA ... topk_blocks=16`），回退不是原因。kernel 级 ~5–12µs/层的收益被 eager host 开销淹没，与 ws4 自己的预测一致。**ws4 knobs 不得默认开启**；收益解锁前置条件是 attend 块可 cudagraph 捕获（metadata indirection）。

## Leg B：atom-nightly aiter overlay（Stretch）

> 来源：agent-12 口头汇报（`ports/integ-e2e.md` 的 "Stretch" 段在写本记录时仍未回填，数字以本节为准）。

- 做法：Stack A 之上叠加 atom-nightly aiter 的 PYTHONPATH overlay（`/app/aiter-test`），镜像本身不动；ws1 的 CSV 机制直接携带 ATOM 的调优行。
- 服务器日志确认 flydsl v33 moe kernel 被选中（`moe2_layout` launcher + v33 fp8q moe1 gen——即 ws1 跨镜像 UT（job 3088）验证过的 ATOM trace 同款 kernel）。
- 结果：tok/s 692.6/1456.8/2032.5 @c8/32/64，三档均零请求错误；**c64 +14.7% vs 基线**。
- 早期两轮 c32/c64 曾因 OOM 失败，已修复（新 aiter 的显存占用差异），重跑通过。

## 集成问题记录（merge 时踩过的坑）

1. **ws3 patch base 不匹配**：patch 针对 overlay worktree 的 model.py（sha 996ae451），基线跑的是镜像内 model.py（sha 7830e9e5）；且 worktree 版引用了镜像不存在的符号。解决：把 ws3 重基于镜像文件，9/11 hunk 直接打上，#4/#9 手工修（FusedMoEFactory 加 `reduce_results`、`self._defer_ffn_ar` 设置）。
2. **aiter custom AR 要求纯 TCPStore，vLLM 用的 FileStore**：首个 smoke（3109）ws3 静默失效。解决：integ overlay 加 `vllm/utils/network_utils.py`，把 `VLLM_MINIMAX_M3_AITER_FUSED_AR_NORM` != off 也纳入 TCP rendezvous 条件。
3. **ws4 的 `amd/ops/sparse_pa.py` 在 worktree 无 base**：从镜像 seed（sha 008a1461，与 `ports/ws4/sparse_pa.py.orig` 逐字节一致）后再打 patch。
4. **Stack B 启动崩溃**（job 3134）：SHUFFLE KV layout 下原生 Triton KV zeroer 挂（`Triton Error [HIP]: invalid argument`）。解决：挂载 `native_kv_zero_overlay/utils.py` + 预编译 `m3_kv_zero_stride.so`。
5. ws1 担心的 model.py 冲突不存在：ws1-moe.patch 只动 CSV、harness、新文件 `aiter_mxfp8_moe.py`。
6. ws2 远端 patch 比其文档新（含 small-M decode guard）；以 doc-canonical 的 `rocm_native.py.patched`（sha 996157cf）为准。

## 剩余差距与下一步（c64：Leg B 2032 vs ATOM 2888 tok/s）

1. **aiter 镜像升级**（ws1 大头）：把 atom-nightly aiter（flydsl `moe2_layout` launcher + v33 fp8q moe1 gen）合入正式镜像，替代 PYTHONPATH overlay；~1.65ms/step 的 moe1/moe2 差已在 Leg B 兑现一半路径。
2. **attend 进 cudagraph**（ws4 解锁）：metadata indirection 改造后可拿 kernel 级 ~300µs/step@c32、~680µs/step@c64。
3. **quant epilogue 接线**（ws2+ws3 组合）：把 ws3 的 fused AR+norm+quant 输出 `(xq, xs)` 接进 ws2 的 prequant hook，消灭 ~1172µs/step 的 activation quant-prep（ws2 "patched+prequant" 列：dense-GEMM 步合计 1.86×）。注意此项依赖 ws2 的 PTPC 路线，受 WS2 数值决策约束。
4. **index top-k 反向移植给 ATOM**：vLLM 的 score/topk split 对（302µs/step）快于 ATOM 的 fused 单 kernel（473µs/step；per-call 25.1µs vs 15.5µs），把 vLLM 的拆法搬回 ATOM。
5. ws1 的 decode CSV 在 0.1.19 上 c32 中性（M=8 +13%、M=1024 +23%）；FSE (129,5) 可作为 decode-only 部署决策（注意 ws1-moe.md 记录的 FSE prefill 回退）。

## 复现指引

- 本地（ATOM 仓库）：`tools/m3_compare/`——
  - harness：`12_capture_atom_decode.sh`、`13_capture_vllm_decode.sh`、`14_capture_vllm_decode_integ.sh`、`15_accuracy_vllm_integ.sh`、`send_decode.py`
  - trace 分析：`analysis-decode/`（job 21/26 采集，rollup + porting candidates）
  - UT：`30_kernel_ut/`（ut_moe_decode.py、test_mxfp8_gemm_decode.py、test_fused_ar_norm_decode.py、test_sparse_attn_decode.py）
  - patch 与报告：`ports/`（ws1-moe.patch、ws2-gemm.patch、ws3-arnorm.patch、ws4/ws4-attn.patch、integ-combined.patch + 各 ws*.md）
- 远端（vultr-mi355x）：`/mnt/vfs/homes/peiyuanz/m3-compare/`（results-decode/ 全部 JSON 与日志），integ overlay `/mnt/vfs/homes/peiyuanz/vllm-minimax-kernel-next-integ`
- 环境变量清单：
  - Stack A：`VLLM_ROCM_MXFP8_MOE_DECODE_FMOE_CSV=$AUDIT_ROOT/benchmarks/kernels/minimax_m3/minimax_m3_mxfp8_decode_fmoe.csv`、`VLLM_ROCM_MXFP8_PTPC_AITER=1`（验证用；生产决策为默认关）、`VLLM_MINIMAX_M3_AITER_FUSED_AR_NORM=all`（默认）
  - Stack B：Stack A + `VLLM_ROCM_USE_AITER=1 VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` + native KV-zero overlay（`M3_USE_NATIVE_KV_ZERO_OVERLAY=1`、`M3_NATIVE_KV_ZERO_LIBRARY`）
  - 通用：`M3_USE_MINIMAX_GRAPH_METADATA_OVERLAY=1`、每节点独立 aiter JIT 目录（冷编译 ~15–20 min）

---

## Codex 阶段存档（open-loop EAGLE3 campaign，2026-08-27 早）

以下为 Codex 此前建立的记录，口径：DecodeBenchConnector、EAGLE3 draft=3、synthetic acceptance [0.7,0.5,0.4]、exact OSL=600、open-loop；SLA TPOT P50 < 16.6ms；当时 deploy-safe 边界 2.20 QPS / TPOT P50 16.439ms。**保留备查；其中 "gluon 慢 2.7×" 结论已被本次 ws4 推翻（见上）。**

### 事件归因（representative graph 39.195ms）

| Area | Graph share | Status |
|---|---:|---|
| Routed MoE + separate shared expert | 44.6% | Largest remaining target |
| Indexer + sparse-decode kernels | 17.3% | Top-K bucket deployed; split/merge remains |
| Sparse QKV, QK-norm/cache, projections | ~10–11% | Major exact buckets deployed |
| AllReduce/norm | ~3.3% | Low priority for D |
| Dense attention | ~2.9% | Only three layers; low priority |

### Qualified and deployed

| Change | Evidence | Capacity effect |
|---|---|---|
| Exact D Top-K/index-score bucket | D C22 throughput +4.02%, TPOT P50 −4.78% | Deploy-safe |
| Static-scale sparse QK-norm/cache fusion | D C24 throughput +4.10%, TPOT P50 −2.88% | Boundary 2.10→2.20 QPS |
| Exact-shape MXFP8 dense GEMM selection | D component +1.53–2.12% | Deploy-safe |
| BF16 EAGLE exact GEMM rows | Kernel +3.04–25.81% | Dispatch-qualified |

### Active candidates（当时）

- **Fused routed+shared expert（最高优先级）**：E128/top4+独立 shared 464.0µs → fused E129/top5 430.0µs（−7.93%）；rel L2 0.0051/cosine 0.99999；同机 D C22 A/B：吞吐 +6.02%、TPOT P50 −4.30%。缺最后一道门：与完整 D 栈组合后在 2.20 QPS 边界做 clean open-loop 搜索。
- **Sparse decode split-K + merge**：split-K 43.3µs + merge 23.8µs/层，57 层 ≈ 3.82ms ≈ 9.8% graph。下一步：correctness-first UT 减 split/merge 中间流量。

### Rejected or exhausted（不要重复尝试）

| Candidate | Result | Decision |
|---|---|---|
| AITER Gluon sparse PA for D | 138.72µs vs Triton 51.32µs | ~~Reject: 2.70x slower~~ **已被 ws4 推翻**：该数字系 eager/不同 geometry 下测得；graph 内 gluon 赢 1.11–1.96× |
| Additional sparse-decode split count | Neutral or slower | Do not repeat |
| Sparse-decode head tiles 8/4 | 80.64/83.64µs vs 51.40µs | Reject |
| Generic routed-FMoE CSV tuning | Best graph gain ~1.57% | Exhausted; kernel change required |
| D INT4 QuickReduce | 通信占比小，精度/launch 风险不值 | Low priority |

## Gluon attention 全量替换（D-only + PD，2026-08-29）

### 背景

- PR52849（gluon 支持 EAGLE3 多 token + dense 层）打通 gluon 全量替换路径；此前 WS4 的 eager 教训（python glue + block-table host 开销淹没 kernel 收益）在 M3-AMD 树上不存在——decode 已全 graph，eager 问题不再适用。

### A/B 判决（M3-AMD 栈，C24/C28，生产协议）

gluon 全开**打平**：kernel 赢、glue 亏，两者相抵。

### 拆账定位

| 项 | 结论 |
|---|---|
| dense gluon | −34%/call，收益已兑现 |
| sparse glue（KV insert + 页表构建） | ~2.2ms/step，吃掉 kernel 收益 |
| **EAGLE3 draft** | 占每步 28–33%，是最大杠杆 |

### 收益实现（两步）

1. **融合 SHUFFLE KV/index 写入**：移植 ATOM 的 fused kernel（UT 3.7×，字节一致）→ 修平 sparse glue 回退。
2. **draft 也上 gluon**：gate 放宽 query-group 1 + draft metadata no-op（排查中发现 draft 实际跑 TP4 而非配置的 TP1）→ **C24 1745.8 tok/s / TPOT 12.52ms（较基线 +17.4% / −14.6%）**，C28 TPOT 13.98ms（−16.8%）。

### 正确性

抓到并修复 sparse-prefill 页表 stride bug（8 vs 16 packed；benchmark 的假 KV 永远踩不到该路径）→ 修复后 GSM8K 0.99（参照 0.96）。

### SLA 甜点（修复后全 gluon 栈）

| C | output tok/s | TPOT p50 (ms) | 备注 |
|--:|---:|---:|---|
| 24 | 1745.8 | 12.52 | |
| 28 | 1857.6 | 13.98 | |
| 32 | 2048 | — | |
| 36 | 2065.9 | — | |
| **40** | **2254.8** | **15.88** | **3.76 QPS，SLA 内最高点** |
| 44 | — | 17.28 | 破线 |

C28 TTFT 尾巴是冷启动假象（7 次干净复测未复现）。

### PD 双端 gluon 验证（2026-08-29）

- 1P1D + MoRIIO（RDMA）跑通；temp 0 钉死下 **PD 0.95 == D-only 0.95**，无 PD 回退。
- 口径注意：历史参考值是 temp=1.0 采样跑的，本次起统一 temp 0。
- stride-aware connector 不需要旧的 shuffled-KV 注册 patch。

### 代码与数据位置

- 代码：`github.com/zhou9402/vllm` 分支 `opt/pr52849-gluon-mtp-m3amd` + `opt/draft-gluon-pa`（de29b03ab）
- 数据（vultr）：`/mnt/vfs/homes/peiyuanz/m3-compare/results-gluon-m3amd/`（`REPORT.md`、`sweetspot-gluon.md`、`results-pd-gluon/verdict.md`）

### 遗留

- C28 TTFT 双峰尾巴的根治
- prefill 融合发射的正确修法
- dense gluon 的进一步调优空间

## Change log

- 2026-08-27：Codex 建立记录，优先级为 final-stack fused-expert 验证先于 sparse-decode kernel 工作。
- 2026-08-27：kernel-port campaign（ws1–ws4 + 集成 E2E + Leg B）合并入本记录；Stack A +4.5~7.3%，Leg B c64 +14.7%；ws4 推翻 "gluon 2.7× slower" 结论。
- 2026-08-29：Gluon attention 全量替换（D-only + PD）——draft 上 gluon 后 C24 +17.4% / TPOT −14.6%；SLA 甜点 C40 2254.8 tok/s / 3.76 QPS / TPOT 15.88ms；PD 双端 gluon 验证通过（PD 0.95 == D-only 0.95，temp 0）。
