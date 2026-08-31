# MiniMax-M3 MXFP8 B300 component capacity update

## Result

The latest B300 D-only sweep moved the stable component boundary to C72.

| Concurrency | Request QPS | Output tok/s | P50 TPOT | P50 TTFT | Acceptance | QPS CV |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 7.298 | 4,379 | 9.54 ms | 3.15 s | 53.10% | 0.50% |
| **72** | **7.642** | **4,585** | **11.86 ms** | **1.88 s** | **53.15%** | **0.24%** |

C72 is the highest stable point in this sweep that passes the D-side
production SLA of P50 TPOT below 16.6 ms. All three trials completed without
failed requests. C80 and higher were unstable and are excluded from the
capacity calculation.

## Updated 72-GPU capacity

Using the current B300 component rates of **P=6.10 QPS** and **D=7.642 QPS**, a
72-GPU deployment contains 18 TP4 engine slots. Exhaustive integer packing
selects **10P + 8D**:

```text
P capacity = 10 * 6.10  = 61.000 QPS
D capacity =  8 * 7.642 = 61.138 QPS
effective QPS = 61.000
```

Using the established MiniMax-M3 workload size of 86,656.702 total tokens per
request:

```text
total TPM = 61.000 * 86,656.702 * 60 = 317.164M
output TPM = 61.000 * 600 * 60 = 2.196M
```

| Metric | Previous P=6.10, D=6.50 | Updated P=6.10, D=7.642 | Change |
|---|---:|---:|---:|
| Packing | 9P + 9D | **10P + 8D** | Rebalanced |
| Effective QPS | 54.90 | **61.00** | **+11.11%** |
| Total TPM | 285.45M | **317.16M** | **+11.11%** |

The updated theoretical deployment is marginally P-bound: the D fleet has
about 0.138 QPS of aggregate headroom. This is a component-capacity estimate,
not a replacement for a sustained open-loop 10-minute PD qualification.

## Measurement contract

- Hardware: B300, TP4 D engine.
- Precision: MiniMax-M3 MXFP8 with FP8 main/index KV cache.
- Decode workload: `DecodeBenchConnector`, token-ID prompts, OSL=600, EAGLE3
  draft=3, synthetic acceptance `[0.7, 0.5, 0.4]`.
- Attention: FlashInfer/TRTLLM production path.
- Method: closed-loop concurrency sweep with three stable repeated trials at
  C72. This establishes the component compute envelope; it is not an
  open-loop sustained-load measurement.
- Raw artifacts:
  `/mnt/shared/homes/b300/peiyuanz/m3-b300-prod-results/2026-08-31-job1788-d-tokenids-r3`

