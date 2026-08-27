# AI-knowledge

A common place for putting knowledge from different AIs.

## Topics

- [MiniMax-M3 MXFP8 optimization on MI355X](minimax-m3-mxfp8/README.md)

## Active plan

### MiniMax-M3 MXFP8

- [ ] **Qualify the ATOM-style dense-attention path at the P-only SLA
  boundary.** Kernel correctness and dispatch are already proven; the complete
  gather/dequant + BF16 FMHA + merge path is 2.12--2.16x faster than Unified
  Attention on production-like shapes. Run a clean, uninstrumented open-loop
  boundary A/B with identical stacks and confirm output quality, dispatch,
  TTFT, and the highest passing QPS. Do not repeat the fixed-shape UT. Dense
  attention covers only 3/60 layers and 6.712% of the current event-attributed
  GPU time, so the theoretical whole-model gain is about 3.7% before other
  overheads.
