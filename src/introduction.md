# pu-rs.org -- Processing Unit Ranking System

**The SPECfp for AI accelerators.**

FLOPS don't tell the full story. A chip rated at 1000 TFLOPS means nothing if your softmax kernel only achieves 5% utilization. pu-rs.org measures what matters: **actual kernel execution time** on real hardware, for the operations that AI workloads actually run.

## Why this exists

| What we measure | What others report |
|---|---|
| Softmax latency at (64, 4096) f16 | Peak TFLOPS |
| LayerNorm throughput per watt | Memory bandwidth (theoretical) |
| MatMul efficiency vs roofline | Marketing benchmarks |
| Cost per real GOPS | Cloud $/hour (opaque) |

## Scope

We benchmark the **kernel primitives** that compose every AI model:

| Category | Kernels |
|---|---|
| Activation | Softmax, GELU, SiLU |
| Normalization | LayerNorm, RMSNorm |
| Linear Algebra | GEMM, batched MatMul |
| Attention | Scaled Dot-Product Attention |
| Quantization | VQ-Quantize, INT8 dequant |
| Convolution | Conv1D, dilated Conv1D |
| Reduction | Scatter-add, L1-smooth loss |

## Devices covered

| Type | Vendors |
|---|---|
| GPU | NVIDIA (A100, H100, H200, B200), AMD (MI300X), Apple (M2/M4 Max), Cambricon |
| TPU | Google (v5e, v6e Trillium) |
| NPU | Huawei Ascend (910B, 910C), AWS Trainium2, Intel Gaudi 3 |

## How it works

1. **Run** standardized benchmark scripts on your hardware
2. **Submit** CSV results via pull request
3. **CI validates** format and sanity checks
4. **Leaderboard updates** automatically with per-kernel rankings

All results tagged with git SHA, driver version, toolchain, and number of runs. Median latency reported. [Full methodology](methodology.md).

## End-to-end complement

Per-kernel latency is only half the story — a chip can win on softmax and
still lose on a real model. The [DeepSeek decode page](deepseek-decode.md)
reports end-to-end throughput across five accelerators (Ascend 910B2, TPU
v2-8, Apple M2 Max, NVIDIA T4, AWS Trainium1) from the *same* 13-kernel Rust
source emitted through the ascend-rs MLIR backends.

---

*Built with [ascend-rs](https://ascend-rs.org) kernel infrastructure. Data updated weekly.*
