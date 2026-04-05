# Methodology

## Measurement protocol

1. **Warmup**: 50 iterations discarded
2. **Measurement**: 500 iterations, **median** latency reported
3. **Amortization**: Dispatch overhead amortized by batching 500 kernel launches into one command buffer where supported
4. **Isolation**: Benchmarks run on idle systems, no background GPU workloads

## What we measure

**Kernel-only time**: the GPU/NPU execution time for a single kernel dispatch, excluding:
- Host-to-device data transfer (data assumed resident)
- Command buffer creation overhead (amortized)
- Python/framework overhead

This isolates the hardware+compiler efficiency from the software stack.

## Reporting

| Metric | Definition |
|---|---|
| Latency (us) | Median kernel execution time in microseconds |
| GOPS | Throughput: operations / latency |
| GOPS/$ | Throughput / device MSRP in USD |
| GOPS/W | Throughput / TDP in watts |

## Standardized configurations

Each kernel is benchmarked at these canonical shapes:

| Kernel | Shapes | Dtypes |
|---|---|---|
| Softmax | (1,1024), (64,1024), (64,4096) | f32, f16 |
| LayerNorm | (1,768), (64,768), (1024,768) | f32, f16 |
| GEMM | (1024,1024,1024), (4096,4096,4096) | f32, f16, bf16 |
| Attention | (1,32,128,128), (32,32,2048,128) | f32, f16 |

## How to submit

See [Submit Results](submit.md).
