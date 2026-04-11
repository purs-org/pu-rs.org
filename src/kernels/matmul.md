# GEMM (MatMul)

**Category**: Linear Algebra | **Complexity**: O(M·K·N) | **Compute**: Cube/Tensor Core bound

## Algorithm

Dense matrix multiplication: **C[M×N] = A[M×K] × B[K×N]**.

The fundamental ML primitive — dominates runtime in transformers (linear projections, attention scores, FFN layers). Performance depends on tiling strategy, memory hierarchy utilization, and hardware matrix units (cube engines, tensor cores).

## Benchmark configurations

| Shape (A × B) | FLOPs | Notes |
|---|---|---|
| [1024, 1024] × [1024, 1024] | 2.1 G | Small square, tests dispatch overhead |
| [4096, 4096] × [4096, 4096] | 137 G | Standard benchmark, bandwidth→compute transition |
| [8192, 8192] × [8192, 8192] | 1.1 T | Large square, saturates compute units |
| [16384, 16384] × [16384, 16384] | 8.8 T | Full hardware saturation |
| [1024, 4096] × [4096, 1024] | 8.6 G | Rectangular, typical FFN down-projection |
| [4096, 1024] × [1024, 4096] | 34.4 G | Rectangular, typical FFN up-projection |
| [2048, 8192] × [8192, 2048] | 67.1 G | Transformer-scale attention projection |

All benchmarks use f16 input with f16 output (or f32 accumulation where supported).

## Results

| Device | Shape | Latency (μs) | TFLOPS | GOPS/W |
|---|---|---:|---:|---:|
| Ascend 910B | [4096²]×[4096²] | 437 | 314.5 | 1014 |
| Ascend 910B | [8192²]×[8192²] | 3,614 | 304.2 | 981 |
| Ascend 910B | [16384²]×[16384²] | 27,467 | **320.2** | 1033 |
| Ascend 910B | [2048, 8192]×[8192, 2048] | 245 | 280.0 | 903 |
| Ascend 910B | [4096, 1024]×[1024, 4096] | 132 | 260.1 | 839 |

Peak: **320 TFLOPS** (f16) on Ascend 910B — saturating the theoretical maximum.

<div id="kernel-results" data-kernel="matmul"></div>

*See [Leaderboard](../leaderboard.md) filtered to MatMul for the full filterable view.*
