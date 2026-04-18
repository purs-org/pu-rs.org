# GEMM (MatMul)

**Category**: Linear Algebra | **Complexity**: O(M·K·N) | **Compute**: Cube/Tensor Core bound

## Algorithm

Dense matrix multiplication: **C[M×N] = A[M×K] × B[K×N]**.

The fundamental ML primitive — dominates runtime in transformers (linear projections, attention scores, FFN layers). Performance depends on tiling strategy, memory hierarchy utilization, and hardware matrix units (cube engines, tensor cores).

## ascend-rs Kernel Source

Matrix multiplication in ascend-rs uses the tile API, which compiles to hardware-specific matmul units (cube engine on Ascend, tensor cores on CUDA, etc.).

**Safe entry form** — kernel body is pure safe Rust, shape committed at the type level:

```rust
use ascend_std::tile::{GmView, GmViewMut, safe, tile_load_view_f32, tile_store_view_f32};

#[ascend_std::aiv_kernel]
pub fn tile_matmul(
    a:      GmView<'_, M, K, f32>,
    b:      GmView<'_, K, N, f32>,
    output: GmViewMut<'_, M, N, f32>,
) {
    let at = tile_load_view_f32(&a);
    let bt = tile_load_view_f32(&b);
    let c  = safe::tile_matmul_f32(at, bt);
    tile_store_view_f32(&output, c);
}
```

No `unsafe` blocks in the body. A mismatched `K` between the two operands is a compile-time error. The `#[aiv_kernel]` attribute rewrites the emitted signature to raw `*const f32` / `*mut f32` so the launcher toolchain sees an unchanged C ABI — `#[repr(transparent)]` makes the rewrite free at the LLVM IR level.

This compiles via `rustc_codegen_mlir` → MLIR → target code on **all 9 backends**:
- **Ascend AIV**: PTO-MLIR `pto.tmatmul` → cube engine (320 TFLOPS f16 on 910B)
- **CUDA**: `__shared__` tiled GEMM with `__syncthreads()`
- **Apple Metal / Vulkan SPIR-V**: compute shader with shared-memory tiling
- **AWS NKI** (Trainium): `nki.isa.nc_matmul`
- **AMD AIE**: AIE2P cascade matmul
- **Cambricon BANG**: `__bang_matmul` on MLU tensor units
- **Intel Gaudi**: HPU matmul intrinsic
- **Google TPU**: XLA `dot_general` via OpenXLA

For benchmarking, vendor-optimized libraries are used: aclnnMatmul (Ascend), cuBLAS (CUDA), MPSMatrixMultiplication (Metal).

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
| Apple M2 Max | [4096²]×[4096²] | 17,374 | 7.9 | 53 |
| Apple M2 Max | [8192²]×[8192²] | 139,596 | 7.9 | 53 |
| Apple M2 Max | [2048, 8192]×[8192, 2048] | 8,972 | 7.7 | 51 |
| Apple M2 Max | [4096, 1024]×[1024, 4096] | 4,345 | 7.9 | 53 |
| Tesla T4 | [4096²]×[4096²] | 5,698 | 24.1 | 345 |
| Tesla T4 | [8192²]×[8192²] | 44,099 | 24.9 | 356 |
| Tesla T4 | [2048, 8192]×[8192, 2048] | 2,567 | 26.8 | 383 |
| Tesla T4 | [4096, 1024]×[1024, 4096] | 1,549 | 22.2 | 317 |

Peak: **320 TFLOPS** (f16) on Ascend 910B — saturating the theoretical maximum.
Tesla T4 peaks at **26.8 TFLOPS** (f16) via cuBLAS (torch.matmul).
Apple M2 Max peaks at **7.9 TFLOPS** (f16) via MPSMatrixMultiplication.

<div id="kernel-results" data-kernel="matmul"></div>

*See [Leaderboard](../leaderboard.md) filtered to MatMul for the full filterable view.*
