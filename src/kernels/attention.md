# Attention (Scaled Dot-Product)

**Category**: Attention | **Complexity**: O(B·H·S²·D) | **Compute**: Cube/Tensor Core bound

## Algorithm

Scaled dot-product attention: **Output = softmax(Q·K^T / √d) · V**

The core transformer primitive — computes attention weights from queries and keys, applies softmax normalization, then produces a weighted sum of values. Dominates runtime in all transformer architectures (GPT, BERT, LLaMA, etc.).

Pipeline:
1. **Scores** = Q × K^T — matmul (S×D) × (D×S) → (S×S)
2. **Scale** by 1/√d — element-wise multiply
3. **Softmax** along last axis — numerically stable (max → sub → exp → sum → div)
4. **Output** = Weights × V — matmul (S×S) × (S×D) → (S×D)

## ascend-rs Kernel Source

The attention pipeline in ascend-rs is built from composable Rust kernels compiled to NPU/GPU/Metal via MLIR:

**Scale kernel** — element-wise multiply by 1/√d:
```rust
#[ascend_std::aiv_kernel]
pub unsafe fn scale_f16(
    input: *const u16, output: *mut u16,
    n: *const u32, scale: *const f32,
) {
    let count = *n;
    let scale_val = *scale;
    let buf_in = ascend_std::ascend_buf_alloc(count);
    let buf_out = ascend_std::ascend_buf_alloc(count);

    ascend_std::ascend_buf_load_f16(buf_in, input, count);
    ascend_std::ascend_pipe_barrier();
    ascend_std::ascend_muls_f16(buf_out, buf_in, scale_val, count);
    ascend_std::ascend_pipe_barrier();
    ascend_std::ascend_buf_store_f16(output, buf_out, count);
}
```

**Softmax kernel** — row-wise numerically stable softmax:
```rust
#[ascend_std::aiv_kernel]
pub unsafe fn softmax_rows_f16(
    input: *const u16, output: *mut u16,
    row_len: *const u32, num_rows: *const u32,
) {
    let cols = *row_len;
    let rows = *num_rows;
    let buf_in = ascend_std::ascend_buf_alloc(cols);
    let buf_out = ascend_std::ascend_buf_alloc(cols);
    let buf_work = ascend_std::ascend_buf_alloc(cols);
    let buf_rwork = ascend_std::ascend_buf_alloc(cols);

    let mut row = 0u32;
    loop {
        if row >= rows { break; }
        let in_ptr = input.wrapping_add((row * cols) as usize);
        let out_ptr = output.wrapping_add((row * cols) as usize);

        ascend_std::ascend_buf_load_f16(buf_in, in_ptr, cols);
        ascend_std::ascend_pipe_barrier();

        let max_val = ascend_std::ascend_reduce_max_f16(
            buf_work, buf_in, buf_rwork, cols);
        ascend_std::ascend_adds_f16(buf_out, buf_in, -max_val, cols);
        ascend_std::ascend_pipe_barrier();
        ascend_std::ascend_exp_f16(buf_out, buf_out, cols);
        ascend_std::ascend_pipe_barrier();
        let sum_val = ascend_std::ascend_reduce_sum_f16(
            buf_work, buf_out, buf_rwork, cols);
        ascend_std::ascend_muls_f16(buf_out, buf_out, 1.0 / sum_val, cols);

        ascend_std::ascend_pipe_barrier();
        ascend_std::ascend_buf_store_f16(out_ptr, buf_out, cols);
        row += 1;
    }
}
```

These Rust kernels compile via `rustc_codegen_mlir` to MLIR, then to AscendC (NPU), CUDA (GPU), GLSL/SPIR-V (Vulkan/Metal), NKI (Trainium), or AIE (AMD). The GEMMs (Q×K^T and Weights×V) use vendor-optimized libraries (aclnnMatmul, cuBLAS, MPSMatrixMultiplication).

## Benchmark configurations

| Shape (B, H, S, D) | FLOPs | Notes |
|---|---|---|
| (1, 1, 128, 64) | 4.2 M | Small baseline, dispatch overhead test |
| (1, 1, 512, 64) | 67 M | Medium sequence |
| (1, 1, 1024, 64) | 268 M | GPT-2 scale |
| (1, 1, 2048, 64) | 1.1 G | Long context |
| (1, 1, 4096, 64) | 4.3 G | Very long context (quadratic scaling) |
| (1, 8, 512, 64) | 537 M | 8-head, GPT-2 like |
| (1, 12, 512, 64) | 805 M | 12-head, BERT-base |
| (1, 32, 512, 64) | 2.1 G | 32-head, LLaMA-7B |
| (1, 32, 1024, 128) | 17.2 G | 32-head, LLaMA-2-7B |
| (1, 32, 2048, 128) | 68.7 G | 32-head, long context |

All benchmarks use f16 input with f16 output. FLOPs ≈ 4·B·H·S²·D (two matmuls dominate).

## Results

| Device | Shape (B,H,S,D) | Latency (μs) | TFLOPS | Notes |
|---|---|---:|---:|---|
| Ascend 910B | (1,32,1024,128) | 310 | **55.4** | aclnnMatmul+Softmax, manual pipeline |
| Ascend 910B | (1,32,2048,128) | 1,459 | 47.1 | Memory-bound at long context |
| Ascend 910B | (1,1,4096,64) | 149 | 28.9 | Single-head, large S |
| Ascend 910B | (1,32,512,64) | 105 | 20.4 | 32-head, short context |
| Tesla T4 | (1,32,1024,128) | 2,609 | 6.6 | F.scaled_dot_product_attention |
| Tesla T4 | (1,32,2048,128) | 5,067 | 13.6 | Flash attention backend |
| Tesla T4 | (1,1,4096,64) | 974 | 4.4 | Single-head |
| Tesla T4 | (1,32,512,64) | 427 | 5.0 | 32-head |
| Apple M2 Max | (1,32,1024,128) | 138,819 | 0.12 | MPS GEMM + CPU softmax |
| Apple M2 Max | (1,1,4096,64) | 60,647 | 0.07 | MPS GEMM + CPU softmax |

Peak: **55.4 TFLOPS** (f16) on Ascend 910B.
Tesla T4 peaks at **13.6 TFLOPS** (f16) via PyTorch SDPA.
Apple M2 Max peaks at **0.14 TFLOPS** — bottlenecked by CPU softmax (no fused MPS attention).

<div id="kernel-results" data-kernel="attention"></div>

*See [Leaderboard](../leaderboard.md) filtered to Attention for the full filterable view.*
