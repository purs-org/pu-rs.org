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

## ascend-rs Implementation

The attention pipeline in ascend-rs combines tile-API matmul with custom Rust kernels for scale and softmax:

```rust
use ascend_rs::prelude::*;

let scale = 1.0f32 / (d_k as f32).sqrt();

// Step 1: scores = Q × K^T  (HGEMM via cube engine)
acl_blas_hgemm(TransN, TransT, TransN,
    seq_len, seq_len, d_k,
    &alpha, &d_q, d_k, &d_k_mat, d_k,
    &beta, &mut d_scores, seq_len,
    HighPrecision, &stream)?;

// Step 2: scores *= 1/√d_k  (custom Rust kernel → NPU)
scale_kernel.launch(1, &stream, &mut [
    d_scores.as_mut_ptr(),  // in-place
    d_scores.as_mut_ptr(),  // output (same buffer)
    d_n_scores.as_mut_ptr(),
    d_scale.as_mut_ptr(),
])?;

// Step 3: weights = softmax(scores)  (custom Rust kernel → NPU)
softmax_kernel.launch(1, &stream, &mut [
    d_scores.as_mut_ptr(),
    d_weights.as_mut_ptr(),
    d_row_len.as_mut_ptr(),
    d_num_rows.as_mut_ptr(),
])?;

// Step 4: output = weights × V  (HGEMM via cube engine)
acl_blas_hgemm(TransN, TransN, TransN,
    seq_len, d_k, seq_len,
    &alpha, &d_weights, seq_len, &d_v, d_k,
    &beta, &mut d_output, d_k,
    HighPrecision, &stream)?;
```

The scale and softmax kernels are written in Rust and compiled via `rustc_codegen_mlir` → MLIR → AscendC (NPU), CUDA (GPU), or GLSL (Vulkan/Metal). The GEMMs use vendor-optimized libraries (aclnnMatmul, cuBLAS, MPSMatrixMultiplication).

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
