# LayerNorm

**Category**: Normalization | **Complexity**: O(N) per row | **Memory**: 3 passes

## Algorithm

3-pass fused: mean, variance, normalize+affine in one workgroup:

1. **Mean**: Parallel sum reduction, divide by N
2. **Variance**: Parallel sum of (x - mean)^2, compute inverse std
3. **Affine**: `gamma * (x - mean) * inv_std + beta`

Uses SIMD group shuffles for warp-level reductions (1 threadgroup barrier instead of 8).

## ascend-rs Kernel Source

LayerNorm in ascend-rs using vectorized AscendC intrinsics (f32, benchmarked implementation):

```rust
#[ascend_std::aiv_kernel]
pub unsafe fn layernorm(input: *const f32, output: *mut f32, len_buf: *const u32) {
    let n = *len_buf;
    let eps = 1.0e-5f32;

    let in_buf = ascend_std::ascend_buf_alloc(n);
    let out_buf = ascend_std::ascend_buf_alloc(n);
    let work = ascend_std::ascend_buf_alloc(n);
    let rwork = ascend_std::ascend_buf_alloc(n);

    // DMA load: GM -> local buffer
    ascend_std::ascend_buf_load_f32(in_buf, input, n);
    ascend_std::ascend_pipe_barrier();

    // Step 1: mean = sum(x) / n
    let sum_val = ascend_std::ascend_reduce_sum_f32(work, in_buf, rwork, n);
    let mean = sum_val / (n as f32);

    // Step 2: centered = x - mean
    ascend_std::ascend_adds_f32(out_buf, in_buf, -mean, n);
    ascend_std::ascend_pipe_barrier();

    // Step 3: var = sum((x - mean)^2) / n
    ascend_std::ascend_mul_f32(work, out_buf, out_buf, n);
    ascend_std::ascend_pipe_barrier();
    let var_sum = ascend_std::ascend_reduce_sum_f32(work, work, rwork, n);
    let inv_std = 1.0 / (var_sum / (n as f32) + eps).sqrt();

    // Step 4: output = centered * inv_std
    ascend_std::ascend_muls_f32(out_buf, out_buf, inv_std, n);

    ascend_std::ascend_pipe_barrier();
    ascend_std::ascend_buf_store_f32(output, out_buf, n);
}
```

This compiles via `rustc_codegen_mlir` → MLIR → AscendC (NPU), CUDA (GPU), GLSL (Vulkan/Metal), or other targets.

## Why ascend-rs beats MPS by 3x

1. Single-pass kernel vs MPS's separate dispatches
2. No Python/ATen overhead (Rust metal crate -> Metal API directly)
3. Fused command buffer (500 dispatches per commit)
4. No intermediate buffer allocations

## Benchmark configurations

| Shape | Notes |
|---|---|
| (1, 768) | GPT-2 hidden dim, single position |
| (64, 768) | Typical batch |
| (1024, 768) | Large batch |

## Results

<div id="kernel-results" data-kernel="layernorm"></div>

*See [Leaderboard](../leaderboard.md) filtered to LayerNorm for the full filterable view.*
