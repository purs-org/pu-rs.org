# LayerNorm

**Category**: Normalization | **Complexity**: O(N) per row | **Memory**: 3 passes

## Algorithm

3-pass fused: mean, variance, normalize+affine in one workgroup:

1. **Mean**: Parallel sum reduction, divide by N
2. **Variance**: Parallel sum of (x - mean)^2, compute inverse std
3. **Affine**: `gamma * (x - mean) * inv_std + beta`

Uses SIMD group shuffles for warp-level reductions (1 threadgroup barrier instead of 8).

## tile-rs Kernel Source

LayerNorm in ascend-rs using vectorized AscendC intrinsics (f32, benchmarked implementation):

```rust
#[tile_std::tile_kernel]
pub fn layernorm(input: *const f32, output: *mut f32, len_buf: *const u32) {
    let n = *len_buf;
    let eps = 1.0e-5f32;

    let in_buf = tile_std::__tile_buf_alloc(n);
    let out_buf = tile_std::__tile_buf_alloc(n);
    let work = tile_std::__tile_buf_alloc(n);
    let rwork = tile_std::__tile_buf_alloc(n);

    // DMA load: GM -> local buffer
    tile_std::__tile_buf_load_f32(in_buf, input, n);
    tile_std::__tile_pipe_barrier();

    // Step 1: mean = sum(x) / n
    let sum_val = tile_std::__tile_reduce_sum_f32(work, in_buf, rwork, n);
    let mean = sum_val / (n as f32);

    // Step 2: centered = x - mean
    tile_std::__tile_adds_f32(out_buf, in_buf, -mean, n);
    tile_std::__tile_pipe_barrier();

    // Step 3: var = sum((x - mean)^2) / n
    tile_std::__tile_mul_f32(work, out_buf, out_buf, n);
    tile_std::__tile_pipe_barrier();
    let var_sum = tile_std::__tile_reduce_sum_f32(work, work, rwork, n);
    let inv_std = 1.0 / (var_sum / (n as f32) + eps).sqrt();

    // Step 4: output = centered * inv_std
    tile_std::__tile_muls_f32(out_buf, out_buf, inv_std, n);

    tile_std::__tile_pipe_barrier();
    tile_std::__tile_buf_store_f32(output, out_buf, n);
}
```

This buffer-API kernel is the primary implementation and runs on the Ascend AIV backend. A tile-API `safe::tile_layernorm_f32` variant is additionally lowered by `rustc_codegen_tile` to **Apple Metal** (1/9) — the other 8 backend lowerings (Ascend AIV / CUDA / Vulkan SPIR-V / AWS NKI / AMD AIE / Cambricon BANG / Intel Gaudi / Google TPU) are **future work**. On non-Metal backends, LayerNorm is currently composed at the buffer API as shown above (mean → sub → mul² → mean → sqrt → mul) rather than emitted as a single tile op.

## Benchmark configurations

| Shape | Notes |
|---|---|
| (1, 768) | GPT-2 hidden dim, single position |
| (64, 768) | Typical batch |
| (1024, 768) | Large batch |

## Results

<div id="kernel-results" data-kernel="layernorm"></div>

*See [Leaderboard](../leaderboard.md) filtered to LayerNorm for the full filterable view.*
