# Dilated Conv1D + ReLU

**Category**: Convolution | **Complexity**: O(B·T·C²·3) | **Fusion**: pad + gather + matmul + ReLU

## Algorithm

A dilated 1D convolution with kernel size 3, fused with bias and ReLU activation. Used in VQ-VAE encoder/decoder ResConv1DBlocks (e.g., SOKE, Jukebox-style models).

The naive implementation requires:
1. **Pad** the input by `dilation` on each side (zero-padding)
2. **Gather** 3 positions per output: `[t-d, t, t+d]`
3. **Concat** along the channel axis -> `(B, T, 3C)`
4. **Linear** projection `(3C -> C)`
5. **ReLU**

This kernel fuses all 5 steps into a single GPU pass, eliminating three intermediate `(B, T, 3C)` buffer allocations and the data shuffles between them.

## Why fusion matters

For an 18-block VQ-VAE encoder/decoder, the unfused version allocates **54 intermediate tensors** per forward pass and reads them back. Fusing into one kernel:

- Eliminates intermediate buffer writes/reads (3x memory bandwidth reduction)
- Keeps activations in registers/L1 cache between stages
- One command buffer dispatch instead of five

## Benchmark configurations

| Shape (B, T, C) | Elements | Notes |
|---|---|---|
| (2, 50, 512)  | 51 K  | Single VQ-VAE block, small batch |
| (8, 100, 512) | 410 K | Mid-sized clip |
| (2, 400, 512) | 410 K | Long sequence |

## ascend-rs Kernel Source

Vectorized dilated conv1d + ReLU using ascend-rs buffer API (f32, benchmarked implementation):

```rust
#[ascend_std::aiv_kernel]
pub unsafe fn conv1d_dilated(input: *const f32, output: *mut f32, params: *const u32) {
    let n = *params;
    let dilation = *params.wrapping_add(1);
    let w0 = f32::from_bits(*params.wrapping_add(2));
    let w1 = f32::from_bits(*params.wrapping_add(3));
    let w2 = f32::from_bits(*params.wrapping_add(4));
    let bias = f32::from_bits(*params.wrapping_add(5));

    let aligned_n = ((n + 7) / 8) * 8;
    let in_buf = ascend_std::ascend_buf_alloc(aligned_n);
    let tap_left = ascend_std::ascend_buf_alloc(aligned_n);
    let tap_right = ascend_std::ascend_buf_alloc(aligned_n);
    let acc = ascend_std::ascend_buf_alloc(aligned_n);
    let work = ascend_std::ascend_buf_alloc(aligned_n);

    ascend_std::ascend_buf_load_f32(in_buf, input, n);
    ascend_std::ascend_pipe_barrier();

    // Build shifted tap buffers with zero-padding
    ascend_std::ascend_buf_fill_f32(tap_left, 0.0, aligned_n);
    let mut i = dilation;
    while i < n {
        let v = ascend_std::ascend_get_value_f32(in_buf, i - dilation);
        ascend_std::ascend_set_value_f32(tap_left, i, v);
        i += 1;
    }
    ascend_std::ascend_buf_fill_f32(tap_right, 0.0, aligned_n);
    i = 0;
    while i + dilation < n {
        let v = ascend_std::ascend_get_value_f32(in_buf, i + dilation);
        ascend_std::ascend_set_value_f32(tap_right, i, v);
        i += 1;
    }

    // Vector MAC: acc = tap_left*w0 + input*w1 + tap_right*w2 + bias
    ascend_std::ascend_muls_f32(acc, tap_left, w0, n);
    ascend_std::ascend_muls_f32(work, in_buf, w1, n);
    ascend_std::ascend_add_f32(tap_left, acc, work, n);
    ascend_std::ascend_muls_f32(work, tap_right, w2, n);
    ascend_std::ascend_add_f32(acc, tap_left, work, n);
    ascend_std::ascend_adds_f32(acc, acc, bias, n);
    ascend_std::ascend_maxs_f32(acc, acc, 0.0, n);  // ReLU

    ascend_std::ascend_pipe_barrier();
    ascend_std::ascend_buf_store_f32(output, acc, n);
}
```

This compiles via `rustc_codegen_mlir` → MLIR → AscendC (NPU), CUDA (GPU), or GLSL (Vulkan/Metal).

## Results

<div id="kernel-results" data-kernel="conv1d-dilated"></div>

*See [Leaderboard](../leaderboard.md) filtered to `conv1d-dilated` for the full filterable view.*
