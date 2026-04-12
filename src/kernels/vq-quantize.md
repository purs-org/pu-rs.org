# VQ Quantize + EMA Update

**Category**: Quantization | **Complexity**: O(N·K·D) | **Fusion**: L2 distance + argmin + scatter-add

## Algorithm

Vector quantization maps each input vector to its nearest codebook entry, then updates the codebook via exponential moving average (EMA). Used in VQ-VAE training (SOKE, Jukebox, SoundStream).

Pipeline:
1. **L2 distance**: For each input vector x[i] (dim D), compute `||x[i] - c[k]||²` against all K codebook entries
2. **Argmin**: Find nearest codebook entry `k* = argmin_k ||x[i] - c[k]||²`
3. **Quantize**: Output `q[i] = c[k*]` (the nearest codebook vector)
4. **EMA scatter-add**: Accumulate `x[i]` into codebook slot `k*` for EMA update: `sum[k*] += x[i]`, `count[k*] += 1`

Fusing all 4 steps into one kernel eliminates 3 intermediate buffers (distance matrix, index array, scatter workspace).

## ascend-rs Kernel Source

VQ quantize kernel using ascend-rs buffer API (f32):

```rust
/// VQ Quantize: for each input vector, find nearest codebook entry (L2),
/// output the quantized vector, and scatter-add for EMA codebook update.
///
/// params: [n_vectors: u32, n_codes: u32, dim: u32]
#[ascend_std::aiv_kernel]
pub unsafe fn vq_quantize(
    input: *const f32,      // (N, D) input vectors
    codebook: *const f32,   // (K, D) codebook
    output: *mut f32,       // (N, D) quantized output
    cb_sum: *mut f32,       // (K, D) EMA numerator accumulator
    cb_count: *mut u32,     // (K,)   EMA denominator counter
    params: *const u32,
) {
    let n = *params;                      // number of input vectors
    let k = *params.wrapping_add(1);      // codebook size
    let d = *params.wrapping_add(2);      // vector dimension

    let buf_x = ascend_std::ascend_buf_alloc(d);     // current input vector
    let buf_c = ascend_std::ascend_buf_alloc(d);     // current codebook entry
    let buf_diff = ascend_std::ascend_buf_alloc(d);  // x - c
    let buf_work = ascend_std::ascend_buf_alloc(d);
    let buf_rwork = ascend_std::ascend_buf_alloc(d);

    let mut i: u32 = 0;
    while i < n {
        // Load input vector x[i]
        let x_ptr = input.wrapping_add((i * d) as usize);
        ascend_std::ascend_buf_load_f32(buf_x, x_ptr, d);
        ascend_std::ascend_pipe_barrier();

        // Find nearest codebook entry (L2 argmin)
        let mut best_k: u32 = 0;
        let mut best_dist: f32 = f32::MAX;

        let mut j: u32 = 0;
        while j < k {
            let c_ptr = codebook.wrapping_add((j * d) as usize);
            ascend_std::ascend_buf_load_f32(buf_c, c_ptr, d);
            ascend_std::ascend_pipe_barrier();

            // diff = x - c
            ascend_std::ascend_sub_f32(buf_diff, buf_x, buf_c, d);
            ascend_std::ascend_pipe_barrier();
            // diff² = diff * diff
            ascend_std::ascend_mul_f32(buf_diff, buf_diff, buf_diff, d);
            ascend_std::ascend_pipe_barrier();
            // dist = sum(diff²)
            let dist = ascend_std::ascend_reduce_sum_f32(
                buf_work, buf_diff, buf_rwork, d);

            if dist < best_dist {
                best_dist = dist;
                best_k = j;
            }
            j += 1;
        }

        // Output: quantized = codebook[best_k]
        let best_ptr = codebook.wrapping_add((best_k * d) as usize);
        ascend_std::ascend_buf_load_f32(buf_c, best_ptr, d);
        ascend_std::ascend_pipe_barrier();
        let out_ptr = output.wrapping_add((i * d) as usize);
        ascend_std::ascend_buf_store_f32(out_ptr, buf_c, d);

        // EMA scatter-add: cb_sum[best_k] += x[i], cb_count[best_k] += 1
        let sum_ptr = cb_sum.wrapping_add((best_k * d) as usize);
        let sum_buf = ascend_std::ascend_buf_alloc(d);
        ascend_std::ascend_buf_load_f32(sum_buf, sum_ptr, d);
        ascend_std::ascend_pipe_barrier();
        ascend_std::ascend_add_f32(sum_buf, sum_buf, buf_x, d);
        ascend_std::ascend_pipe_barrier();
        ascend_std::ascend_buf_store_f32(sum_ptr, sum_buf, d);

        let count_val = *cb_count.wrapping_add(best_k as usize);
        *cb_count.wrapping_add(best_k as usize) = count_val + 1;

        i += 1;
    }
}
```

This compiles via `rustc_codegen_mlir` → MLIR → AscendC (NPU), CUDA (GPU), or GLSL (Vulkan/Metal). The fused kernel avoids materializing the N×K distance matrix and K-element index array.

## Benchmark configurations

| Shape (N, K, D) | FLOPs | Notes |
|---|---|---|
| (256, 512, 64) | 16.8 M | Small codebook, low-latency inference |
| (1024, 512, 64) | 67.1 M | Typical VQ-VAE batch |
| (1024, 1024, 128) | 268 M | Large codebook, high-dim embeddings |
| (4096, 512, 64) | 268 M | Large batch training |

All benchmarks use f32.

## Results

| Device | Shape (N, K, D) | Latency (μs) | GFLOPS | Notes |
|---|---|---:|---:|---|
| Ascend 910B | (4096, 1024, 128) | 94 | **11,411** | aclnnMatmul L2 trick |
| Ascend 910B | (1024, 1024, 128) | 31 | 8,604 | Large codebook |
| Ascend 910B | (4096, 512, 64) | 43 | 6,243 | Large batch |
| Apple M2 Max | (4096, 1024, 128) | 646 | 1,662 | MPS GEMM + CPU argmin |
| Apple M2 Max | (8192, 512, 64) | 450 | 1,193 | Large batch |

Peak: **11.4 TFLOPS** on Ascend 910B (cube engine via L2 distance matmul trick).
Apple M2 Max peaks at **1.7 TFLOPS** via MPS.

<div id="kernel-results" data-kernel="vq-quantize"></div>

*See [Leaderboard](../leaderboard.md) filtered to VQ Quantize for the full filterable view.*
