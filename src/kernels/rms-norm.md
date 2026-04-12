# RMS Norm

**Category**: Normalization | **Complexity**: O(N) per row | **Memory**: 2 passes

## Algorithm

RMSNorm (Zhang & Sennrich 2019) is a simplified LayerNorm used in LLaMA, Gemma, and most modern LLMs. It omits the mean-centering step:

1. **RMS**: Compute root-mean-square: `rms = sqrt(mean(x²) + ε)`
2. **Normalize + Scale**: `y = (x / rms) * gamma`

Compared to LayerNorm, RMSNorm saves one reduction pass (no mean computation) and one elementwise subtraction, yielding ~15% faster inference at equal accuracy.

## ascend-rs Kernel Source

RMS Norm using ascend-rs buffer API (f32):

```rust
/// RMS Norm: y[i] = (x[i] / rms) * gamma[i]
/// where rms = sqrt(mean(x²) + eps)
///
/// params: [n: u32]
#[ascend_std::aiv_kernel]
pub unsafe fn rms_norm(
    input: *const f32,
    gamma: *const f32,
    output: *mut f32,
    params: *const u32,
) {
    let n = *params;
    let eps = 1.0e-5f32;

    let in_buf = ascend_std::ascend_buf_alloc(n);
    let gamma_buf = ascend_std::ascend_buf_alloc(n);
    let work = ascend_std::ascend_buf_alloc(n);
    let rwork = ascend_std::ascend_buf_alloc(n);

    // Load input and gamma
    ascend_std::ascend_buf_load_f32(in_buf, input, n);
    ascend_std::ascend_buf_load_f32(gamma_buf, gamma, n);
    ascend_std::ascend_pipe_barrier();

    // Step 1: x² → work
    ascend_std::ascend_mul_f32(work, in_buf, in_buf, n);
    ascend_std::ascend_pipe_barrier();

    // Step 2: rms = sqrt(mean(x²) + eps)
    let sq_sum = ascend_std::ascend_reduce_sum_f32(work, work, rwork, n);
    let inv_rms = 1.0 / (sq_sum / (n as f32) + eps).sqrt();

    // Step 3: output = (x * inv_rms) * gamma
    ascend_std::ascend_muls_f32(work, in_buf, inv_rms, n);
    ascend_std::ascend_pipe_barrier();
    ascend_std::ascend_mul_f32(work, work, gamma_buf, n);

    ascend_std::ascend_pipe_barrier();
    ascend_std::ascend_buf_store_f32(output, work, n);
}
```

This compiles via `rustc_codegen_mlir` → MLIR → AscendC (NPU), CUDA (GPU), GLSL (Vulkan/Metal), or other targets.

## Benchmark configurations

| Shape | Elements | Bytes (f32) | Notes |
|---|---|---|---|
| (1, 768) | 768 | 3 KB | GPT-2 hidden dim, single token |
| (1, 4096) | 4K | 16 KB | LLaMA-7B hidden dim |
| (64, 768) | 49K | 192 KB | Typical batch, GPT-2 |
| (64, 4096) | 262K | 1 MB | Typical batch, LLaMA |
| (1024, 4096) | 4.2M | 16 MB | Large batch, bandwidth-bound |

All benchmarks use f32.

## Results

<div id="kernel-results" data-kernel="rms-norm"></div>

*See [Leaderboard](../leaderboard.md) filtered to RMS Norm for the full filterable view.*
