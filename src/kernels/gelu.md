# GELU

**Category**: Activation | **Complexity**: O(N) elementwise | **Memory**: 1 pass (fused read+write)

## Algorithm

GELU (Gaussian Error Linear Unit, Hendrycks & Gimpel 2016) is the standard activation in BERT, GPT, LLaMA, and most transformer models:

```
GELU(x) = x · Φ(x) = x · 0.5 · (1 + erf(x / √2))
```

The fast tanh approximation (used in PyTorch `gelu(approximate='tanh')`):

```
GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

GELU is memory-bandwidth bound — the compute-to-byte ratio is low (a few FLOPs per 4-byte element), so peak throughput is measured in GB/s rather than TFLOPS.

## ascend-rs Kernel Source

GELU using ascend-rs buffer API (f32, tanh approximation):

```rust
/// GELU activation: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///
/// params: [n: u32]
#[ascend_std::aiv_kernel]
pub unsafe fn gelu(
    input: *const f32,
    output: *mut f32,
    params: *const u32,
) {
    let n = *params as usize;
    let sqrt_2_pi: f32 = 0.7978845608; // sqrt(2/pi)
    let coeff: f32 = 0.044715;

    let mut i: usize = 0;
    while i < n {
        let x = *input.wrapping_add(i);
        let x3 = x * x * x;
        let inner = sqrt_2_pi * (x + coeff * x3);
        // tanh via exp: tanh(z) = (e^2z - 1)/(e^2z + 1)
        let e2z = (2.0 * inner).exp();
        let tanh_val = (e2z - 1.0) / (e2z + 1.0);
        *output.wrapping_add(i) = 0.5 * x * (1.0 + tanh_val);
        i += 1;
    }
}
```

**Vectorized version** using buffer intrinsics:

```rust
#[ascend_std::aiv_kernel]
pub unsafe fn gelu_vec(
    input: *const f32,
    output: *mut f32,
    params: *const u32,
) {
    let n = *params;
    let in_buf = ascend_std::ascend_buf_alloc(n);
    let work = ascend_std::ascend_buf_alloc(n);
    let work2 = ascend_std::ascend_buf_alloc(n);

    ascend_std::ascend_buf_load_f32(in_buf, input, n);
    ascend_std::ascend_pipe_barrier();

    // x³
    ascend_std::ascend_mul_f32(work, in_buf, in_buf, n);
    ascend_std::ascend_pipe_barrier();
    ascend_std::ascend_mul_f32(work, work, in_buf, n);
    ascend_std::ascend_pipe_barrier();
    // 0.044715 * x³
    ascend_std::ascend_muls_f32(work, work, 0.044715, n);
    ascend_std::ascend_pipe_barrier();
    // x + 0.044715 * x³
    ascend_std::ascend_add_f32(work, in_buf, work, n);
    ascend_std::ascend_pipe_barrier();
    // sqrt(2/pi) * (x + 0.044715 * x³)
    ascend_std::ascend_muls_f32(work, work, 0.7978845608, n);
    ascend_std::ascend_pipe_barrier();

    // Store result
    ascend_std::ascend_buf_store_f32(output, work, n);
}
```

These Rust kernels compile via `rustc_codegen_mlir` → MLIR → AscendC (NPU), CUDA (GPU), GLSL (Vulkan/Metal), or other targets.

## Benchmark configurations

| Shape | Elements | Bytes (f32) | Notes |
|---|---|---|---|
| (1, 768) | 768 | 3 KB | GPT-2 hidden dim |
| (1, 4096) | 4K | 16 KB | LLaMA hidden dim |
| (64, 768) | 49K | 192 KB | Typical batch |
| (64, 4096) | 262K | 1 MB | Bandwidth-bound |
| (1024, 4096) | 4.2M | 16 MB | Large batch |

All benchmarks use f32.

## Results

<div id="kernel-results" data-kernel="gelu"></div>

*See [Leaderboard](../leaderboard.md) filtered to GELU for the full filterable view.*
