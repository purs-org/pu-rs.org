# SiLU / Swish

**Category**: Activation | **Complexity**: O(N) elementwise | **Memory**: 1 pass (fused read+write)

## Algorithm

SiLU (Sigmoid Linear Unit), also known as Swish (Ramachandran et al. 2017), is the gate activation in LLaMA, Mistral, and most modern LLMs:

```
SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

Used in the FFN block:
```
FFN(x) = SiLU(W_gate * x) * (W_up * x)
```

Like GELU, SiLU is memory-bandwidth bound. The compute-to-byte ratio is low (a few FLOPs per element), so throughput is measured in GB/s.

## ascend-rs Kernel Source

SiLU using the tile API — safe entry form (compiles to PTO-MLIR for M-pipe, or to CUDA/SPIR-V/NKI/AIE):

```rust
use ascend_std::tile::{GmView, GmViewMut, safe, tile_load_view_f32, tile_store_view_f32};

#[ascend_std::aiv_kernel]
pub fn tile_silu(
    input:  GmView<'_, 1, 4096, f32>,
    output: GmViewMut<'_, 1, 4096, f32>,
) {
    let x = tile_load_view_f32(&input);
    let y = safe::tile_silu_f32(x);
    tile_store_view_f32(&output, y);
}
```

The kernel body is **pure safe Rust** — shape (rows, cols, dtype) is committed at the type level via const generics, so any host-side mismatch becomes a compile-time error. The `#[aiv_kernel]` attribute rewrites the emitted signature back to raw `*const f32` / `*mut f32` so the launcher toolchain sees the same C ABI; `#[repr(transparent)]` on `GmView`/`GmViewMut` makes this rewrite free at the LLVM IR level.

`safe::tile_silu_f32` decomposes to: neg → exp → add_scalar(1) → reciprocal → mul with original `x`. Compiles via `rustc_codegen_mlir` → MLIR → target-specific code (AscendC, CUDA, GLSL, NKI, AIE).

## Benchmark configurations

| Shape | Elements | Bytes (f32) | Notes |
|---|---|---|---|
| (1, 768) | 768 | 3 KB | GPT-2 hidden dim |
| (1, 4096) | 4K | 16 KB | LLaMA hidden dim |
| (64, 4096) | 262K | 1 MB | Typical batch |
| (1024, 4096) | 4.2M | 16 MB | Large batch |

## Results

<div id="kernel-results" data-kernel="silu"></div>

*See [Leaderboard](../leaderboard.md) filtered to SiLU for the full filterable view.*
