# RoPE (Rotary Position Embedding)

**Category**: Positional Encoding | **Complexity**: O(S*D) elementwise | **Memory**: 2 passes (read+write, plus cos/sin tables)

## Algorithm

RoPE (Su et al. 2021) encodes position by rotating pairs of dimensions at frequency-dependent rates:

```
For each pair (x[2i], x[2i+1]):
  theta = pos / 10000^(2i/d)
  x'[2i]   = x[2i]*cos(theta) - x[2i+1]*sin(theta)
  x'[2i+1] = x[2i]*sin(theta) + x[2i+1]*cos(theta)
```

Used in every modern LLM (LLaMA, Mistral, GPT-NeoX, Qwen, etc.) to encode token position in Q/K vectors. RoPE is bandwidth-bound for short sequences and compute-bound (cos/sin) for long sequences.

## ascend-rs Kernel Source

RoPE using the tile API — safe entry form (compiles to PTO-MLIR for M-pipe, or to CUDA/SPIR-V/NKI/AIE):

```rust
use ascend_std::tile::{GmView, GmViewMut, safe, tile_load_view_f32, tile_store_view_f32};

#[ascend_std::aiv_kernel]
pub fn tile_rope(
    input:  GmView<'_, 1, 128, f32>,
    output: GmViewMut<'_, 1, 128, f32>,
) {
    let x = tile_load_view_f32(&input);
    let y = safe::tile_rope_f32(x, 0);  // base position = 0
    tile_store_view_f32(&output, y);
}
```

The kernel body is **pure safe Rust** — shape (rows, cols, dtype) is committed at the type level via const generics, so any host-side mismatch becomes a compile-time error. The `#[aiv_kernel]` attribute rewrites the emitted signature back to raw `*const f32` / `*mut f32` so the launcher toolchain sees the same C ABI; `#[repr(transparent)]` on `GmView`/`GmViewMut` makes this rewrite free at the LLVM IR level.

Compiles via `rustc_codegen_mlir` → MLIR → target-specific code (AscendC, CUDA, GLSL, NKI, AIE).

## Benchmark configurations

| Shape (B, S, D) | Elements | Bytes (f32) | Notes |
|---|---|---|---|
| (1, 64, 128) | 8K | 32 KB | Single query, short context |
| (32, 64, 128) | 262K | 1 MB | Batched queries |
| (1, 128, 128) | 16K | 64 KB | Longer head dim |

## Results

<div id="kernel-results" data-kernel="rope"></div>

*See [Leaderboard](../leaderboard.md) filtered to RoPE for the full filterable view.*
