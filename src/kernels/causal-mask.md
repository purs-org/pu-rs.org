# Causal Mask

**Category**: Masking | **Complexity**: O(S^2) elementwise | **Memory**: 1 pass (read+write)

## Algorithm

Causal masking sets the upper triangle of the attention score matrix to negative infinity, preventing tokens from attending to future positions:

```
For i, j in [0..S) x [0..S):
  if j > i:  scores[i,j] = -inf
  else:      scores[i,j] = scores[i,j]
```

Applied between Q@K^T and softmax in autoregressive (decoder) attention:
```
scores = Q @ K^T / sqrt(d)
scores = causal_mask(scores)   <-- this kernel
weights = softmax(scores)
```

This is memory-bandwidth bound (simple conditional copy), but critical for correctness in all decoder-only models (GPT, LLaMA, etc.).

## tile-rs Kernel Source

Causal mask using the tile API — safe entry form:

```rust
use tile_std::tile::{GmView, GmViewMut, safe, tile_load_view_f32, tile_store_view_f32};

#[tile_std::tile_kernel]
pub fn tile_causal_mask(
    input:  GmView<'_, 64, 64, f32>,
    output: GmViewMut<'_, 64, 64, f32>,
) {
    let scores = tile_load_view_f32(&input);
    let masked = safe::tile_causal_mask_f32(scores);
    tile_store_view_f32(&output, masked);
}
```

The kernel body is **pure safe Rust** — shape (rows, cols, dtype) is committed at the type level via const generics, so any host-side mismatch becomes a compile-time error. Square-shape enforcement (rows == cols) is also enforced at the type level. The `#[tile_kernel]` attribute rewrites the emitted signature back to raw `*const f32` / `*mut f32` so the launcher toolchain sees the same C ABI; `#[repr(transparent)]` on `GmView`/`GmViewMut` makes this rewrite free at the LLVM IR level.

**Backend status** (lowered by `rustc_codegen_tile`): Cambricon BANG, Intel Gaudi, Apple Metal, Vulkan SPIR-V (4/9). Ascend AIV / CUDA / AWS NKI / AMD AIE / Google TPU lowerings are **TODO** — on those backends causal masking is currently applied as a buffer-API element-wise compare-and-select rather than a single fused tile op.

## Benchmark configurations

| Shape (S, S) | Elements | Bytes (f32) | Notes |
|---|---|---|---|
| (64, 64) | 4K | 16 KB | Small attention window |
| (128, 128) | 16K | 64 KB | Standard context |
| (256, 256) | 65K | 256 KB | Medium context |
| (512, 512) | 262K | 1 MB | Long context |

## Results

<div id="kernel-results" data-kernel="causal-mask"></div>

*See [Leaderboard](../leaderboard.md) filtered to Causal Mask for the full filterable view.*
