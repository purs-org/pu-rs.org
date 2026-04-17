# Embedding Lookup

**Category**: Memory Access | **Complexity**: O(N*D) gather | **Memory**: Random access (bandwidth-bound)

## Algorithm

Embedding lookup gathers rows from a (V, D) weight table by token indices:

```
For each token index t[i] in [0..V):
  output[i, :] = weight[t[i], :]
```

This is the first operation in any transformer: tokens (integers) become vectors. It is purely bandwidth-bound with random access patterns, making it a key memory subsystem benchmark.

## ascend-rs Kernel Source

Embedding using the tile API — safe entry form (compiles to PTO-MLIR for M-pipe, or to CUDA/SPIR-V/NKI/AIE):

```rust
use ascend_std::tile::{GmView, GmViewMut, safe, tile_load_view_f32, tile_store_view_f32};

#[ascend_std::aiv_kernel]
pub fn tile_embedding(
    weight:  GmView<'_, 32000, 128, f32>,  // (V, D) codebook
    indices: *const u32,                   // (N,) token ids — integer gather source
    output:  GmViewMut<'_, 32, 128, f32>,  // (N, D) gathered rows
) {
    let w = tile_load_view_f32(&weight);
    let emb = safe::tile_embedding_f32(w, indices);
    tile_store_view_f32(&output, emb);
}
```

The kernel body is **pure safe Rust** — weight table and output shapes are committed at the type level via const generics (`V`, `D`, `N`), so any host-side mismatch becomes a compile-time error. The indices pointer remains a raw `*const u32` since gather indices are not a tile. The `#[aiv_kernel]` attribute rewrites the emitted signature back to raw `*const f32` / `*mut f32` for the tile params so the launcher toolchain sees the same C ABI; `#[repr(transparent)]` on `GmView`/`GmViewMut` makes this rewrite free at the LLVM IR level.

Compiles via `rustc_codegen_mlir` → MLIR → target-specific code (AscendC, CUDA, GLSL, NKI, AIE).

## Benchmark configurations

| Shape (N, V, D) | Output Elements | Bytes (f32) | Notes |
|---|---|---|---|
| (32, 32000, 128) | 4K | 16 KB | LLaMA-2 vocab, small dim |
| (128, 32000, 128) | 16K | 64 KB | Larger batch |
| (32, 32000, 4096) | 131K | 512 KB | Full hidden dim |

## Results

<div id="kernel-results" data-kernel="embedding"></div>

*See [Leaderboard](../leaderboard.md) filtered to Embedding for the full filterable view.*
