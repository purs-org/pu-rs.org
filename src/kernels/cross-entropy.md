# Cross-Entropy Loss

**Category**: Loss Function | **Complexity**: O(N*V) reduction | **Memory**: 2 passes (max + sum-exp)

## Algorithm

Cross-entropy loss is the standard training objective for classification and language modeling:

```
loss[i] = -logits[i, target[i]] + log(sum(exp(logits[i, :])))
```

Numerically stable version (log-sum-exp trick):

```
m = max(logits[i, :])
loss[i] = -(logits[i, target[i]] - m) + log(sum(exp(logits[i, :] - m)))
```

This kernel is compute-heavy for large vocabularies (V=32000+) due to the row-wise exp and reduction. It combines softmax-like reduction with an index gather.

## ascend-rs Kernel Source

Cross-entropy using the tile API — safe entry form (compiles to PTO-MLIR for M-pipe, or to CUDA/SPIR-V/NKI/AIE):

```rust
use ascend_std::tile::{GmView, GmViewMut, safe, tile_load_view_f32, tile_store_view_f32};

#[ascend_std::aiv_kernel]
pub fn tile_cross_entropy(
    logits:  GmView<'_, 32, 32000, f32>,  // (N, V)
    targets: *const u32,                  // (N,) target class ids — integer gather
    loss:    GmViewMut<'_, 32, 1, f32>,   // (N, 1) per-row loss
) {
    let x = tile_load_view_f32(&logits);
    let y = safe::tile_cross_entropy_f32(x, targets);
    tile_store_view_f32(&loss, y);
}
```

The kernel body is **pure safe Rust** — logits and loss shapes (and their shared `N`) are committed at the type level via const generics, so any host-side mismatch becomes a compile-time error. The targets pointer remains a raw `*const u32` since target ids are not a tile. The `#[aiv_kernel]` attribute rewrites the emitted signature back to raw `*const f32` / `*mut f32` for the tile params so the launcher toolchain sees the same C ABI; `#[repr(transparent)]` on `GmView`/`GmViewMut` makes this rewrite free at the LLVM IR level.

Compiles via `rustc_codegen_mlir` → MLIR → target-specific code (AscendC, CUDA, GLSL, NKI, AIE).

## Benchmark configurations

| Shape (N, V) | Elements | Bytes (f32) | Notes |
|---|---|---|---|
| (32, 32000) | 1M | 4 MB | LLaMA-2 vocab, small batch |
| (128, 32000) | 4M | 16 MB | Larger batch |
| (32, 50257) | 1.6M | 6.4 MB | GPT-2 vocab |

## Results

<div id="kernel-results" data-kernel="cross-entropy"></div>

*See [Leaderboard](../leaderboard.md) filtered to Cross-Entropy for the full filterable view.*
