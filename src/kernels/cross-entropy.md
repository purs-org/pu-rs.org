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

Cross-entropy using the tile API — safe entry form with one `unsafe` block (the targets pointer is an integer gather source, not a tile, so `safe::tile_cross_entropy_f32` is declared `pub unsafe fn`):

```rust
use ascend_std::tile::{GmView, GmViewMut, safe, tile_load_view_f32, tile_store_view_f32};

#[ascend_std::aiv_kernel]
pub fn tile_cross_entropy(
    logits:  GmView<'_, 32, 32000, f32>,  // (N, V)
    targets: *const u32,                  // (N,) target class ids — integer gather
    loss:    GmViewMut<'_, 32, 1, f32>,   // (N, 1) per-row loss
) {
    let x = tile_load_view_f32(&logits);
    // SAFETY: `targets` is a valid *const u32 of length R=32, guaranteed by
    // the launcher. The unsafe wrapper is the only non-safe surface.
    let y = unsafe { safe::tile_cross_entropy_f32(x, targets) };
    tile_store_view_f32(&loss, y);
}
```

Logits and loss shapes (and their shared `N`) are committed at the type level via const generics, so any host-side mismatch becomes a compile-time error. The `#[aiv_kernel]` attribute rewrites the emitted signature back to raw `*const f32` / `*mut f32` for the tile params so the launcher toolchain sees the same C ABI; `#[repr(transparent)]` on `GmView`/`GmViewMut` makes this rewrite free at the LLVM IR level.

**Backend status** (lowered by `rustc_codegen_mlir`): Cambricon BANG, Intel Gaudi. Ascend AIV / CUDA / Apple Metal / Vulkan SPIR-V / AWS NKI / AMD AIE / Google TPU lowerings are **TODO** — this is the narrowest backend coverage of any kernel page, reflecting that cross-entropy is primarily a training-loss primitive.

## Benchmark configurations

| Shape (N, V) | Elements | Bytes (f32) | Notes |
|---|---|---|---|
| (32, 32000) | 1M | 4 MB | LLaMA-2 vocab, small batch |
| (128, 32000) | 4M | 16 MB | Larger batch |
| (32, 50257) | 1.6M | 6.4 MB | GPT-2 vocab |

## Results

<div id="kernel-results" data-kernel="cross-entropy"></div>

*See [Leaderboard](../leaderboard.md) filtered to Cross-Entropy for the full filterable view.*
