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

Cross-entropy using the tile API:

```rust
#[ascend_std::aiv_kernel]
pub unsafe fn cross_entropy_tile(
    logits_ptr: *const f32,
    targets_ptr: *const u32,
    loss_ptr: *mut f32,
) {
    const N: usize = 32;
    const V: usize = 32000;

    let logits: Tile<N, V, f32> = tile_load_f32::<N, V>(logits_ptr);
    let losses: Tile<N, 1, f32> = tile_cross_entropy_f32::<N, V>(logits, targets_ptr);
    tile_store_f32::<N, 1>(loss_ptr, losses);
}
```

## Benchmark configurations

| Shape (N, V) | Elements | Bytes (f32) | Notes |
|---|---|---|---|
| (32, 32000) | 1M | 4 MB | LLaMA-2 vocab, small batch |
| (128, 32000) | 4M | 16 MB | Larger batch |
| (32, 50257) | 1.6M | 6.4 MB | GPT-2 vocab |

## Results

<div id="kernel-results" data-kernel="cross_entropy"></div>

*See [Leaderboard](../leaderboard.md) filtered to Cross-Entropy for the full filterable view.*
