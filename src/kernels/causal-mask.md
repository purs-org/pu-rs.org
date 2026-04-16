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

## ascend-rs Kernel Source

Causal mask using the tile API:

```rust
#[ascend_std::aiv_kernel]
pub unsafe fn causal_mask_tile(input: *const f32, output: *mut f32) {
    const S: usize = 64;

    let scores: Tile<S, S, f32> = tile_load_f32::<S, S>(input);
    let masked: Tile<S, S, f32> = tile_causal_mask_f32::<S>(scores);
    tile_store_f32::<S, S>(output, masked);
}
```

## Benchmark configurations

| Shape (S, S) | Elements | Bytes (f32) | Notes |
|---|---|---|---|
| (64, 64) | 4K | 16 KB | Small attention window |
| (128, 128) | 16K | 64 KB | Standard context |
| (256, 256) | 65K | 256 KB | Medium context |
| (512, 512) | 262K | 1 MB | Long context |

## Results

<div id="kernel-results" data-kernel="causal_mask"></div>

*See [Leaderboard](../leaderboard.md) filtered to Causal Mask for the full filterable view.*
