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

Embedding using the tile API:

```rust
#[ascend_std::aiv_kernel]
pub unsafe fn embedding_tile(
    weight_ptr: *const f32,
    indices_ptr: *const u32,
    output: *mut f32,
) {
    const V: usize = 32000;
    const D: usize = 128;
    const N: usize = 32;

    let w: Tile<V, D, f32> = tile_load_f32::<V, D>(weight_ptr);
    let emb: Tile<N, D, f32> = tile_embedding_f32::<V, D, N>(w, indices_ptr);
    tile_store_f32::<N, D>(output, emb);
}
```

## Benchmark configurations

| Shape (N, V, D) | Output Elements | Bytes (f32) | Notes |
|---|---|---|---|
| (32, 32000, 128) | 4K | 16 KB | LLaMA-2 vocab, small dim |
| (128, 32000, 128) | 16K | 64 KB | Larger batch |
| (32, 32000, 4096) | 131K | 512 KB | Full hidden dim |

## Results

<div id="kernel-results" data-kernel="embedding"></div>

*See [Leaderboard](../leaderboard.md) filtered to Embedding for the full filterable view.*
