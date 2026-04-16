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

RoPE using the tile API:

```rust
#[ascend_std::aiv_kernel]
pub unsafe fn rope_tile(input: *const f32, output: *mut f32) {
    const S: usize = 1;
    const D: usize = 128;

    let x: Tile<S, D, f32> = tile_load_f32::<S, D>(input);
    let y: Tile<S, D, f32> = tile_rope_f32::<S, D>(x, 0);
    tile_store_f32::<S, D>(output, y);
}
```

## Benchmark configurations

| Shape (B, S, D) | Elements | Bytes (f32) | Notes |
|---|---|---|---|
| (1, 64, 128) | 8K | 32 KB | Single query, short context |
| (32, 64, 128) | 262K | 1 MB | Batched queries |
| (1, 128, 128) | 16K | 64 KB | Longer head dim |

## Results

<div id="kernel-results" data-kernel="rope"></div>

*See [Leaderboard](../leaderboard.md) filtered to RoPE for the full filterable view.*
