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

SiLU using the tile API (single source, compiles to all backends):

```rust
#[ascend_std::aiv_kernel]
pub unsafe fn silu_tile(input: *const f32, output: *mut f32) {
    const R: usize = 1;
    const C: usize = 4096;

    let x: Tile<R, C, f32> = tile_load_f32::<R, C>(input);
    let y: Tile<R, C, f32> = tile_silu_f32::<R, C>(x);
    tile_store_f32::<R, C>(output, y);
}
```

Decomposes to: neg -> exp -> add_scalar(1) -> reciprocal -> mul with original x.

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
