# Softmax

**Category**: Activation | **Complexity**: O(N) per row | **Memory**: 2 passes over input

## Algorithm

The online 2-pass softmax (Milakov & Gimelshein 2018):

**Pass 1** (single traversal): Maintain running `(max, sum)` pair per thread. When a new maximum is found, rescale the accumulated sum:
```
sum_new = sum_old * exp(max_old - max_new) + exp(x - max_new)
```

**Pass 2**: Write `exp(x - global_max) / global_sum` per element.

This is 33% less memory traffic than the naive 3-pass algorithm (max, exp+sum, normalize).

## ascend-rs Kernel Source

Softmax in ascend-rs uses the buffer API for element-wise backends and the tile API for matrix-oriented backends:

**Scalar kernel** (f32, benchmarked implementation):
```rust
#[ascend_std::aiv_kernel]
pub unsafe fn softmax(input: *const f32, output: *mut f32, len: *const u32) {
    let n = *len as usize;

    // Step 1: Find max for numerical stability
    let mut max_val = *input;
    let mut i = 1usize;
    loop {
        if i >= n { break; }
        let val = *input.wrapping_add(i);
        if val > max_val { max_val = val; }
        i += 1;
    }

    // Step 2: exp(x - max) and accumulate sum
    let mut sum: f32 = 0.0;
    i = 0;
    loop {
        if i >= n { break; }
        let exp_val = (*input.wrapping_add(i) - max_val).exp();
        *output.wrapping_add(i) = exp_val;
        sum += exp_val;
        i += 1;
    }

    // Step 3: Normalize
    i = 0;
    loop {
        if i >= n { break; }
        *output.wrapping_add(i) = *output.wrapping_add(i) / sum;
        i += 1;
    }
}
```

**Tile API** (compiles to PTO-MLIR for M-pipe, or to CUDA/SPIR-V/NKI/AIE):
```rust
use ascend_std::tile::*;

let src: Tile<1, 1024, f32> = tile_load_f32(&input);
let result = tile_softmax_f32(src);
tile_store_f32(&mut output, result);
```

These Rust kernels compile via `rustc_codegen_mlir` → MLIR → target-specific code (AscendC, CUDA, GLSL, NKI, AIE).

## Benchmark configurations

| Shape | Elements | Bytes (f32) | Notes |
|---|---|---|---|
| (1, 1024) | 1K | 4 KB | L1-resident, tests dispatch overhead |
| (64, 1024) | 64K | 256 KB | L2-resident, typical batch |
| (64, 4096) | 256K | 1 MB | Bandwidth-bound regime |

## Results

<div id="kernel-results" data-kernel="softmax"></div>

*See [Leaderboard](../leaderboard.md) filtered to Softmax for the full filterable view.*
