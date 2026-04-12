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

**Buffer API** (V-pipe, vector engine):
```rust
#[ascend_std::aiv_kernel]
pub unsafe fn softmax_rows_f16(
    input: *const u16, output: *mut u16,
    row_len: *const u32, num_rows: *const u32,
) {
    let cols = *row_len;
    let rows = *num_rows;
    let buf_in = ascend_std::ascend_buf_alloc(cols);
    let buf_out = ascend_std::ascend_buf_alloc(cols);
    let buf_work = ascend_std::ascend_buf_alloc(cols);
    let buf_rwork = ascend_std::ascend_buf_alloc(cols);

    let mut row = 0u32;
    loop {
        if row >= rows { break; }
        let in_ptr = input.wrapping_add((row * cols) as usize);
        let out_ptr = output.wrapping_add((row * cols) as usize);

        ascend_std::ascend_buf_load_f16(buf_in, in_ptr, cols);
        ascend_std::ascend_pipe_barrier();
        let max_val = ascend_std::ascend_reduce_max_f16(
            buf_work, buf_in, buf_rwork, cols);
        ascend_std::ascend_adds_f16(buf_out, buf_in, -max_val, cols);
        ascend_std::ascend_pipe_barrier();
        ascend_std::ascend_exp_f16(buf_out, buf_out, cols);
        ascend_std::ascend_pipe_barrier();
        let sum_val = ascend_std::ascend_reduce_sum_f16(
            buf_work, buf_out, buf_rwork, cols);
        ascend_std::ascend_muls_f16(buf_out, buf_out, 1.0 / sum_val, cols);
        ascend_std::ascend_pipe_barrier();
        ascend_std::ascend_buf_store_f16(out_ptr, buf_out, cols);
        row += 1;
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
