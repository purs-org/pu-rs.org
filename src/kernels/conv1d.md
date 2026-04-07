# Dilated Conv1D + ReLU

**Category**: Convolution | **Complexity**: O(B·T·C²·3) | **Fusion**: pad + gather + matmul + ReLU

## Algorithm

A dilated 1D convolution with kernel size 3, fused with bias and ReLU activation. Used in VQ-VAE encoder/decoder ResConv1DBlocks (e.g., SOKE, Jukebox-style models).

The naive implementation requires:
1. **Pad** the input by `dilation` on each side (zero-padding)
2. **Gather** 3 positions per output: `[t-d, t, t+d]`
3. **Concat** along the channel axis -> `(B, T, 3C)`
4. **Linear** projection `(3C -> C)`
5. **ReLU**

This kernel fuses all 5 steps into a single GPU pass, eliminating three intermediate `(B, T, 3C)` buffer allocations and the data shuffles between them.

## Why fusion matters

For an 18-block VQ-VAE encoder/decoder, the unfused version allocates **54 intermediate tensors** per forward pass and reads them back. Fusing into one kernel:

- Eliminates intermediate buffer writes/reads (3x memory bandwidth reduction)
- Keeps activations in registers/L1 cache between stages
- One command buffer dispatch instead of five

## Benchmark configurations

| Shape (B, T, C) | Elements | Notes |
|---|---|---|
| (2, 50, 512)  | 51 K  | Single VQ-VAE block, small batch |
| (8, 100, 512) | 410 K | Mid-sized clip |
| (2, 400, 512) | 410 K | Long sequence |

## Reference implementation

```rust
// templates/dilated_conv1d_relu.metal — see ascend-rs
kernel void dilated_conv1d_relu(
    device const float* input,    // (B, T, C)
    device const float* weight,   // (C, 3*C) row-major
    device const float* bias,     // (C,)
    device       float* output,   // (B, T, C)
    constant DilConvParams* params [[buffer(4)]],
    ...
);
```

Each threadgroup handles one `(b, t)` output position. Threads stride across output channels, gathering 3 input positions and computing the dot product with the weight row.

## Results

<div id="kernel-results" data-kernel="conv1d-dilated"></div>

*See [Leaderboard](../leaderboard.md) filtered to `conv1d-dilated` for the full filterable view.*
