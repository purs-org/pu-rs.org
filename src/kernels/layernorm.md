# LayerNorm

**Category**: Normalization | **Complexity**: O(N) per row | **Memory**: 3 passes

## Algorithm

3-pass fused: mean, variance, normalize+affine in one workgroup:

1. **Mean**: Parallel sum reduction, divide by N
2. **Variance**: Parallel sum of (x - mean)^2, compute inverse std
3. **Affine**: `gamma * (x - mean) * inv_std + beta`

Uses SIMD group shuffles for warp-level reductions (1 threadgroup barrier instead of 8).

## Why ascend-rs beats MPS by 3x

1. Single-pass kernel vs MPS's separate dispatches
2. No Python/ATen overhead (Rust metal crate -> Metal API directly)
3. Fused command buffer (500 dispatches per commit)
4. No intermediate buffer allocations

## Benchmark configurations

| Shape | Notes |
|---|---|
| (1, 768) | GPT-2 hidden dim, single position |
| (64, 768) | Typical batch |
| (1024, 768) | Large batch |

*See [Leaderboard](../leaderboard.md) filtered to LayerNorm for full results.*
