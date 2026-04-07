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

## Benchmark configurations

| Shape | Elements | Bytes (f32) | Notes |
|---|---|---|---|
| (1, 1024) | 1K | 4 KB | L1-resident, tests dispatch overhead |
| (64, 1024) | 64K | 256 KB | L2-resident, typical batch |
| (64, 4096) | 256K | 1 MB | Bandwidth-bound regime |

## Results

<div id="kernel-results" data-kernel="softmax"></div>

*See [Leaderboard](../leaderboard.md) filtered to Softmax for the full filterable view.*
