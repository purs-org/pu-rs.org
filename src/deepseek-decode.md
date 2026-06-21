# DeepSeek Decode Throughput (Cross-Vendor)

Per-kernel latency (the rest of this site) is the microbenchmark view. This
page reports the complementary **end-to-end decode throughput** for a full
DeepSeek-R1-Distill-Qwen-1.5B decode step, measured on five production
accelerators from the *same* 13-kernel Rust source emitted through
[tile-rs](https://ascend-rs.org/)'s per-vendor MLIR backends.

## Headline

| Rank | Device                          | Backend             | Emitted language | Decode tok/s |
|------|---------------------------------|---------------------|------------------|-------------:|
| 1    | Huawei Ascend 910B2             | `mlir_to_cpp`+`mlir_to_pto` | AscendC C++ + PTO-MLIR | **168.9** |
| 2    | Google TPU v2-8 (Colab)         | `mlir_to_tpu`       | Pallas           | **162.9** |
| 3    | Apple M2 Max                    | `mlir_to_msl`       | Metal            | **91.7**  |
| 4    | NVIDIA T4 (Colab)               | `mlir_to_gpu`       | CUDA             | **53.7**  |
| 5    | AWS Trainium1 (`trn1.2xlarge`)  | `mlir_to_nki`       | NKI Python       | **12.2**  |
| –    | CPU reference                   | –                   | plain Rust       | 3.7       |

All five numbers come from identical Rust kernel sources. The MLIR → vendor
backend is the only thing that changes between rows. 168.9 tok/s on 910B2 is
2.47× the aclnn-only baseline and 45.6× the CPU reference.

## Why this complements the kernel leaderboard

The [kernel leaderboard](leaderboard.md) tells you which chip runs a given
softmax or GEMM fastest in isolation. Decode tok/s tells you what a real
inference workload actually achieves once those kernels are composed with
host-side launch overhead, KV cache traffic, and HBM pressure. A chip can win
per-kernel and still lose on decode (the Trainium row is the clearest
example — strong per-op latency, 9.5% bandwidth utilisation end-to-end).

## Notes per device

- **Ascend 910B2** — the `+pto` half of the joint path contributes the four
  decode matmul shapes (1.75×–2.98× vs aclnn). RMSNorm stays on CPU; every
  other op is on-NPU. See the main
  [tile-rs blog ch10](https://ascend-rs.org/ch10-deepseek-benchmark.html)
  for the per-kernel breakdown.
- **TPU v2-8** — Colab-visible Pallas on a 4-chip v2 pod; the emitted kernel
  set is `rms_norm`, `matvec_f16`, and an attention fusion.
- **Apple M2 Max** — emitted Metal beats Apple's hand-tuned MLX on decode at
  this model size. Measured via `deepseek_metal`.
- **NVIDIA T4** — Colab Tesla T4; same three kernels as the TPU row, emitted
  in CUDA. 53.7 tok/s is below-roofline (T4 is HBM-bandwidth-starved for this
  model shape).
- **AWS Trainium1** — `trn1.2xlarge`. Six emitted NKI kernels
  (`rms_norm`, three `matvec_f16` variants, `gate_up_silu`). Traced via
  `torch_neuronx.trace` in **two halves** (eager single-NEFF runs at 2.5 tok/s
  — 5× slower — because the single compile unit can't pipeline across the
  whole decode path). Trace time: 461 s; wall time: 5.23 s for 64 decode
  steps.

## Reproduction

Each number is reproducible with the commands documented in the per-device
sections of [tile-rs ch10](https://ascend-rs.org/ch10-deepseek-benchmark.html).
The source kernels live at
[`crates/tile_std/src/tile.rs`](https://github.com/yijunyu/tile-rs/blob/main/crates/tile_std/src/tile.rs)
in the public repo; the per-vendor emitters live in
`crates/rustc_codegen_tile/src/mlir_to_*.rs`.

---

*The per-kernel [leaderboard](leaderboard.md) remains the authoritative view
for isolated-kernel efficiency. Decode throughput is reported here as the
complementary end-to-end metric.*
