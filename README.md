# pu-rs.org -- Processing Unit Ranking System

**The SPECfp for AI accelerators.** Live at [pu-rs.org](https://pu-rs.org).

FLOPS don't tell the full story. A chip rated at 1000 TFLOPS means nothing if
your softmax kernel only achieves 5% utilization. pu-rs.org measures what
matters: **actual kernel execution time** on real hardware, for the operations
that AI workloads actually run.

## Kernels

We benchmark the kernel primitives that compose every AI model:

| Category | Kernels |
|---|---|
| Activation | Softmax, GELU |
| Normalization | LayerNorm, RMSNorm |
| Linear Algebra | GEMM (MatMul) |
| Attention | Scaled Dot-Product Attention |
| Quantization | VQ-Quantize |
| Convolution | Dilated Conv1D + ReLU |

Each kernel is benchmarked across a sweep of batch sizes and column widths
(1--4096 rows x 1024--4096 columns) in f32 to measure both single-core
latency and multi-core throughput saturation.

## Devices with results

| Device | Type | Backend | Benchmark script |
|---|---|---|---|
| Huawei Ascend 910B | NPU | CANN / ascend-rs | `scripts/bench_ascend.sh` |
| NVIDIA Tesla T4 | GPU | CUDA | `submissions/nvidia-tesla-t4-all.csv` |
| Apple M2 Max (38-core) | GPU | Metal | `scripts/bench_metal.py` |
| Apple M4 | GPU | Metal | `scripts/bench_metal.py` |

## Quick start

### Run a benchmark

```bash
# Ascend NPU (Huawei 910B/910C)
# Requires: CANN SDK + ascend-rs repo
bash scripts/bench_ascend.sh --device huawei-910b --ascend-rs ~/ascend-rs

# Apple Metal (M-series)
# Requires: ascend_metal_kernels Python module
ASCEND_METAL_KERNELS=1 python3 scripts/bench_metal.py --device apple-m4-max-40

# Vulkan (portable, any discrete GPU)
bash scripts/bench_vulkan.sh
```

### Compare results across devices

```bash
# All kernels, all devices
python3 scripts/report.py submissions/*.csv

# Filter by kernel
python3 scripts/report.py --kernel layernorm submissions/*.csv

# Filter by shape
python3 scripts/report.py --shape "[4096, 4096]" submissions/*.csv
```

Example output:
```
  layernorm
  [4096, 4096]:
    huawei-910b (rust)                335.7 us     399.8 GB/s  100.0%  ########
    nvidia-tesla-t4 (cuda)            820.6 us     163.6 GB/s   40.9%  ####
```

### Submit results

1. Fork this repo
2. Run the benchmark script for your hardware
3. Place the CSV in `submissions/<device-name>.csv`
4. Open a pull request -- CI validates format automatically
5. Leaderboard updates on merge

## CSV format

```csv
device_id,kernel_id,dtype,input_shape,batch_size,impl_lang,latency_us,throughput_gbps,driver_version,toolchain
huawei-910b,layernorm,f32,"[4096, 4096]",4096,rust,335.71,399.80,CANN 8.5.0,ascend-rs (tile)
```

See [Submit Results](https://pu-rs.org/submit.html) for full column reference.

## Project structure

```
pu-rs.org/
  submissions/          # Benchmark CSV files (one per device)
  db/
    schema.sql          # SQLite schema for xpu_bench.db
    seed_devices.sql    # Device specs (TDP, MSRP, memory BW)
    seed_kernels.sql    # Kernel definitions and categories
  scripts/
    bench_ascend.sh     # Ascend NPU benchmark runner
    bench_metal.py      # Apple Metal benchmark runner
    bench_vulkan.sh     # Portable Vulkan benchmark launcher
    report.py           # Cross-device comparison report
    ingest_bench_csv.py # CSV -> SQLite ingestion
    export_leaderboard_json.py  # SQLite -> JSON for the site
    build_site.sh       # Full build pipeline (DB + JSON + mdbook)
  src/                  # mdbook source (leaderboard, methodology, kernel docs)
  book/                 # Built site (generated, not committed)
```

## How the site is built

```
submissions/*.csv
    |
    v
ingest_bench_csv.py --> xpu_bench.db --> export_leaderboard_json.py --> leaderboard.json
                                                                            |
                                                                            v
                                                                     mdbook build --> book/
```

Push to `main` triggers a GitHub Actions workflow that runs `scripts/build_site.sh`
and deploys to GitHub Pages at [pu-rs.org](https://pu-rs.org).

## Methodology

- **Warmup**: iterations discarded before measurement
- **Measurement**: median latency across multiple runs (10--500 depending on backend)
- **Kernel-only time**: GPU/NPU execution time, excluding host transfers and framework overhead
- **Throughput**: `(input_bytes + output_bytes) / latency` in GB/s
- **Reproducibility**: all results tagged with driver version, toolchain, and git SHA

Full details at [pu-rs.org/methodology](https://pu-rs.org/methodology.html).

## Adding a new device

1. Add the device to `db/seed_devices.sql` with specs (TDP, MSRP, memory bandwidth)
2. Create a benchmark script or adapt an existing one
3. Run benchmarks and place CSV in `submissions/`
4. Open a PR

## Adding a new kernel

1. Add the kernel definition to `db/seed_kernels.sql`
2. Create a kernel documentation page in `src/kernels/`
3. Add the entry to `src/SUMMARY.md`
4. Implement the kernel in at least one benchmark script
5. Open a PR

## License

MIT
