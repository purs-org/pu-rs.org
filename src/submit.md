# Submit Results

## CSV format

Create a CSV file named `<device-slug>.csv` with these columns:

```csv
device_id,kernel_id,dtype,input_shape,batch_size,impl_lang,latency_us,driver_version,toolchain,git_sha,submitter
nvidia-h100-sxm,softmax,f32,"[64, 1024]",1,cuda,12.3,CUDA 12.4,nvcc 12.4,abc1234,your-name
```

## Steps

1. Fork the [pu-rs.org repo](https://github.com/purs-org/pu-rs.org)
2. Add your CSV to `submissions/`
3. Open a pull request
4. CI validates format and sanity checks
5. Maintainers review and merge

## Requirements

- Minimum 20 runs per (kernel, shape) pair
- Report median latency
- Include driver version and toolchain
- Device must exist in `db/seed_devices.sql` (or add it in the same PR)

## Running the benchmark

All benchmark scripts live in this repo under `scripts/`.

```bash
# Metal (Apple Silicon)
# Requires: ascend_metal_kernels Python module
#   (build: cd ascend-rs/crates/ascend_metal_py && maturin develop --release)
ASCEND_METAL_KERNELS=1 python3 scripts/bench_metal.py --device apple-m2-max-38
ASCEND_METAL_KERNELS=1 python3 scripts/bench_metal.py --device apple-m4-max-40 -o submissions/m4-max.csv

# Ascend NPU (Huawei 910B/910C)
# Requires: CANN SDK + ascend-rs repo cloned locally
bash scripts/bench_ascend.sh --device huawei-910b
bash scripts/bench_ascend.sh --device huawei-910c --only softmax --ascend-rs ~/ascend-rs
```

### Supported backends

| Backend | Script | Prerequisites |
|---|---|---|
| Apple Metal | `scripts/bench_metal.py` | `ascend_metal_kernels` Python module ([build instructions](https://ascend-rs.org)) |
| Huawei Ascend | `scripts/bench_ascend.sh` | CANN SDK + [ascend-rs](https://ascend-rs.org) repo |
| NVIDIA CUDA | `scripts/bench_cuda.py` | Planned |
| AMD ROCm | `scripts/bench_rocm.py` | Planned |
