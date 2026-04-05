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

```bash
# Metal (Apple Silicon)
ASCEND_METAL_KERNELS=1 python3 scripts/bench_metal.py --device apple-m2-max-38

# CUDA (NVIDIA)
python3 scripts/bench_cuda.py --device nvidia-h100-sxm

# Ascend (Huawei NPU)
bash benchmarks/kernel_bench/bench.sh
```
