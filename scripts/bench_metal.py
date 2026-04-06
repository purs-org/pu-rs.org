#!/usr/bin/env python3
"""
bench_metal.py — Systematic benchmark runner for ascend-rs Metal kernels.

Outputs CSV in pu-rs.org standard format to stdout or a file.
Requires: ASCEND_METAL_KERNELS=1 and ascend_metal_kernels installed
          (build: cd ascend-rs-priv/crates/ascend_metal_py && maturin develop --release)

Usage:
  ASCEND_METAL_KERNELS=1 python3 scripts/bench_metal.py --device apple-m2-max-38
  ASCEND_METAL_KERNELS=1 python3 scripts/bench_metal.py --device apple-m4-max-40 -o submissions/apple-m4-max.csv

Note: Use the same Python that maturin installed into. Check with:
  python3 -c "import ascend_metal_kernels"
"""

import argparse
import os
import sys
import time
import subprocess
import numpy as np

os.environ.setdefault("ASCEND_METAL_KERNELS", "1")
import ascend_metal_kernels as _amk


# ── Benchmark configurations ──────────────────────────────────────────────────
# Each entry: (kernel_id, dtype, shape_str, batch_size, runner_fn_name)

CONFIGS = [
    # Softmax
    ("softmax", "f32", "[1, 1024]",    1),
    ("softmax", "f32", "[16, 1024]",   1),
    ("softmax", "f32", "[64, 1024]",   1),
    ("softmax", "f32", "[256, 1024]",  1),
    ("softmax", "f32", "[1024, 1024]", 1),
    ("softmax", "f32", "[64, 4096]",   1),
    ("softmax", "f32", "[1, 4096]",    1),
    # LayerNorm
    ("layernorm", "f32", "[1, 768]",    1),
    ("layernorm", "f32", "[64, 768]",   1),
    ("layernorm", "f32", "[256, 768]",  1),
    ("layernorm", "f32", "[1024, 768]", 1),
    ("layernorm", "f32", "[1, 512]",    1),
    ("layernorm", "f32", "[64, 512]",   1),
    # Dilated Conv1D + ReLU
    ("conv1d-dilated", "f32", "[2, 50, 512]",  1),
    ("conv1d-dilated", "f32", "[8, 100, 512]", 1),
    ("conv1d-dilated", "f32", "[2, 400, 512]", 1),
]

WARMUP = 50
ITERS = 500  # amortized via MetalStream


def parse_shape(s):
    """Parse shape string like '[64, 1024]' into a tuple."""
    return tuple(int(x.strip()) for x in s.strip("[]").split(","))


def bench_softmax(shape):
    """Benchmark softmax using MetalStream (amortized overhead)."""
    x = np.random.randn(*shape).astype(np.float32)
    # Warmup
    for _ in range(WARMUP):
        _amk.softmax_batched(x)
    # Amortized: N dispatches in one stream
    times = []
    for _ in range(20):
        stream = _amk.MetalStream()
        for _ in range(ITERS):
            stream.softmax(x)
        t0 = time.perf_counter_ns()
        stream.flush()
        elapsed = time.perf_counter_ns() - t0
        # But we need total time including encoding
        times.append(elapsed / ITERS / 1000)  # ns -> us per dispatch

    # Also measure total (encode + flush) time
    total_times = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        stream = _amk.MetalStream()
        for _ in range(ITERS):
            stream.softmax(x)
        stream.flush()
        elapsed = time.perf_counter_ns() - t0
        total_times.append(elapsed / ITERS / 1000)

    return total_times


def bench_layernorm(shape):
    """Benchmark layernorm using MetalStream."""
    x = np.random.randn(*shape).astype(np.float32)
    n = shape[-1]
    w = np.ones(n, dtype=np.float32)
    b = np.zeros(n, dtype=np.float32)
    # Warmup
    for _ in range(WARMUP):
        _amk.layernorm(x, w, b)
    # Amortized
    total_times = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        stream = _amk.MetalStream()
        for _ in range(ITERS):
            stream.layernorm(x, w, b)
        stream.flush()
        elapsed = time.perf_counter_ns() - t0
        total_times.append(elapsed / ITERS / 1000)
    return total_times


def bench_conv1d_dilated(shape):
    """Benchmark dilated conv1d + ReLU (per-call, no stream yet)."""
    B, T, C = shape
    x = np.random.randn(B, T, C).astype(np.float32)
    w = np.random.randn(C, 3 * C).astype(np.float32) * 0.01
    b = np.zeros(C, dtype=np.float32)
    # Warmup
    for _ in range(WARMUP):
        _amk.dilated_conv1d_relu(x, w, b, 1)
    # Measure
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        _amk.dilated_conv1d_relu(x, w, b, 1)
        times.append((time.perf_counter_ns() - t0) / 1000)
    return times


RUNNERS = {
    "softmax": bench_softmax,
    "layernorm": bench_layernorm,
    "conv1d-dilated": bench_conv1d_dilated,
}


def get_git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


def get_toolchain():
    return f"ascend-rs {_amk.__version__} ({_amk.__backend__})"


def main():
    parser = argparse.ArgumentParser(description="Benchmark ascend-rs Metal kernels")
    parser.add_argument("--device", required=True, help="Device ID (e.g. apple-m2-max-38)")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path (default: stdout)")
    parser.add_argument("--submitter", default="bench_metal.py", help="Submitter name")
    args = parser.parse_args()

    git_sha = get_git_sha()
    toolchain = get_toolchain()
    driver = f"Metal ({_amk.__backend__})"

    lines = ["device_id,kernel_id,dtype,input_shape,batch_size,impl_lang,"
             "latency_us,driver_version,toolchain,git_sha,submitter"]

    print(f"Benchmarking {len(CONFIGS)} configs on {args.device}...", file=sys.stderr)
    print(f"Toolchain: {toolchain}", file=sys.stderr)
    print(f"Warmup: {WARMUP}, Iters: {ITERS}", file=sys.stderr)
    print(file=sys.stderr)

    for kernel_id, dtype, shape_str, batch_size in CONFIGS:
        shape = parse_shape(shape_str)
        runner = RUNNERS.get(kernel_id)
        if runner is None:
            print(f"  SKIP {kernel_id} (no runner)", file=sys.stderr)
            continue

        print(f"  {kernel_id:20s} {shape_str:20s} {dtype} ... ", end="", file=sys.stderr)
        try:
            latencies = runner(shape)
            median_us = sorted(latencies)[len(latencies) // 2]
            print(f"{median_us:.1f} us", file=sys.stderr)

            # Emit one row per run for full data
            for lat in latencies:
                lines.append(
                    f"{args.device},{kernel_id},{dtype},\"{shape_str}\","
                    f"{batch_size},metal,{lat:.2f},{driver},{toolchain},"
                    f"{git_sha},{args.submitter}"
                )
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)

    output = "\n".join(lines) + "\n"
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"\nWritten to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
