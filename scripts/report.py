#!/usr/bin/env python3
"""
Unified kernel benchmark report.

Aggregates CSV results from all kernel benchmark pairs (Rust vs C++) and
prints comparison tables with median/min/max statistics plus speedup ratios.

Input CSV format (no header):
  BENCH,<size>,<kernel>,<run>,<time_ms>

Usage:
  python3 report.py results/combined.csv
"""

import sys
import csv
from collections import defaultdict


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


# Kernel pairs: (rust_name, cpp_name, display_name)
KERNEL_PAIRS = [
    ("rust_vector", "cpp_vector", "vec_add (f16)"),
    ("rust_vector", "cpp_naive", "softmax (vs naive)"),
    ("rust_vector", "cpp_opt", "softmax (vs opt)"),
    ("rust_matmul", "cpp_matmul", "matmul (f16→f32)"),
]


def main():
    if len(sys.argv) < 2:
        print("Usage: report.py <combined.csv>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]

    # Collect times: {(size, kernel): [time_ms, ...]}
    data = defaultdict(list)
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 5 or row[0] != "BENCH":
                continue
            size = row[1]
            kernel = row[2]
            time_ms = float(row[4])
            data[(size, kernel)].append(time_ms)

    if not data:
        print("No benchmark data found.", file=sys.stderr)
        sys.exit(1)

    # Compute stats
    stats = {}
    for key, times in data.items():
        stats[key] = {
            "min": min(times),
            "max": max(times),
            "mean": sum(times) / len(times),
            "median": median(times),
            "n": len(times),
        }

    # Discover all kernels and sizes
    all_kernels = sorted({k[1] for k in data})
    all_sizes = []
    seen = set()
    for k in data:
        if k[0] not in seen:
            seen.add(k[0])
            all_sizes.append(k[0])

    # --- Per-kernel summary table ---
    print()
    print("=" * 72)
    print("  KERNEL BENCHMARK RESULTS")
    print("=" * 72)
    print()

    # Group by kernel for a simple listing
    for kernel in all_kernels:
        sizes_for_kernel = [s for s in all_sizes if (s, kernel) in stats]
        if not sizes_for_kernel:
            continue
        print(f"  {kernel}:")
        for size in sizes_for_kernel:
            s = stats[(size, kernel)]
            print(
                f"    size={size:>12s}  "
                f"median={s['median']:>10.4f}ms  "
                f"min={s['min']:>10.4f}ms  "
                f"max={s['max']:>10.4f}ms  "
                f"(n={s['n']})"
            )
        print()

    # --- Comparison table ---
    print("=" * 72)
    print("  RUST vs C++ COMPARISON")
    print("=" * 72)
    print()

    header = f"  {'Kernel':<25s} {'Size':>12s} {'Rust(ms)':>10s} {'C++(ms)':>10s} {'Speedup':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for rust_name, cpp_name, display_name in KERNEL_PAIRS:
        sizes_rust = {s for s in all_sizes if (s, rust_name) in stats}
        sizes_cpp = {s for s in all_sizes if (s, cpp_name) in stats}
        common_sizes = sorted(sizes_rust & sizes_cpp)

        for size in common_sizes:
            rust_med = stats[(size, rust_name)]["median"]
            cpp_med = stats[(size, cpp_name)]["median"]
            if cpp_med > 0:
                speedup = cpp_med / rust_med
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "-"
            print(
                f"  {display_name:<25s} {size:>12s} "
                f"{rust_med:>10.4f} {cpp_med:>10.4f} {speedup_str:>8s}"
            )

    print()

    # --- Detailed statistics ---
    print("=" * 72)
    print("  DETAILED STATISTICS")
    print("=" * 72)
    print()
    print(
        f"  {'Kernel':<15s} {'Size':>12s} {'N':>3s} "
        f"{'Min':>10s} {'Median':>10s} {'Mean':>10s} {'Max':>10s}"
    )
    print("  " + "-" * 72)
    for size in all_sizes:
        for kernel in all_kernels:
            key = (size, kernel)
            if key in stats:
                s = stats[key]
                print(
                    f"  {kernel:<15s} {size:>12s} {s['n']:>3d} "
                    f"{s['min']:>10.4f} {s['median']:>10.4f} "
                    f"{s['mean']:>10.4f} {s['max']:>10.4f}"
                )
    print()


if __name__ == "__main__":
    main()
