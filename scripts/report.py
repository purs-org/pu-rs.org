#!/usr/bin/env python3
"""
Cross-device kernel benchmark comparison report.

Reads pu-rs.org submission CSVs and prints comparison tables showing
throughput, latency, and relative performance across devices.

Usage:
  python3 report.py submissions/*.csv
  python3 report.py submissions/huawei-910b-tile.csv submissions/nvidia-tesla-t4-all.csv
  python3 report.py --kernel softmax submissions/*.csv
  python3 report.py --shape "[4096, 4096]" submissions/*.csv
"""

import csv
import sys
import os
import statistics
from collections import defaultdict


def parse_args():
    args = {"files": [], "kernel": None, "shape": None, "sort": "throughput"}
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--kernel" and i + 1 < len(sys.argv):
            args["kernel"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--shape" and i + 1 < len(sys.argv):
            args["shape"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--sort" and i + 1 < len(sys.argv):
            args["sort"] = sys.argv[i + 1]
            i += 2
        else:
            args["files"].append(sys.argv[i])
            i += 1
    return args


def load_csvs(files):
    """Load all CSV files, group by (device, kernel, shape)."""
    groups = defaultdict(list)
    for path in files:
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                device = row.get("device_id", "")
                kernel = row.get("kernel_id", "")
                shape = row.get("input_shape", "")
                batch = row.get("batch_size", "1")
                latency = float(row.get("latency_us", 0))
                lang = row.get("impl_lang", "")
                tp = row.get("throughput_gbps", "")
                if latency > 0:
                    groups[(device, kernel, shape, batch, lang)].append({
                        "latency_us": latency,
                        "throughput_gbps": float(tp) if tp else 0,
                    })
    return groups


def compute_stats(entries):
    """Compute median latency and throughput from a list of entries."""
    latencies = [e["latency_us"] for e in entries]
    throughputs = [e["throughput_gbps"] for e in entries]
    return {
        "latency_us": statistics.median(latencies),
        "throughput_gbps": max(throughputs) if throughputs else 0,
        "n": len(latencies),
    }


def main():
    args = parse_args()
    if not args["files"]:
        print(__doc__)
        sys.exit(1)

    groups = load_csvs(args["files"])
    if not groups:
        print("No benchmark data found.", file=sys.stderr)
        sys.exit(1)

    # Aggregate stats per group
    stats = {}
    for key, entries in groups.items():
        stats[key] = compute_stats(entries)

    # Collect unique values
    all_devices = sorted({k[0] for k in stats})
    all_kernels = sorted({k[1] for k in stats})
    all_shapes = sorted({k[2] for k in stats})

    # Apply filters
    if args["kernel"]:
        all_kernels = [k for k in all_kernels if args["kernel"] in k]
    if args["shape"]:
        all_shapes = [s for s in all_shapes if args["shape"] in s]

    # ── Per-kernel comparison across devices ─────────────────────────────────
    print()
    print("=" * 90)
    print("  CROSS-DEVICE KERNEL COMPARISON")
    print("=" * 90)

    for kernel in all_kernels:
        # Collect all (device, shape) entries for this kernel
        entries = []
        for key, st in stats.items():
            dev, kid, shape, batch, lang = key
            if kid != kernel:
                continue
            if args["shape"] and args["shape"] not in shape:
                continue
            entries.append({
                "device": dev,
                "shape": shape,
                "batch": batch,
                "lang": lang,
                "latency_us": st["latency_us"],
                "throughput_gbps": st["throughput_gbps"],
                "n": st["n"],
            })

        if not entries:
            continue

        print()
        print(f"  {kernel}")
        print(f"  {'-' * 86}")
        print(f"  {'Device':<22s} {'Shape':>14s} {'Lang':>6s} "
              f"{'Latency(us)':>12s} {'GB/s':>10s} {'Runs':>5s}")
        print(f"  {'-' * 86}")

        # Sort by shape then throughput descending
        entries.sort(key=lambda e: (e["shape"], -e["throughput_gbps"]))

        prev_shape = None
        for e in entries:
            if prev_shape and e["shape"] != prev_shape:
                print()  # blank line between shapes
            prev_shape = e["shape"]
            tp_str = f'{e["throughput_gbps"]:.1f}' if e["throughput_gbps"] > 0 else "-"
            print(f'  {e["device"]:<22s} {e["shape"]:>14s} {e["lang"]:>6s} '
                  f'{e["latency_us"]:>12.1f} {tp_str:>10s} {e["n"]:>5d}')

    # ── Head-to-head comparison for common shapes ────────────────────────────
    if len(all_devices) >= 2:
        print()
        print("=" * 90)
        print("  HEAD-TO-HEAD (common shapes)")
        print("=" * 90)

        for kernel in all_kernels:
            # Find shapes that have results from multiple devices
            shape_devices = defaultdict(dict)
            for key, st in stats.items():
                dev, kid, shape, batch, lang = key
                if kid != kernel:
                    continue
                if args["shape"] and args["shape"] not in shape:
                    continue
                label = f"{dev} ({lang})"
                if label not in shape_devices[shape] or st["throughput_gbps"] > shape_devices[shape][label]["throughput_gbps"]:
                    shape_devices[shape][label] = st

            common = {s: devs for s, devs in shape_devices.items() if len(devs) >= 2}
            if not common:
                continue

            print()
            print(f"  {kernel}")
            print(f"  {'-' * 86}")

            for shape in sorted(common):
                devs = common[shape]
                # Sort by throughput descending
                ranked = sorted(devs.items(), key=lambda x: -x[1]["throughput_gbps"])
                best_tp = ranked[0][1]["throughput_gbps"]

                print(f"  {shape}:")
                for label, st in ranked:
                    ratio = (st["throughput_gbps"] / best_tp * 100) if best_tp > 0 else 0
                    bar = "#" * int(ratio / 2)
                    print(f"    {label:<30s} {st['latency_us']:>8.1f} us  "
                          f"{st['throughput_gbps']:>8.1f} GB/s  "
                          f"{ratio:>5.1f}%  {bar}")

    print()


if __name__ == "__main__":
    main()
