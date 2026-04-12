#!/usr/bin/env python3
"""Ingest benchmark CSV into xpu_bench.db.

CSV format (one row per run):
  device_id,kernel_id,dtype,input_shape,batch_size,impl_lang,latency_us,driver_version,toolchain,git_sha,submitter

Or the legacy BENCH format:
  BENCH,<size>,<kernel>,<run>,<time_ms>
"""
import csv
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <db_path> <csv_path>")
        sys.exit(1)

    db_path = sys.argv[1]
    csv_path = sys.argv[2]

    conn = sqlite3.connect(db_path)

    with open(csv_path) as f:
        first_line = f.readline().strip()

    if first_line.startswith("BENCH,"):
        ingest_legacy(conn, csv_path)
    else:
        ingest_standard(conn, csv_path)

    conn.commit()
    conn.close()


def ingest_standard(conn, csv_path):
    """Ingest standard pu-rs.org CSV format."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        # Group runs by (device, kernel, dtype, shape, batch, impl)
        groups = defaultdict(list)
        meta = {}
        for row in reader:
            key = (
                row["device_id"],
                row["kernel_id"],
                row["dtype"],
                row["input_shape"],
                int(row.get("batch_size", 1)),
                row.get("impl_lang", "rust"),
            )
            groups[key].append(float(row["latency_us"]))
            meta[key] = row

    for key, latencies in groups.items():
        device_id, kernel_id, dtype, shape, batch, impl_lang = key
        row = meta[key]

        config_id = get_or_create_config(conn, kernel_id, dtype, shape, batch)

        # Use throughput from CSV if provided, otherwise compute from shape
        throughput = None
        if row.get("throughput_gbps"):
            throughput = float(row["throughput_gbps"])
        else:
            med = statistics.median(latencies)
            throughput = _auto_throughput(kernel_id, shape, med)

        conn.execute("""
            INSERT INTO bench_results
                (device_id, config_id, impl_lang, latency_us, latency_min_us,
                 latency_max_us, latency_p99_us, throughput_gops, num_runs,
                 driver_version, toolchain, git_sha, submitter)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            device_id, config_id, impl_lang,
            statistics.median(latencies),
            min(latencies), max(latencies),
            sorted(latencies)[int(len(latencies) * 0.99)],
            throughput,
            len(latencies),
            row.get("driver_version"),
            row.get("toolchain"),
            row.get("git_sha"),
            row.get("submitter"),
        ))

    print(f"  Ingested {len(groups)} result groups from {csv_path}")


def ingest_legacy(conn, csv_path):
    """Ingest legacy BENCH,<size>,<kernel>,<run>,<time_ms> format."""
    groups = defaultdict(list)
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5 or parts[0] != "BENCH":
                continue
            size, kernel, _run, time_ms = parts[1], parts[2], parts[3], parts[4]
            groups[(kernel, size)].append(float(time_ms) * 1000)  # ms -> us

    # Infer device from filename
    device_id = Path(csv_path).stem.replace("_results", "")

    for (kernel, size), latencies in groups.items():
        config_id = get_or_create_config(conn, kernel, "f32", size, 1)
        conn.execute("""
            INSERT INTO bench_results
                (device_id, config_id, impl_lang, latency_us, latency_min_us,
                 latency_max_us, num_runs, submitter)
            VALUES (?, ?, 'rust', ?, ?, ?, ?, 'auto-ingest')
        """, (
            device_id, config_id,
            statistics.median(latencies),
            min(latencies), max(latencies),
            len(latencies),
        ))

    print(f"  Ingested {len(groups)} legacy results from {csv_path}")


def _auto_throughput(kernel_id, shape, median_us):
    """Compute throughput (GB/s or GFLOPS) from kernel type and shape."""
    import re

    if kernel_id == "attention":
        # Shape: "B=1,H=32,S=1024,D=128" → FLOPs = 4*B*H*S*S*D
        m = re.search(r'B=(\d+).*H=(\d+).*S=(\d+).*D=(\d+)', shape)
        if m:
            B, H, S, D = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            flops = 4.0 * B * H * S * S * D
            return flops / (median_us * 1e-6) / 1e9  # GFLOPS

    if kernel_id == "vq-quantize":
        # Shape: "N=1024,K=512,D=64" → FLOPs = 2*N*K*D (L2 distance matmul)
        m = re.search(r'N=(\d+).*K=(\d+).*D=(\d+)', shape)
        if m:
            N, K, D = int(m.group(1)), int(m.group(2)), int(m.group(3))
            flops = 2.0 * N * K * D
            return flops / (median_us * 1e-6) / 1e9

    if kernel_id == "conv1d":
        # Shape: "B=16,L=4096,C=64,K=3" → FLOPs = 2*B*L*C*C*K
        m = re.search(r'B=(\d+).*L=(\d+).*C=(\d+).*K=(\d+)', shape)
        if m:
            B, L, C, K = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            flops = 2.0 * B * L * C * C * K
            return flops / (median_us * 1e-6) / 1e9

    if kernel_id == "matmul":
        # Shape: "[M, N]" (square) or "[M, K] x [K, N]" (rectangular)
        m = re.match(r'\[(\d+),\s*(\d+)\]\s*x\s*\[(\d+),\s*(\d+)\]', shape)
        if m:
            M, K, K2, N = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            flops = 2.0 * M * K * N
            return flops / (median_us * 1e-6) / 1e9
        m = re.match(r'\[(\d+),\s*(\d+)\]', shape)
        if m:
            M, N = int(m.group(1)), int(m.group(2))
            flops = 2.0 * M * M * N  # square: M=K=N
            return flops / (median_us * 1e-6) / 1e9

    # Default: bandwidth-based (2 * elements * 4 bytes / time)
    m = re.match(r'\[?\s*(\d+)\s*[,x]\s*(\d+)\s*\]?', shape)
    if m:
        elements = int(m.group(1)) * int(m.group(2))
        return 2.0 * elements * 4.0 / (median_us * 1e-6) / 1e9

    return None


def get_or_create_config(conn, kernel_id, dtype, shape, batch):
    row = conn.execute("""
        SELECT config_id FROM bench_configs
        WHERE kernel_id=? AND dtype=? AND input_shape=? AND batch_size=?
    """, (kernel_id, dtype, shape, batch)).fetchone()
    if row:
        return row[0]
    conn.execute("""
        INSERT INTO bench_configs (kernel_id, dtype, input_shape, batch_size)
        VALUES (?, ?, ?, ?)
    """, (kernel_id, dtype, shape, batch))
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


if __name__ == "__main__":
    main()
