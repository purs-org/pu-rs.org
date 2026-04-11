#!/usr/bin/env bash
# =============================================================================
# bench_ascend.sh — Ascend NPU kernel benchmark for pu-rs.org
# =============================================================================
#
# Runs tile-API kernel benchmarks on Huawei Ascend NPUs (910B/910C) using
# ascend-rs. Outputs CSV in pu-rs.org submission format.
#
# Benchmarks: softmax, layernorm, conv1d-dilated
# All kernels use the tile-API (vectorized DMA + SIMD) implementation.
#
# Prerequisites:
#   - CANN SDK installed (source set_env.sh)
#   - ascend-rs repo cloned
#   - Rust nightly toolchain with rustc_codegen_mlir built
#
# Usage:
#   bash scripts/bench_ascend.sh --device huawei-910b --ascend-rs /path/to/ascend-rs
#   bash scripts/bench_ascend.sh --device huawei-910c --only layernorm
#   bash scripts/bench_ascend.sh --device huawei-910b --runs 20
#
# Output: submissions/<device>-tile.csv in pu-rs.org format

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEVICE_ID=""
ASCEND_RS=""
ONLY_KERNEL=""
NPU_DEVICE=0
RUNS=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)     DEVICE_ID="$2"; shift 2 ;;
        --ascend-rs)  ASCEND_RS="$2"; shift 2 ;;
        --only)       ONLY_KERNEL="$2"; shift 2 ;;
        --npu-id)     NPU_DEVICE="$2"; shift 2 ;;
        --runs)       RUNS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$DEVICE_ID" ]; then
    cat <<EOF
Usage: $0 --device <device-id> [--ascend-rs /path/to/ascend-rs] [--only <kernel>]

Device IDs: huawei-910b, huawei-910c
Kernels:    softmax, layernorm, conv1d-dilated

Options:
  --device    Device identifier for the CSV output
  --ascend-rs Path to ascend-rs repository (auto-detected if omitted)
  --only      Run only the specified kernel benchmark
  --npu-id    NPU device index (default: 0)
  --runs      Number of timed iterations per shape (default: 10)

Example:
  $0 --device huawei-910b --ascend-rs ~/ascend-rs
  $0 --device huawei-910c --only layernorm --runs 20
EOF
    exit 1
fi

# ── Auto-source CANN environment ─────────────────────────────────────────────
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    for SETENV in \
        /usr/local/Ascend/cann-8.5.0/set_env.sh \
        /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash \
        /usr/local/Ascend/cann-8.5.0/bin/setenv.bash; do
        if [ -f "$SETENV" ]; then
            echo "Sourcing CANN: $SETENV"
            source "$SETENV"
            break
        fi
    done
fi

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "ERROR: CANN SDK not found. Install it or set ASCEND_HOME_PATH."
    exit 1
fi

export ACLRS_RUN_MODE="${ACLRS_RUN_MODE:-npu}"

# ── Find ascend-rs repo ──────────────────────────────────────────────────────
if [ -z "$ASCEND_RS" ]; then
    for candidate in \
        "$HOME/ascend-rs" \
        "$HOME/ascend-rs-priv" \
        "/data/ascend-rs" \
        "/data/$(whoami)/ascend-rs" \
        "../ascend-rs-priv" \
        "../ascend-rs"; do
        if [ -f "$candidate/Cargo.toml" ]; then
            ASCEND_RS="$candidate"
            break
        fi
    done
fi

if [ -z "$ASCEND_RS" ]; then
    echo "ERROR: ascend-rs repo not found."
    echo "  Specify: $0 --device $DEVICE_ID --ascend-rs /path/to/ascend-rs"
    exit 1
fi

ASCEND_RS="$(cd "$ASCEND_RS" && pwd)"

# ── Detect CANN version ─────────────────────────────────────────────────────
CANN_VERSION="CANN $(basename "$(readlink -f "$ASCEND_HOME_PATH" 2>/dev/null || echo "$ASCEND_HOME_PATH")" | sed 's/cann-//' | sed 's/ascend-toolkit/unknown/')"

echo "========================================"
echo " pu-rs.org Ascend NPU Benchmark"
echo "========================================"
echo "Device ID:    $DEVICE_ID"
echo "NPU Device:   $NPU_DEVICE"
echo "ascend-rs:    $ASCEND_RS"
echo "CANN:         $ASCEND_HOME_PATH"
echo "CANN Version: $CANN_VERSION"
echo "Runs:         $RUNS"
echo ""

OUT_DIR="$ROOT_DIR/submissions"
mkdir -p "$OUT_DIR"
RAW_DIR=$(mktemp -d)
trap "rm -rf $RAW_DIR" EXIT

should_run() { [ -z "$ONLY_KERNEL" ] || [ "$ONLY_KERNEL" = "$1" ]; }

# ── Tile benchmark crate map ─────────────────────────────────────────────────
# Each entry: kernel_id -> crate directory name under examples/
declare -A TILE_CRATES=(
    [softmax]="bench_softmax_tile"
    [layernorm]="bench_layernorm_tile"
    [conv1d-dilated]="bench_conv1d_tile"
)

# ── Build codegen backend ────────────────────────────────────────────────────
echo "--- Building rustc_codegen_mlir ---"
if (cd "$ASCEND_RS" && cargo build -p rustc_codegen_mlir --release 2>&1); then
    echo "  codegen build OK"
else
    echo "WARN: codegen build failed (may already be built)"
fi
export LD_LIBRARY_PATH="$ASCEND_RS/target/release:${LD_LIBRARY_PATH:-}"

# ── Run tile benchmarks ──────────────────────────────────────────────────────
FAILED_KERNELS=()

for kernel_id in softmax layernorm conv1d-dilated; do
    should_run "$kernel_id" || continue

    crate="${TILE_CRATES[$kernel_id]}"
    crate_dir="$ASCEND_RS/examples/$crate"
    manifest="$crate_dir/Cargo.toml"

    if [ ! -f "$manifest" ]; then
        echo "SKIP: $crate (Cargo.toml not found at $manifest)"
        continue
    fi

    echo ""
    echo "--- Building $crate ---"
    if ! (cd "$crate_dir" && cargo build --release 2>&1); then
        echo "ERROR: build failed for $crate"
        FAILED_KERNELS+=("$kernel_id")
        continue
    fi

    # Find binary (in the crate's own target dir, or workspace target)
    bin=""
    for candidate in \
        "$crate_dir/target/release/$crate" \
        "$ASCEND_RS/target/release/$crate"; do
        if [ -x "$candidate" ]; then
            bin="$candidate"
            break
        fi
    done

    if [ -z "$bin" ]; then
        echo "ERROR: binary not found for $crate after build"
        FAILED_KERNELS+=("$kernel_id")
        continue
    fi

    echo "--- Running $crate ---"
    raw_csv="$RAW_DIR/${kernel_id}.csv"
    if "$bin" --csv "$raw_csv" 2>&1; then
        echo "  $kernel_id benchmark complete"
    else
        echo "ERROR: $crate exited with error"
        FAILED_KERNELS+=("$kernel_id")
    fi
done

# ── Convert BENCH output → pu-rs.org CSV ─────────────────────────────────────
echo ""
echo "--- Converting results to pu-rs.org CSV format ---"

OUT_CSV="$OUT_DIR/${DEVICE_ID}-tile.csv"

python3 - "$DEVICE_ID" "$RAW_DIR" "$OUT_CSV" "$CANN_VERSION" << 'PYEOF'
import csv, sys, os, glob, statistics, re
from collections import defaultdict

device_id = sys.argv[1]
raw_dir = sys.argv[2]
out_path = sys.argv[3]
cann_version = sys.argv[4]

# Map kernel names from benchmark output to pu-rs.org kernel_id
def kernel_to_id(kernel_name):
    """Map tile_softmax_1x1024 -> softmax, tile_conv1d_4096 -> conv1d-dilated, etc."""
    if kernel_name.startswith("tile_softmax"):
        return "softmax"
    elif kernel_name.startswith("tile_layernorm"):
        return "layernorm"
    elif kernel_name.startswith("tile_conv1d"):
        return "conv1d-dilated"
    return kernel_name

# Parse BENCH lines from all raw CSVs
# Format: BENCH,<rows>x<cols>,<kernel>,<run>,<time_ms>
groups = defaultdict(list)
for csvfile in sorted(glob.glob(os.path.join(raw_dir, "*.csv"))):
    with open(csvfile) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("BENCH,"):
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            shape_str = parts[1]   # e.g. "64x4096"
            kernel = parts[2]      # e.g. "tile_layernorm_4096"
            run = int(parts[3])
            time_ms = float(parts[4])
            kernel_id = kernel_to_id(kernel)
            groups[(kernel_id, shape_str)].append(time_ms)

if not groups:
    print("WARNING: No benchmark data found in raw CSV files", file=sys.stderr)
    sys.exit(0)

# Compute median latency and throughput for each (kernel, shape)
results = []
for (kernel_id, shape_str), times_ms in sorted(groups.items()):
    rows, cols = map(int, shape_str.split("x"))
    n_elements = rows * cols
    batch_size = rows

    # Median latency
    med_ms = statistics.median(times_ms)
    med_us = med_ms * 1000  # ms -> us

    # Throughput: read + write in bytes / time
    bytes_rw = n_elements * 4 * 2  # f32 read + write
    throughput_gbps = bytes_rw / (med_ms / 1000) / 1e9

    # Format shape as "[rows, cols]"
    input_shape = f"[{rows}, {cols}]"

    results.append({
        "kernel_id": kernel_id,
        "input_shape": input_shape,
        "batch_size": batch_size,
        "latency_us": med_us,
        "throughput_gbps": throughput_gbps,
        "n_runs": len(times_ms),
    })

# Write pu-rs.org format CSV
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "device_id", "kernel_id", "dtype", "input_shape", "batch_size",
        "impl_lang", "latency_us", "throughput_gbps",
        "driver_version", "toolchain",
    ])
    for r in results:
        w.writerow([
            device_id,
            r["kernel_id"],
            "f32",
            r["input_shape"],
            r["batch_size"],
            "rust",
            f'{r["latency_us"]:.2f}',
            f'{r["throughput_gbps"]:.2f}',
            cann_version,
            "ascend-rs (tile)",
        ])

print(f"  Written {len(results)} result groups to {out_path}")
PYEOF

# ── Summary report ───────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Summary"
echo "========================================"

if [ -f "$OUT_CSV" ]; then
    echo ""
    echo "Results by kernel and shape (sorted by throughput):"
    echo ""
    python3 - "$OUT_CSV" << 'PYEOF'
import csv, sys

rows = []
with open(sys.argv[1]) as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

if not rows:
    print("  (no results)")
    sys.exit(0)

# Group by kernel
kernels = {}
for r in rows:
    kid = r["kernel_id"]
    kernels.setdefault(kid, []).append(r)

for kid in sorted(kernels):
    print(f"  {kid}:")
    entries = sorted(kernels[kid], key=lambda x: -float(x["throughput_gbps"]))
    for e in entries:
        lat = float(e["latency_us"])
        tp = float(e["throughput_gbps"])
        print(f"    {e['input_shape']:>16s}  {lat:>8.1f} us  {tp:>8.1f} GB/s")
    print()
PYEOF
fi

if [ ${#FAILED_KERNELS[@]} -gt 0 ]; then
    echo ""
    echo "WARNING: The following kernels failed: ${FAILED_KERNELS[*]}"
fi

echo ""
echo "CSV output:  $OUT_CSV"
echo "To update the site: bash scripts/build_site.sh"
