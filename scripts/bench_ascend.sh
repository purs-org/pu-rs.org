#!/usr/bin/env bash
# =============================================================================
# bench_ascend.sh — Ascend NPU kernel benchmark for pu-rs.org
# =============================================================================
#
# Runs kernel benchmarks on Huawei Ascend NPUs (910B/910C) using pre-built
# binaries from the ascend-rs project. Outputs CSV in pu-rs.org format.
#
# Prerequisites:
#   - CANN SDK installed (source setenv.bash)
#   - ascend-rs repo cloned and built (cargo build --release)
#   - Or: pre-built benchmark binaries in scripts/ascend/bin/
#
# Usage:
#   bash scripts/bench_ascend.sh --device huawei-910b [--ascend-rs /path/to/ascend-rs]
#   bash scripts/bench_ascend.sh --device huawei-910c --only softmax
#
# Output: submissions/<device>.csv in pu-rs.org format

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEVICE_ID=""
ASCEND_RS=""
ONLY_KERNEL=""
NPU_DEVICE=0
RUNS=20

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
    echo "Usage: $0 --device <device-id> [--ascend-rs /path/to/ascend-rs] [--only <kernel>]"
    echo ""
    echo "Device IDs: huawei-910b, huawei-910c"
    echo "Kernels: softmax, vec_add, matmul, layernorm"
    exit 1
fi

# Auto-source CANN environment
if [ -z "$ASCEND_HOME_PATH" ]; then
    for SETENV in \
        /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash \
        /usr/local/Ascend/cann-8.5.0/bin/setenv.bash; do
        if [ -f "$SETENV" ]; then
            echo "Sourcing CANN: $SETENV"
            source "$SETENV"
            break
        fi
    done
fi

if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "ERROR: CANN SDK not found. Install it or set ASCEND_HOME_PATH."
    exit 1
fi

export ACLRS_RUN_MODE="${ACLRS_RUN_MODE:-npu}"

# Find ascend-rs repo (for building/running benchmarks)
if [ -z "$ASCEND_RS" ]; then
    # Common locations
    for candidate in \
        "$HOME/ascend-rs" \
        "$HOME/ascend-rs-priv" \
        "/data/ascend-rs" \
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
    echo "  Clone it: git clone https://github.com/ascend-rs/ascend-rs.git"
    echo "  Or specify: $0 --device $DEVICE_ID --ascend-rs /path/to/ascend-rs"
    exit 1
fi

ASCEND_RS="$(cd "$ASCEND_RS" && pwd)"
echo "========================================"
echo " pu-rs.org Ascend NPU Benchmark"
echo "========================================"
echo "Device ID:    $DEVICE_ID"
echo "NPU Device:   $NPU_DEVICE"
echo "ascend-rs:    $ASCEND_RS"
echo "CANN:         $ASCEND_HOME_PATH"
echo "Runs:         $RUNS"
echo ""

OUT_DIR="$ROOT_DIR/submissions"
mkdir -p "$OUT_DIR"
LEGACY_DIR=$(mktemp -d)

should_run() { [ -z "$ONLY_KERNEL" ] || [ "$ONLY_KERNEL" = "$1" ]; }

run_bench() {
    local name="$1"
    local bin="$2"
    local csv="$3"
    echo "--- Running $name ---"
    if [ -x "$bin" ]; then
        "$bin" --csv "$csv" 2>&1
    else
        echo "SKIP: $bin not found (build with: cargo build --release -p $(basename $bin))"
    fi
    echo ""
}

# Build codegen backend
echo "--- Building rustc_codegen_mlir ---"
(cd "$ASCEND_RS" && cargo build -p rustc_codegen_mlir --release 2>&1) || echo "WARN: codegen build failed"
export LD_LIBRARY_PATH="$ASCEND_RS/target/release:${LD_LIBRARY_PATH:-}"

# Run each kernel benchmark
for kernel in softmax vec_add matmul layernorm; do
    should_run "$kernel" || continue

    for lang in rs cpp; do
        crate="bench_${kernel}_${lang}"
        manifest="$ASCEND_RS/examples/$crate/Cargo.toml"
        bin="$ASCEND_RS/examples/$crate/target/release/$crate"

        if [ -f "$manifest" ]; then
            echo "--- Building $crate ---"
            (cd "$ASCEND_RS" && cargo build --release --manifest-path "$manifest" 2>&1) || echo "WARN: build failed"
        fi

        run_bench "$kernel ($lang)" "$bin" "$LEGACY_DIR/${kernel}_${lang}.csv"
    done
done

# Convert legacy BENCH format to pu-rs.org CSV
echo "--- Converting results to pu-rs.org format ---"
python3 - "$DEVICE_ID" "$LEGACY_DIR" "$OUT_DIR/${DEVICE_ID}.csv" << 'PYEOF'
import csv, sys, os, glob, statistics
from collections import defaultdict

device_id = sys.argv[1]
legacy_dir = sys.argv[2]
out_path = sys.argv[3]

groups = defaultdict(list)
for csvfile in glob.glob(os.path.join(legacy_dir, "*.csv")):
    with open(csvfile) as f:
        for row in csv.reader(f):
            if len(row) >= 5 and row[0] == "BENCH":
                size, kernel, run, time_ms = row[1], row[2], row[3], float(row[4])
                groups[(kernel, size)].append(time_ms * 1000)  # ms -> us

with open(out_path, "w") as f:
    f.write("device_id,kernel_id,dtype,input_shape,batch_size,impl_lang,latency_us,"
            "driver_version,toolchain,git_sha,submitter\n")
    for (kernel, size), latencies in sorted(groups.items()):
        # Map kernel names to pu-rs.org IDs
        kid = kernel.replace("rust_vector", "softmax").replace("cpp_vector", "vec-add") \
                     .replace("cpp_naive", "softmax").replace("cpp_opt", "softmax") \
                     .replace("rust_matmul", "matmul").replace("cpp_matmul", "matmul") \
                     .replace("rust_layernorm", "layernorm").replace("cpp_layernorm", "layernorm")
        lang = "rust" if "rust" in kernel else "cpp"
        for lat in latencies:
            f.write(f'{device_id},{kid},f32,"[1, {size}]",1,{lang},{lat:.2f},'
                    f'CANN,ascend-rs,,bench_ascend.sh\n')

print(f"Written {sum(len(v) for v in groups.values())} results to {out_path}")
PYEOF

# Also generate report
echo ""
echo "--- Combined Report ---"
> "$LEGACY_DIR/combined.csv"
for f in "$LEGACY_DIR"/*.csv; do
    [ "$f" != "$LEGACY_DIR/combined.csv" ] && [ -f "$f" ] && cat "$f" >> "$LEGACY_DIR/combined.csv"
done

if [ -f "$ROOT_DIR/scripts/report.py" ]; then
    python3 "$ROOT_DIR/scripts/report.py" "$LEGACY_DIR/combined.csv"
fi

rm -rf "$LEGACY_DIR"

echo ""
echo "Results saved to: $OUT_DIR/${DEVICE_ID}.csv"
echo "To update the site: bash scripts/build_site.sh"
