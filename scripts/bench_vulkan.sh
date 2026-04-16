#!/usr/bin/env bash
# bench_vulkan.sh — USB portable GPU benchmark launcher
#
# Usage (from USB stick):
#   ./bench_vulkan.sh
#   BENCH_SUBMITTER=alice ./bench_vulkan.sh
#
# Requires: Linux x86_64 with a Vulkan-capable GPU driver (any discrete GPU
# from NVIDIA, AMD, Intel Arc on Ubuntu 20.04+ / Fedora 35+ / any glibc≥2.31).
# Zero installation needed — Vulkan loader is bundled alongside this script.
#
# Output: results/bench-<hostname>-<date>.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/bench-linux-x86_64"
RESULTS_DIR="${SCRIPT_DIR}/results"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
HOSTNAME_SAFE="$(hostname | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9-')"
CSV_OUT="${RESULTS_DIR}/bench-${HOSTNAME_SAFE}-${TIMESTAMP}.csv"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -f "${BINARY}" ]]; then
    echo "ERROR: benchmark binary not found: ${BINARY}"
    echo "  Download pre-built binary from https://pu-rs.org/bench/"
    echo "  or build from source:"
    echo "    cargo build --release --target x86_64-unknown-linux-musl \\"
    echo "      --manifest-path examples/tile_softmax_vulkan/Cargo.toml"
    exit 1
fi

if [[ ! -x "${BINARY}" ]]; then
    chmod +x "${BINARY}"
fi

mkdir -p "${RESULTS_DIR}"

# ── Vulkan loader: prefer system, fall back to bundled ────────────────────────
BUNDLED_VULKAN="${SCRIPT_DIR}/libvulkan.so.1"
if [[ -f "${BUNDLED_VULKAN}" ]]; then
    export LD_LIBRARY_PATH="${SCRIPT_DIR}:${LD_LIBRARY_PATH:-}"
fi

# ── GPU detection hint ────────────────────────────────────────────────────────
echo "────────────────────────────────────────────────────────"
echo " pu-rs.org USB Benchmark — softmax kernel latency sweep"
echo "────────────────────────────────────────────────────────"
if command -v lspci &>/dev/null; then
    GPU_LINE="$(lspci 2>/dev/null | grep -iE 'vga|display|3d|gpu' | head -3 || true)"
    if [[ -n "${GPU_LINE}" ]]; then
        echo "PCI GPU(s) detected:"
        echo "${GPU_LINE}" | sed 's/^/  /'
    fi
fi
echo

# ── Run benchmark ─────────────────────────────────────────────────────────────
BENCH_SUBMITTER="${BENCH_SUBMITTER:-anonymous}" \
    "${BINARY}" --csv "${CSV_OUT}"

echo
echo "────────────────────────────────────────────────────────"
echo " Results saved to:"
echo "   ${CSV_OUT}"
echo
echo " To submit to pu-rs.org leaderboard:"
echo "   1. Copy the CSV file to a machine with git access"
echo "   2. Place it in: submissions/<device-name>.csv"
echo "   3. Open a pull request at https://github.com/pu-rs-org/results"
echo "────────────────────────────────────────────────────────"
