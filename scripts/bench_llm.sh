#!/usr/bin/env bash
# =============================================================================
# bench_llm.sh — End-to-end LLM throughput benchmark for pu-rs.org
# =============================================================================
#
# Measures prefill + decode throughput (tokens/sec) for a target model on
# NVIDIA GPU (vLLM), Apple Silicon (llama.cpp Metal), or Ascend NPU (MindIE).
#
# Default model: DeepSeek-R1-Distill-Qwen-1.5B
#
# Outputs a CSV compatible with scripts/ingest_bench_csv.py. The model is
# registered as a pseudo-kernel (kernel_id = slug of model name) and shape
# encodes the workload (prompt_len, gen_len, batch_size). Throughput is
# reported in tokens/sec via the throughput_gbps column (unit repurposed —
# see NOTES below).
#
# Usage:
#   bash scripts/bench_llm.sh --device apple-m2-max-38 --backend llama-cpp
#   bash scripts/bench_llm.sh --device nvidia-tesla-t4 --backend vllm
#   bash scripts/bench_llm.sh --device huawei-910b     --backend mindie
#   bash scripts/bench_llm.sh --device nvidia-tesla-t4 --backend vllm \
#       --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --runs 5
#
# Output: submissions/<device>-llm-<model-slug>.csv
#
# NOTES on schema fit:
#   The pu-rs.org DB is kernel-level. We piggyback on it by encoding the
#   model as a pseudo-kernel. Tokens/sec goes into throughput_gops (the
#   column is typed REAL). Before ingesting, run:
#     python3 -c "import sqlite3; c=sqlite3.connect('db/xpu_bench.db'); \
#       c.execute(\"INSERT OR IGNORE INTO kernels(kernel_id,display_name,category,description) \
#         VALUES('llm-r1qwen15','DeepSeek-R1-Distill-Qwen-1.5B','model', \
#                'End-to-end decode throughput (tokens/sec), bs=1 unless noted')\"); c.commit()"
#
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DEVICE_ID=""
BACKEND=""
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_SLUG="llm-r1qwen15"
RUNS=3
WARMUP=1
QUANT=""            # e.g. "Q4_K_M" for llama.cpp, "w8a8" for vllm
MODEL_PATH=""       # optional local path (GGUF for llama.cpp, HF dir elsewhere)
GPU_ID=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)     DEVICE_ID="$2"; shift 2 ;;
        --backend)    BACKEND="$2"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
        --model-slug) MODEL_SLUG="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --quant)      QUANT="$2"; shift 2 ;;
        --runs)       RUNS="$2"; shift 2 ;;
        --warmup)     WARMUP="$2"; shift 2 ;;
        --gpu-id)     GPU_ID="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,40p' "$0"
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$DEVICE_ID" ] || [ -z "$BACKEND" ]; then
    echo "Usage: $0 --device <id> --backend <llama-cpp|vllm|mindie> [options]" >&2
    echo "Run with --help for full options." >&2
    exit 1
fi

case "$BACKEND" in
    llama-cpp|vllm|mindie) ;;
    *) echo "ERROR: unknown backend '$BACKEND' (want: llama-cpp, vllm, mindie)" >&2; exit 1 ;;
esac

OUT_DIR="$ROOT_DIR/submissions"
mkdir -p "$OUT_DIR"
OUT_CSV="$OUT_DIR/${DEVICE_ID}-${MODEL_SLUG}.csv"
RAW_DIR=$(mktemp -d)
trap "rm -rf $RAW_DIR" EXIT

GIT_SHA=$(cd "$ROOT_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "")
SUBMITTER="${BENCH_SUBMITTER:-$(whoami)@$(hostname)}"

echo "========================================"
echo " pu-rs.org LLM Benchmark"
echo "========================================"
echo "Device:      $DEVICE_ID"
echo "Backend:     $BACKEND"
echo "Model:       $MODEL"
echo "Slug:        $MODEL_SLUG"
echo "Quant:       ${QUANT:-(backend default)}"
echo "Runs:        $RUNS (warmup $WARMUP)"
echo "Output:      $OUT_CSV"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Workload matrix: (tag, prompt_len, gen_len, batch)
# Prefill-dominated: long prompt, 1 token generated
# Decode-dominated:  short prompt, long generation
# ─────────────────────────────────────────────────────────────────────────────
WORKLOADS=(
    "prefill_p512 512 1   1"
    "decode_g512  16  512 1"
    "decode_g128  16  128 1"
    "batch8_g256  16  256 8"
)

# ─────────────────────────────────────────────────────────────────────────────
# Backend: llama.cpp (Apple Metal / CUDA / CPU)
# ─────────────────────────────────────────────────────────────────────────────
run_llama_cpp() {
    command -v llama-bench >/dev/null || {
        echo "ERROR: llama-bench not found. Install llama.cpp:" >&2
        echo "  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp" >&2
        echo "  cmake -B build -DGGML_METAL=ON && cmake --build build --target llama-bench -j" >&2
        exit 1
    }

    local gguf="$MODEL_PATH"
    if [ -z "$gguf" ]; then
        # Default: expect user to pre-download a GGUF
        echo "ERROR: llama-cpp backend requires --model-path /path/to/model.gguf" >&2
        echo "  Download e.g.:" >&2
        echo "    huggingface-cli download bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF \\"
        echo "      DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf --local-dir ~/models" >&2
        exit 1
    fi

    [ -f "$gguf" ] || { echo "ERROR: GGUF not found: $gguf" >&2; exit 1; }

    local driver
    driver=$(llama-bench --version 2>&1 | head -1 | tr -d ',' || echo "llama.cpp")

    # llama-bench runs single-stream only (no parallel batch flag). We emit
    # prefill (-p N -n 0) and decode (-p 0 -n N) measurements; bs=8 is skipped
    # for this backend.
    for wl in "${WORKLOADS[@]}"; do
        read -r tag plen glen bs <<<"$wl"
        if [ "$bs" != "1" ]; then
            echo "--- SKIP $tag (llama-bench has no parallel-batch flag)"
            continue
        fi
        echo "--- $tag (p=$plen g=$glen) ---"
        local raw="$RAW_DIR/${tag}.json"
        # Prefill-dominated workload: -p plen -n 0
        # Decode-dominated workload:  -p 0    -n glen
        local pp_arg="$plen"
        local tg_arg="$glen"
        if [ "$glen" = "1" ]; then tg_arg=0; fi     # pure prefill
        if [ "$plen" -ge 512 ] && [ "$glen" != "1" ]; then : ; fi
        if ! llama-bench -m "$gguf" -p "$pp_arg" -n "$tg_arg" \
                -r "$RUNS" -o json > "$raw" 2>/dev/null; then
            echo "  llama-bench failed for $tag" >&2
            continue
        fi
        # llama-bench JSON: array of {n_prompt,n_gen,avg_ts,stddev_ts,...}.
        # Row with n_gen==0 is prefill tok/s, row with n_prompt==0 is decode tok/s.
        python3 - "$raw" "$tag" "$plen" "$glen" "$bs" >> "$RAW_DIR/parsed.csv" <<'PY'
import json, sys
raw, tag, plen, glen, bs = sys.argv[1:]
data = json.load(open(raw))
want_np = int(plen) if int(glen) == 1 else 0
want_ng = 0 if int(glen) == 1 else int(glen)
for row in data:
    if int(row["n_prompt"]) == want_np and int(row["n_gen"]) == want_ng:
        print(f'{tag},{plen},{glen},{bs},{row["avg_ts"]:.3f},{row.get("stddev_ts",0):.3f}')
        break
PY
    done

    DRIVER_VERSION="$driver"
    TOOLCHAIN="llama.cpp (Metal)"
}

# ─────────────────────────────────────────────────────────────────────────────
# Backend: vLLM (NVIDIA CUDA)
# ─────────────────────────────────────────────────────────────────────────────
run_vllm() {
    python3 -c "import vllm" 2>/dev/null || {
        echo "ERROR: vllm not importable. Install:" >&2
        echo "  pip install vllm" >&2
        exit 1
    }

    local model_arg="${MODEL_PATH:-$MODEL}"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python3 - "$model_arg" "$RAW_DIR" "$RUNS" "$WARMUP" <<'PY' > "$RAW_DIR/parsed.csv"
import json, os, sys, time
from vllm import LLM, SamplingParams

model, raw_dir, runs, warmup = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

llm = LLM(model=model, dtype="auto", enforce_eager=False, gpu_memory_utilization=0.85)
tok = llm.get_tokenizer()

def synth_prompt(n_tokens):
    # Deterministic prompt of approx n_tokens after tokenization
    base = "The quick brown fox jumps over the lazy dog. "
    s = base * max(1, n_tokens // 10)
    ids = tok.encode(s)[:n_tokens]
    return tok.decode(ids)

workloads = [
    ("prefill_p512", 512, 1,   1),
    ("decode_g512",  16,  512, 1),
    ("decode_g128",  16,  128, 1),
    ("batch8_g256",  16,  256, 8),
]

for tag, plen, glen, bs in workloads:
    prompt = synth_prompt(plen)
    prompts = [prompt] * bs
    sp = SamplingParams(temperature=0, max_tokens=glen, ignore_eos=True)
    # warmup
    for _ in range(warmup):
        llm.generate(prompts, sp, use_tqdm=False)
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        total_gen = sum(len(o.outputs[0].token_ids) for o in outs)
        # tokens/sec over the whole batch (prefill+decode combined if plen>>glen,
        # else decode-dominated). For prefill_p512 glen=1, so we report prefill
        # throughput = plen*bs / dt.
        if glen == 1:
            tps = plen * bs / dt
        else:
            tps = total_gen / dt
        samples.append(tps)
    import statistics
    med = statistics.median(samples)
    sd  = statistics.stdev(samples) if len(samples) > 1 else 0.0
    print(f'{tag},{plen},{glen},{bs},{med:.3f},{sd:.3f}', flush=True)
PY

    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "$GPU_ID" | head -1)
    TOOLCHAIN="vllm $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo '?')"
}

# ─────────────────────────────────────────────────────────────────────────────
# Backend: MindIE (Ascend 910B/C)
# ─────────────────────────────────────────────────────────────────────────────
run_mindie() {
    [ -n "${ASCEND_HOME_PATH:-}" ] || {
        for se in /usr/local/Ascend/cann-*/set_env.sh /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash; do
            [ -f "$se" ] && { source "$se"; break; }
        done
    }
    [ -n "${ASCEND_HOME_PATH:-}" ] || {
        echo "ERROR: CANN not sourced; set ASCEND_HOME_PATH" >&2; exit 1;
    }

    command -v mindie-benchmark >/dev/null || python3 -c "import mindie_llm" 2>/dev/null || {
        echo "ERROR: neither mindie-benchmark nor python mindie_llm found." >&2
        echo "  Install MindIE from https://www.hiascend.com/developer/download/community/result" >&2
        exit 1
    }

    # MindIE has its own benchmark CLI; we drive it per-workload and parse JSON.
    # Adjust these paths/args to the MindIE release you have installed.
    for wl in "${WORKLOADS[@]}"; do
        read -r tag plen glen bs <<<"$wl"
        echo "--- $tag (p=$plen g=$glen bs=$bs) ---"
        local raw="$RAW_DIR/${tag}.json"
        if mindie-benchmark \
                --model "${MODEL_PATH:-$MODEL}" \
                --input-len "$plen" --output-len "$glen" --batch-size "$bs" \
                --iterations "$RUNS" --warmup "$WARMUP" \
                --output-format json > "$raw" 2>/dev/null; then
            python3 - "$raw" "$tag" "$plen" "$glen" "$bs" >> "$RAW_DIR/parsed.csv" <<'PY'
import json, sys
raw, tag, plen, glen, bs = sys.argv[1:]
d = json.load(open(raw))
# MindIE schema varies by release. Try common keys:
tps = d.get("output_tokens_per_second") or d.get("throughput_tps") \
      or d.get("generate_throughput") or (d.get("total_tokens",0)/d.get("total_time",1))
sd  = d.get("stddev_tps", 0)
print(f'{tag},{plen},{glen},{bs},{tps:.3f},{sd:.3f}')
PY
        else
            echo "  mindie-benchmark failed for $tag" >&2
        fi
    done

    DRIVER_VERSION="CANN $(basename "$(readlink -f "$ASCEND_HOME_PATH")" | sed 's/cann-//')"
    TOOLCHAIN="mindie"
}

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────
: > "$RAW_DIR/parsed.csv"

case "$BACKEND" in
    llama-cpp) run_llama_cpp ;;
    vllm)      run_vllm ;;
    mindie)    run_mindie ;;
esac

# ─────────────────────────────────────────────────────────────────────────────
# Emit pu-rs.org CSV
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -s "$RAW_DIR/parsed.csv" ]; then
    echo "ERROR: no benchmark data produced." >&2
    exit 1
fi

python3 - "$DEVICE_ID" "$MODEL_SLUG" "$RAW_DIR/parsed.csv" "$OUT_CSV" \
        "$DRIVER_VERSION" "$TOOLCHAIN" "$GIT_SHA" "$SUBMITTER" "$QUANT" <<'PY'
import csv, sys

device_id, slug, raw, out, drv, tc, sha, sub, quant = sys.argv[1:]
dtype = quant if quant else "f16"

rows = []
with open(raw) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        tag, plen, glen, bs, tps, sd = line.split(",")
        shape = f"prompt={plen},gen={glen},bs={bs}"
        # We only have tok/s, not latency. Derive a per-token latency as
        # total_tokens_emitted / tps * 1e6 us — represents "time per generated
        # token averaged across batch". For prefill workloads, this is the
        # time per prefill token.
        tps_f = float(tps)
        if tps_f <= 0:
            continue
        lat_us = 1e6 / tps_f   # microseconds per token
        rows.append({
            "device_id": device_id,
            "kernel_id": slug,
            "dtype": dtype,
            "input_shape": shape,
            "batch_size": bs,
            "impl_lang": tc.split()[0] if tc else "python",
            "latency_us": f"{lat_us:.2f}",
            "throughput_gbps": f"{tps_f:.3f}",  # tok/s reused for this column
            "driver_version": drv,
            "toolchain": tc,
            "git_sha": sha,
            "submitter": sub,
        })

with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "device_id","kernel_id","dtype","input_shape","batch_size",
        "impl_lang","latency_us","throughput_gbps",
        "driver_version","toolchain","git_sha","submitter",
    ])
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"Wrote {len(rows)} rows to {out}")
for r in rows:
    print(f"  {r['input_shape']:40s}  {float(r['throughput_gbps']):8.1f} tok/s")
PY

echo ""
echo "Next steps:"
echo "  1. (once) register the pseudo-kernel:"
echo "     python3 -c \"import sqlite3; c=sqlite3.connect('db/xpu_bench.db'); \\"
echo "       c.execute(\\\"INSERT OR IGNORE INTO kernels(kernel_id,display_name,category,description) \\"
echo "         VALUES('${MODEL_SLUG}','${MODEL##*/}','model','End-to-end tok/s, bs=1 unless noted')\\\"); c.commit()\""
echo "  2. ingest:"
echo "     python3 scripts/ingest_bench_csv.py db/xpu_bench.db $OUT_CSV"
