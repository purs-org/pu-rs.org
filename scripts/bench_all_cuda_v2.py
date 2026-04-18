#!/usr/bin/env python3
"""
bench_all_cuda_v2.py — A100 / CUDA benchmark v2.

Fixes the methodology issues in the v1 (bench_all_cuda.ipynb) CSV:

  1. Dtype: default f16 (tensor cores). v1 was f32-only, which caps A100 at
     ~19.5 TFLOPS and makes it look 16x slower than Ascend's f16 numbers.
     Optional --dtype f32 run enables TF32 on Ampere for a fair f32 baseline.
  2. Shapes: match the Ascend 910B grid (matmul up to 16384^2, attention up
     to S=4096) so rows are directly comparable on the leaderboard.
  3. Timing: CUDA events around measurement loop, not time.perf_counter.
     Warmup (10 iters) discarded; measurement (50 iters) kept.
  4. Toolchain: PyTorch/cuBLAS/cuDNN/Flash-Attention rather than eager
     F.gelu/F.layer_norm wrappers on tiny shapes.

Output is the standard pu-rs.org submission CSV (same columns as
submissions/google-tpu-v5e_*.csv), one row per measurement iteration so the
ingester can compute min/median/p99 itself.

Usage (run on a box with an A100):

  python3 scripts/bench_all_cuda_v2.py \
      --device nvidia-a100-80 \
      --dtype f16 \
      -o submissions/nvidia-a100-80_v2_submission.csv

  # Optional: tf32 pass for f32 baseline
  python3 scripts/bench_all_cuda_v2.py --device nvidia-a100-80 --dtype tf32 \
      -o submissions/nvidia-a100-80_v2_tf32_submission.csv
"""

import argparse
import csv
import math
import os
import sys

import torch
import torch.nn.functional as F


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", required=True,
                   help="device_id to record in CSV (e.g. nvidia-a100-80)")
    p.add_argument("--dtype", default="f16", choices=["f16", "bf16", "tf32", "f32"],
                   help="data type: f16/bf16 hit tensor cores; tf32 enables "
                        "TF32 on f32 matmuls; f32 is the strict-IEEE baseline")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--submitter", default="bench_all_cuda_v2.py")
    p.add_argument("-o", "--out", default="-", help="output CSV path or '-' for stdout")
    p.add_argument("--only", default=None,
                   help="comma-separated kernel ids to run (default: all)")
    return p.parse_args()


# ── Timing ──────────────────────────────────────────────────────────────────

def time_kernel(fn, warmup, iters):
    """Run fn() warmup+iters times. Return list of per-iter GPU times in us."""
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    # ms -> us
    return [starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(iters)]


# ── Throughput helpers ──────────────────────────────────────────────────────

def bandwidth_gbps(bytes_moved, us):
    return bytes_moved / (us * 1e-6) / 1e9


def flops_to_gflops(flops, us):
    return flops / (us * 1e-6) / 1e9


# ── Kernels ─────────────────────────────────────────────────────────────────
# Each runner returns: (torch_dtype, shape_str, batch_size, flops_or_bytes, toolchain, fn)
# flops_or_bytes is used only for notes; the CSV throughput is computed per iter.

def dtype_to_torch(dt):
    return {"f16": torch.float16, "bf16": torch.bfloat16,
            "tf32": torch.float32, "f32": torch.float32}[dt]


def runners_matmul(dt):
    td = dtype_to_torch(dt)
    shapes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (16384, 16384, 16384),
        (1024, 4096, 1024),  # FFN down
        (4096, 1024, 4096),  # FFN up
        (2048, 8192, 2048),  # attention proj
    ]
    for M, K, N in shapes:
        a = torch.randn(M, K, device="cuda", dtype=td)
        b = torch.randn(K, N, device="cuda", dtype=td)
        out = torch.empty(M, N, device="cuda", dtype=td)
        if M == K == N:
            shape_str = f"[{M}, {N}]"
        else:
            shape_str = f"[{M}, {K}] x [{K}, {N}]"
        flops = 2.0 * M * K * N
        yield ("matmul", shape_str, 1, flops, "torch.matmul (cuBLAS)",
               lambda a=a, b=b, out=out: torch.matmul(a, b, out=out))


def runners_attention(dt):
    td = dtype_to_torch(dt)
    # (B, H, S, D) — matches Ascend grid
    shapes = [
        (1, 1,  128, 64),
        (1, 1,  512, 64),
        (1, 1, 1024, 64),
        (1, 1, 2048, 64),
        (1, 1, 4096, 64),
        (1, 8,  512, 64),
        (1, 12, 512, 64),
        (1, 32, 512, 64),
        (1, 32, 1024, 128),
        (1, 32, 2048, 128),
    ]
    for B, H, S, D in shapes:
        q = torch.randn(B, H, S, D, device="cuda", dtype=td)
        k = torch.randn(B, H, S, D, device="cuda", dtype=td)
        v = torch.randn(B, H, S, D, device="cuda", dtype=td)
        shape_str = f"B={B},H={H},S={S},D={D}"
        flops = 4.0 * B * H * S * S * D
        yield ("attention", shape_str, B, flops, "F.scaled_dot_product_attention (Flash)",
               lambda q=q, k=k, v=v: F.scaled_dot_product_attention(q, k, v))


def runners_softmax(dt):
    td = dtype_to_torch(dt)
    shapes = [(1, 1024), (64, 1024), (64, 4096), (1024, 4096), (1, 16384)]
    for R, C in shapes:
        x = torch.randn(R, C, device="cuda", dtype=td)
        bytes_moved = 2.0 * R * C * x.element_size()  # read + write
        shape_str = f"[{R}, {C}]"
        yield ("softmax", shape_str, R, bytes_moved, "F.softmax",
               lambda x=x: F.softmax(x, dim=-1))


def runners_layernorm(dt):
    td = dtype_to_torch(dt)
    # Ascend used 768/4096 hidden dim
    shapes = [(1, 768), (64, 768), (1024, 768), (1, 4096), (64, 4096), (1024, 4096)]
    for R, C in shapes:
        x = torch.randn(R, C, device="cuda", dtype=td)
        g = torch.randn(C, device="cuda", dtype=td)
        b = torch.randn(C, device="cuda", dtype=td)
        bytes_moved = 2.0 * R * C * x.element_size()
        shape_str = f"[{R}, {C}]"
        yield ("layernorm", shape_str, R, bytes_moved, "F.layer_norm",
               lambda x=x, g=g, b=b, C=C: F.layer_norm(x, (C,), g, b))


def runners_rms_norm(dt):
    td = dtype_to_torch(dt)
    shapes = [(1, 768), (64, 768), (1, 4096), (64, 4096), (1024, 4096)]
    # torch.nn.functional has no rms_norm; inline it
    def rms(x, g, eps=1e-5):
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + eps) * g
    for R, C in shapes:
        x = torch.randn(R, C, device="cuda", dtype=td)
        g = torch.randn(C, device="cuda", dtype=td)
        bytes_moved = 2.0 * R * C * x.element_size()
        shape_str = f"[{R}, {C}]"
        yield ("rms-norm", shape_str, R, bytes_moved, "inline rms (torch ops)",
               lambda x=x, g=g: rms(x, g))


def runners_gelu(dt):
    td = dtype_to_torch(dt)
    shapes = [(1, 768), (1, 4096), (64, 768), (64, 4096), (1024, 4096)]
    for R, C in shapes:
        x = torch.randn(R, C, device="cuda", dtype=td)
        bytes_moved = 2.0 * R * C * x.element_size()
        shape_str = f"[{R}, {C}]"
        yield ("gelu", shape_str, R, bytes_moved, "F.gelu(tanh)",
               lambda x=x: F.gelu(x, approximate="tanh"))


def runners_silu(dt):
    td = dtype_to_torch(dt)
    shapes = [(1, 768), (1, 4096), (64, 768), (64, 4096), (1024, 4096)]
    for R, C in shapes:
        x = torch.randn(R, C, device="cuda", dtype=td)
        bytes_moved = 2.0 * R * C * x.element_size()
        shape_str = f"[{R}, {C}]"
        yield ("silu", shape_str, R, bytes_moved, "F.silu",
               lambda x=x: F.silu(x))


def runners_rope(dt):
    td = dtype_to_torch(dt)
    # (B, S, D) — D must be even
    shapes = [(1, 64, 128), (1, 128, 128), (32, 64, 128), (1, 2048, 128)]
    def rope(x):
        B, S, D = x.shape
        half = D // 2
        freqs = torch.arange(half, device=x.device, dtype=torch.float32)
        freqs = 1.0 / (10000.0 ** (2 * freqs / D))
        pos = torch.arange(S, device=x.device, dtype=torch.float32)
        theta = torch.outer(pos, freqs)  # (S, half)
        cos = theta.cos().to(x.dtype)
        sin = theta.sin().to(x.dtype)
        x1 = x[..., :half]
        x2 = x[..., half:]
        out = torch.empty_like(x)
        out[..., :half] = x1 * cos - x2 * sin
        out[..., half:] = x1 * sin + x2 * cos
        return out
    for B, S, D in shapes:
        x = torch.randn(B, S, D, device="cuda", dtype=td)
        bytes_moved = 2.0 * B * S * D * x.element_size()
        shape_str = f"B={B},S={S},D={D}"
        yield ("rope", shape_str, B, bytes_moved, "inline rope (torch ops)",
               lambda x=x: rope(x))


def runners_embedding(dt):
    td = dtype_to_torch(dt)
    shapes = [(32, 32000, 128), (128, 32000, 128), (32, 32000, 4096)]
    for N, V, D in shapes:
        w = torch.randn(V, D, device="cuda", dtype=td)
        idx = torch.randint(0, V, (N,), device="cuda", dtype=torch.long)
        bytes_moved = N * D * w.element_size()  # read the gathered rows
        shape_str = f"N={N},V={V},D={D}"
        yield ("embedding", shape_str, N, bytes_moved, "F.embedding",
               lambda w=w, idx=idx: F.embedding(idx, w))


def runners_cross_entropy(dt):
    td = dtype_to_torch(dt)
    shapes = [(32, 32000), (128, 32000), (32, 50257)]
    for N, V in shapes:
        logits = torch.randn(N, V, device="cuda", dtype=td)
        tgt = torch.randint(0, V, (N,), device="cuda", dtype=torch.long)
        bytes_moved = N * V * logits.element_size()
        shape_str = f"[{N}, {V}]"
        yield ("cross-entropy", shape_str, N, bytes_moved, "F.cross_entropy",
               lambda logits=logits, tgt=tgt: F.cross_entropy(logits, tgt))


def runners_causal_mask(dt):
    td = dtype_to_torch(dt)
    shapes = [64, 128, 256, 512, 1024, 2048]
    neg_inf = torch.finfo(td).min
    for S in shapes:
        x = torch.randn(S, S, device="cuda", dtype=td)
        mask = torch.triu(torch.ones(S, S, device="cuda", dtype=torch.bool), diagonal=1)
        bytes_moved = 2.0 * S * S * x.element_size()
        shape_str = f"[{S}, {S}]"
        yield ("causal-mask", shape_str, 1, bytes_moved, "masked_fill",
               lambda x=x, mask=mask, neg_inf=neg_inf: x.masked_fill(mask, neg_inf))


def runners_vq_quantize(dt):
    td = dtype_to_torch(dt)
    shapes = [(256, 512, 64), (1024, 512, 64), (1024, 1024, 128),
              (4096, 512, 64), (4096, 1024, 128)]
    def vq(x, cb):
        # dist = x^2 - 2 x·cb^T + cb^2 (broadcast)
        dist = torch.cdist(x, cb, p=2)
        idx = dist.argmin(dim=-1)
        return cb[idx]
    for N, K, D in shapes:
        x = torch.randn(N, D, device="cuda", dtype=td)
        cb = torch.randn(K, D, device="cuda", dtype=td)
        flops = 2.0 * N * K * D
        shape_str = f"N={N},K={K},D={D}"
        yield ("vq-quantize", shape_str, N, flops, "torch.cdist + argmin",
               lambda x=x, cb=cb: vq(x, cb))


def runners_conv1d(dt):
    td = dtype_to_torch(dt)
    # (B, L, C, K=3) — dilated conv1d, kernel=3
    shapes = [(1, 1024, 64), (1, 4096, 64), (16, 1024, 64), (16, 4096, 64)]
    for B, L, C in shapes:
        x = torch.randn(B, C, L, device="cuda", dtype=td)  # NCL for conv1d
        w = torch.randn(C, C, 3, device="cuda", dtype=td)
        flops = 2.0 * B * L * C * C * 3
        shape_str = f"B={B},L={L},C={C},K=3"
        yield ("conv1d-dilated", shape_str, B, flops,
               "F.conv1d(dilation=2) + relu",
               lambda x=x, w=w: F.relu(F.conv1d(x, w, padding=2, dilation=2)))


ALL_RUNNERS = {
    "matmul":        runners_matmul,
    "attention":     runners_attention,
    "softmax":       runners_softmax,
    "layernorm":     runners_layernorm,
    "rms-norm":      runners_rms_norm,
    "gelu":          runners_gelu,
    "silu":          runners_silu,
    "rope":          runners_rope,
    "embedding":     runners_embedding,
    "cross-entropy": runners_cross_entropy,
    "causal-mask":   runners_causal_mask,
    "vq-quantize":   runners_vq_quantize,
    "conv1d-dilated": runners_conv1d,
}


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA device not available")

    # Dtype setup: enable TF32 for the 'tf32' pass, strict f32 otherwise.
    if args.dtype == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        dtype_label = "f32"  # CSV dtype column — output is still f32-typed
    elif args.dtype == "f32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        dtype_label = "f32"
    else:
        dtype_label = args.dtype  # "f16" or "bf16"

    # cuDNN benchmark picks the fastest conv algorithm for the shape.
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    driver = torch.version.cuda or "unknown"
    toolchain_prefix = f"PyTorch {torch.__version__} / CUDA {driver}"

    only = set(args.only.split(",")) if args.only else None

    out = sys.stdout if args.out == "-" else open(args.out, "w", newline="")
    writer = csv.writer(out)
    writer.writerow([
        "device_id", "kernel_id", "dtype", "input_shape", "batch_size",
        "impl_lang", "latency_us", "throughput_gbps",
        "driver_version", "toolchain", "git_sha", "submitter",
    ])

    for kname, builder in ALL_RUNNERS.items():
        if only is not None and kname not in only:
            continue
        for kernel_id, shape_str, batch, work, toolchain, fn in builder(args.dtype):
            # Skip rope B=1,S=2048 with bf16 -- avoids OOM on 8GB-ish test boxes;
            # at A100-80 we always have headroom, so no filter here.
            try:
                lats_us = time_kernel(fn, args.warmup, args.iters)
            except Exception as e:
                print(f"  !! {kernel_id} {shape_str}: {e}", file=sys.stderr)
                continue
            # Per-iter throughput: flops for compute-bound kernels
            # (matmul/attention/vq/conv1d), bytes for bandwidth-bound.
            is_compute = kernel_id in ("matmul", "attention", "vq-quantize",
                                       "conv1d-dilated")
            for lat in lats_us:
                if is_compute:
                    thr = flops_to_gflops(work, lat)
                else:
                    thr = bandwidth_gbps(work, lat)
                writer.writerow([
                    args.device, kernel_id, dtype_label, shape_str, batch,
                    "cuda", f"{lat:.2f}", f"{thr:.2f}",
                    f"CUDA {driver}", f"{toolchain_prefix} / {toolchain}",
                    "", args.submitter,
                ])
            # progress to stderr
            med = sorted(lats_us)[len(lats_us) // 2]
            unit = "GFLOPS" if is_compute else "GB/s"
            thr_med = flops_to_gflops(work, med) if is_compute else bandwidth_gbps(work, med)
            print(f"  {kernel_id:14s} {shape_str:28s} "
                  f"{dtype_label:5s} median {med:8.1f}us  {thr_med:9.1f} {unit}",
                  file=sys.stderr)

    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
