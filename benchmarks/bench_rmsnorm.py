from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch

import rmsnorm_cuda


@dataclass(frozen=True)
class TimingStats:
    mean_ms: float
    median_ms: float
    p90_ms: float
    min_ms: float


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight


def fused_add_rmsnorm_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return rmsnorm_reference(x + residual, weight, eps)


def percentile(samples: list[float], pct: float) -> float:
    if not samples:
        raise ValueError("samples must not be empty")

    ordered = sorted(samples)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def time_cuda(fn, warmup: int, repeat: int) -> TimingStats:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples_ms = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end))

    return TimingStats(
        mean_ms=statistics.mean(samples_ms),
        median_ms=statistics.median(samples_ms),
        p90_ms=percentile(samples_ms, 0.90),
        min_ms=min(samples_ms),
    )


def estimate_custom_bytes(batch: int, hidden_size: int, dtype_size: int) -> int:
    # Current kernel reads x twice, reads weight once, and writes y once.
    x_read = batch * hidden_size * dtype_size * 2
    weight_read = batch * hidden_size * dtype_size
    y_write = batch * hidden_size * dtype_size
    return x_read + weight_read + y_write


def estimate_fused_custom_bytes(batch: int, hidden_size: int, dtype_size: int) -> int:
    # Fused kernel reads x and residual twice, reads weight once, and writes y once.
    x_read = batch * hidden_size * dtype_size * 2
    residual_read = batch * hidden_size * dtype_size * 2
    weight_read = batch * hidden_size * dtype_size
    y_write = batch * hidden_size * dtype_size
    return x_read + residual_read + weight_read + y_write


def gb_per_second(num_bytes: int, elapsed_ms: float) -> float:
    if elapsed_ms <= 0:
        return float("inf")
    return num_bytes / (elapsed_ms / 1000.0) / 1e9


def get_shapes(extended: bool) -> list[tuple[int, int]]:
    if extended:
        return [
            (1, 1024),
            (1, 2048),
            (1, 4096),
            (1, 8192),
            (8, 4096),
            (16, 4096),
            (32, 4096),
            (32, 8192),
            (64, 4096),
            (64, 8192),
        ]

    return [
        (1, 1024),
        (1, 4096),
        (8, 4096),
        (32, 4096),
        (32, 8192),
    ]


def parse_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--extended", action="store_true", help="Run a wider shape sweep.")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Tensor dtype used by the benchmark.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("bfloat16 is not supported by this CUDA device.")

    eps = 1e-6
    dtype = parse_dtype(args.dtype)

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"dtype: {args.dtype}")
    print(f"warmup: {args.warmup}, repeat: {args.repeat}")
    print()
    print("RMSNorm")
    print(
        f"{'batch':>8} {'hidden':>8} "
        f"{'torch':>9} {'shared':>9} {'warp':>9} {'warp p90':>9} "
        f"{'warp GB/s':>10} {'warp/shared':>12} {'torch/warp':>11}"
    )

    for batch, hidden_size in get_shapes(args.extended):
        x = torch.randn(batch, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        torch_stats = time_cuda(lambda: rmsnorm_reference(x, weight, eps), args.warmup, args.repeat)
        shared_stats = time_cuda(lambda: rmsnorm_cuda.rmsnorm(x, weight, eps), args.warmup, args.repeat)
        warp_stats = time_cuda(lambda: rmsnorm_cuda.rmsnorm_warp(x, weight, eps), args.warmup, args.repeat)
        torch_speedup = torch_stats.median_ms / warp_stats.median_ms
        warp_vs_shared = shared_stats.median_ms / warp_stats.median_ms
        bandwidth = gb_per_second(
            estimate_custom_bytes(batch, hidden_size, x.element_size()),
            warp_stats.median_ms,
        )

        print(
            f"{batch:8d} {hidden_size:8d} "
            f"{torch_stats.median_ms:9.4f} {shared_stats.median_ms:9.4f} "
            f"{warp_stats.median_ms:9.4f} {warp_stats.p90_ms:9.4f} "
            f"{bandwidth:10.2f} {warp_vs_shared:12.2f}x {torch_speedup:10.2f}x"
        )

    print()
    print("Fused add + RMSNorm")
    print(
        f"{'batch':>8} {'hidden':>8} "
        f"{'torch':>9} {'shared':>9} {'warp':>9} {'warp p90':>9} "
        f"{'warp GB/s':>10} {'warp/shared':>12} {'torch/warp':>11}"
    )

    for batch, hidden_size in get_shapes(args.extended):
        x = torch.randn(batch, hidden_size, device="cuda", dtype=dtype)
        residual = torch.randn(batch, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        torch_stats = time_cuda(
            lambda: fused_add_rmsnorm_reference(x, residual, weight, eps),
            args.warmup,
            args.repeat,
        )
        shared_stats = time_cuda(
            lambda: rmsnorm_cuda.fused_add_rmsnorm(x, residual, weight, eps),
            args.warmup,
            args.repeat,
        )
        warp_stats = time_cuda(
            lambda: rmsnorm_cuda.fused_add_rmsnorm_warp(x, residual, weight, eps),
            args.warmup,
            args.repeat,
        )
        torch_speedup = torch_stats.median_ms / warp_stats.median_ms
        warp_vs_shared = shared_stats.median_ms / warp_stats.median_ms
        bandwidth = gb_per_second(
            estimate_fused_custom_bytes(batch, hidden_size, x.element_size()),
            warp_stats.median_ms,
        )

        print(
            f"{batch:8d} {hidden_size:8d} "
            f"{torch_stats.median_ms:9.4f} {shared_stats.median_ms:9.4f} "
            f"{warp_stats.median_ms:9.4f} {warp_stats.p90_ms:9.4f} "
            f"{bandwidth:10.2f} {warp_vs_shared:12.2f}x {torch_speedup:10.2f}x"
        )


if __name__ == "__main__":
    main()
