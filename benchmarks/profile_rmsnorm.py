from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

import rmsnorm_cuda


def parse_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight


def build_workload(
    op: str,
    batch: int,
    hidden_size: int,
    dtype: torch.dtype,
    eps: float,
) -> Callable[[], torch.Tensor]:
    x = torch.randn(batch, hidden_size, device="cuda", dtype=dtype)
    residual = torch.randn(batch, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

    if op == "torch_rmsnorm":
        return lambda: rmsnorm_reference(x, weight, eps)
    if op == "torch_fused":
        return lambda: rmsnorm_reference(x + residual, weight, eps)
    if op == "rmsnorm_shared":
        return lambda: rmsnorm_cuda.rmsnorm(x, weight, eps)
    if op == "rmsnorm_warp":
        return lambda: rmsnorm_cuda.rmsnorm_warp(x, weight, eps)
    if op == "rmsnorm_half2":
        return lambda: rmsnorm_cuda.rmsnorm_half2(x, weight, eps)
    if op == "fused_shared":
        return lambda: rmsnorm_cuda.fused_add_rmsnorm(x, residual, weight, eps)
    if op == "fused_warp":
        return lambda: rmsnorm_cuda.fused_add_rmsnorm_warp(x, residual, weight, eps)
    if op == "fused_half2":
        return lambda: rmsnorm_cuda.fused_add_rmsnorm_half2(x, residual, weight, eps)
    raise ValueError(f"unsupported op: {op}")


def validate_args(op: str, dtype: torch.dtype, hidden_size: int) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise SystemExit("bfloat16 is not supported by this CUDA device.")
    if op.endswith("half2") and dtype != torch.float16:
        raise SystemExit("half2 ops require --dtype float16.")
    if op.endswith("half2") and hidden_size % 2 != 0:
        raise SystemExit("half2 ops require an even --hidden-size.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op",
        choices=[
            "torch_rmsnorm",
            "torch_fused",
            "rmsnorm_shared",
            "rmsnorm_warp",
            "rmsnorm_half2",
            "fused_shared",
            "fused_warp",
            "fused_half2",
        ],
        default="rmsnorm_warp",
    )
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    dtype = parse_dtype(args.dtype)
    validate_args(args.op, dtype, args.hidden_size)
    workload = build_workload(args.op, args.batch, args.hidden_size, dtype, args.eps)

    for _ in range(args.warmup):
        workload()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.nvtx.range_push(
        f"{args.op}:batch={args.batch},hidden={args.hidden_size},dtype={args.dtype}"
    )
    start.record()
    for _ in range(args.repeat):
        workload()
    end.record()
    end.synchronize()
    torch.cuda.nvtx.range_pop()

    total_ms = start.elapsed_time(end)
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"op: {args.op}")
    print(f"dtype: {args.dtype}")
    print(f"shape: [{args.batch}, {args.hidden_size}]")
    print(f"repeat: {args.repeat}")
    print(f"total_ms: {total_ms:.4f}")
    print(f"avg_ms: {total_ms / args.repeat:.6f}")


if __name__ == "__main__":
    main()
