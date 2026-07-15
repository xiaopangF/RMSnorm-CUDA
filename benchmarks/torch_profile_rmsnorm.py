from __future__ import annotations

import argparse
from pathlib import Path

import torch

from profile_rmsnorm import build_workload, parse_dtype, validate_args


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
        default="fused_warp",
    )
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--output-dir", default="profiles")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    dtype = parse_dtype(args.dtype)
    validate_args(args.op, dtype, args.hidden_size)
    workload = build_workload(args.op, args.batch, args.hidden_size, dtype, args.eps)

    for _ in range(args.warmup):
        workload()
    torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        torch.cuda.nvtx.range_push(
            f"{args.op}:batch={args.batch},hidden={args.hidden_size},dtype={args.dtype}"
        )
        for _ in range(args.repeat):
            workload()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / (
        f"torch_trace_{args.op}_{args.dtype}_b{args.batch}_h{args.hidden_size}.json"
    )
    profiler.export_chrome_trace(str(trace_path))

    key_averages = profiler.key_averages()
    cuda_time_total = sum(getattr(event, "cuda_time_total", 0) for event in key_averages)
    if cuda_time_total == 0:
        print(
            "warning: no CUDA activity was captured. "
            "CUPTI may be unavailable or incompatible with the current driver."
        )

    print(key_averages.table(sort_by="cuda_time_total", row_limit=12))
    print(f"trace: {trace_path}")


if __name__ == "__main__":
    main()
