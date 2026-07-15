from __future__ import annotations

import torch


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pure PyTorch RMSNorm reference with autograd support."""
    compute_dtype = _compute_dtype(x.dtype)
    x_float = x.to(compute_dtype)
    weight_float = weight.to(compute_dtype)
    mean_square = torch.mean(x_float * x_float, dim=-1, keepdim=True)
    y = x_float * torch.rsqrt(mean_square + eps) * weight_float
    return y.to(x.dtype)


def fused_add_rmsnorm_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pure PyTorch fused residual add + RMSNorm reference."""
    return rmsnorm_reference(x + residual, weight, eps)


def rmsnorm_backward_reference(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Manual RMSNorm backward reference.

    Forward:
        y = x * inv_rms * weight
        inv_rms = rsqrt(mean(x * x) + eps)

    Returns:
        dx and dweight, both cast back to the input dtypes.
    """
    compute_dtype = _compute_dtype(x.dtype)
    x_float = x.to(compute_dtype)
    grad_float = grad_out.to(compute_dtype)
    weight_float = weight.to(compute_dtype)

    hidden_size = x.shape[-1]
    mean_square = torch.mean(x_float * x_float, dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_square + eps)

    scaled_grad = grad_float * weight_float
    dot = torch.sum(scaled_grad * x_float, dim=-1, keepdim=True)
    dx = inv_rms * scaled_grad - (x_float * inv_rms * inv_rms * inv_rms / hidden_size) * dot

    reduce_dims = tuple(range(grad_float.dim() - 1))
    dweight = torch.sum(grad_float * x_float * inv_rms, dim=reduce_dims)

    return dx.to(x.dtype), dweight.to(weight.dtype)


def fused_add_rmsnorm_backward_reference(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Manual backward reference for y = rmsnorm(x + residual, weight)."""
    dx, dweight = rmsnorm_backward_reference(grad_out, x + residual, weight, eps)
    return dx.to(x.dtype), dx.to(residual.dtype), dweight
