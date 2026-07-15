from __future__ import annotations

import torch

from . import _C


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Run RMSNorm forward with the custom CUDA kernel.

    Args:
        x: CUDA float32 tensor with shape [batch, hidden_size].
        weight: CUDA float32 tensor with shape [hidden_size].
        eps: Small value used for numerical stability.

    Returns:
        CUDA tensor with the same shape as x.
    """
    return _C.rmsnorm_forward(x, weight, float(eps))


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Run residual add and RMSNorm in one custom CUDA kernel.

    This computes rmsnorm(x + residual, weight, eps) without materializing
    x + residual as a separate tensor.
    """
    return _C.fused_add_rmsnorm_forward(x, residual, weight, float(eps))


__all__ = ["fused_add_rmsnorm", "rmsnorm"]
