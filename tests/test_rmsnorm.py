import pytest
import torch

import rmsnorm_cuda


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch,hidden_size", [(1, 16), (4, 1024), (32, 4096)])
def test_rmsnorm_matches_pytorch(batch: int, hidden_size: int) -> None:
    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(batch, hidden_size, device="cuda", dtype=torch.float32)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float32)

    actual = rmsnorm_cuda.rmsnorm(x, weight, eps)
    expected = rmsnorm_reference(x, weight, eps)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch,hidden_size", [(1, 16), (4, 1024), (32, 4096)])
def test_fused_add_rmsnorm_matches_pytorch(batch: int, hidden_size: int) -> None:
    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(batch, hidden_size, device="cuda", dtype=torch.float32)
    residual = torch.randn(batch, hidden_size, device="cuda", dtype=torch.float32)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float32)

    actual = rmsnorm_cuda.fused_add_rmsnorm(x, residual, weight, eps)
    expected = rmsnorm_reference(x + residual, weight, eps)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rmsnorm_requires_contiguous_input() -> None:
    x = torch.randn(8, 16, device="cuda", dtype=torch.float32).t()
    weight = torch.randn(8, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="contiguous"):
        rmsnorm_cuda.rmsnorm(x, weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_add_rmsnorm_requires_contiguous_residual() -> None:
    x = torch.randn(16, 8, device="cuda", dtype=torch.float32)
    residual = torch.randn(8, 16, device="cuda", dtype=torch.float32).t()
    weight = torch.randn(8, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="contiguous"):
        rmsnorm_cuda.fused_add_rmsnorm(x, residual, weight)
