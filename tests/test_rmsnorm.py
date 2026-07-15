import pytest
import torch

import rmsnorm_cuda


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    weight_float = weight.float()
    y = x_float / torch.sqrt(torch.mean(x_float * x_float, dim=-1, keepdim=True) + eps) * weight_float
    return y.to(x.dtype)


def supported_dtypes() -> list[torch.dtype]:
    dtypes = [torch.float32, torch.float16]
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    return dtypes


def tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.float16:
        return 1e-3, 1e-3
    return 2e-2, 2e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch,hidden_size", [(1, 16), (4, 1024), (32, 4096)])
@pytest.mark.parametrize("dtype", supported_dtypes())
@pytest.mark.parametrize(
    "rmsnorm_impl",
    [rmsnorm_cuda.rmsnorm, rmsnorm_cuda.rmsnorm_warp],
    ids=["shared", "warp"],
)
def test_rmsnorm_matches_pytorch(
    batch: int,
    hidden_size: int,
    dtype: torch.dtype,
    rmsnorm_impl,
) -> None:
    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(batch, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

    actual = rmsnorm_impl(x, weight, eps)
    expected = rmsnorm_reference(x, weight, eps)
    rtol, atol = tolerances(dtype)

    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch,hidden_size", [(1, 16), (4, 1024), (32, 4096)])
@pytest.mark.parametrize("dtype", supported_dtypes())
@pytest.mark.parametrize(
    "fused_impl",
    [rmsnorm_cuda.fused_add_rmsnorm, rmsnorm_cuda.fused_add_rmsnorm_warp],
    ids=["shared", "warp"],
)
def test_fused_add_rmsnorm_matches_pytorch(
    batch: int,
    hidden_size: int,
    dtype: torch.dtype,
    fused_impl,
) -> None:
    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(batch, hidden_size, device="cuda", dtype=dtype)
    residual = torch.randn(batch, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

    actual = fused_impl(x, residual, weight, eps)
    expected = rmsnorm_reference(x + residual, weight, eps)
    rtol, atol = tolerances(dtype)

    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch,hidden_size", [(1, 16), (4, 1024), (32, 4096)])
def test_rmsnorm_half2_matches_pytorch(batch: int, hidden_size: int) -> None:
    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(batch, hidden_size, device="cuda", dtype=torch.float16)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float16)

    actual = rmsnorm_cuda.rmsnorm_half2(x, weight, eps)
    expected = rmsnorm_reference(x, weight, eps)

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch,hidden_size", [(1, 16), (4, 1024), (32, 4096)])
def test_fused_add_rmsnorm_half2_matches_pytorch(batch: int, hidden_size: int) -> None:
    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(batch, hidden_size, device="cuda", dtype=torch.float16)
    residual = torch.randn(batch, hidden_size, device="cuda", dtype=torch.float16)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float16)

    actual = rmsnorm_cuda.fused_add_rmsnorm_half2(x, residual, weight, eps)
    expected = rmsnorm_reference(x + residual, weight, eps)

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rmsnorm_half2_requires_even_hidden_size() -> None:
    x = torch.randn(2, 15, device="cuda", dtype=torch.float16)
    weight = torch.randn(15, device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="even"):
        rmsnorm_cuda.rmsnorm_half2(x, weight)
