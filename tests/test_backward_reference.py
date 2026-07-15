import pytest
import torch

import rmsnorm_cuda
from rmsnorm_cuda.reference import (
    fused_add_rmsnorm_backward_reference,
    fused_add_rmsnorm_reference,
    rmsnorm_backward_reference,
    rmsnorm_reference,
)


def clone_for_autograd(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape", [(4, 16), (2, 3, 32), (1, 2, 4, 16)])
def test_rmsnorm_backward_reference_matches_autograd(shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(*shape, device="cuda", dtype=torch.float32)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.float32)
    grad_out = torch.randn(*shape, device="cuda", dtype=torch.float32)

    x_autograd = clone_for_autograd(x)
    weight_autograd = clone_for_autograd(weight)
    y = rmsnorm_reference(x_autograd, weight_autograd, eps)
    y.backward(grad_out)

    dx, dweight = rmsnorm_backward_reference(grad_out, x, weight, eps)

    torch.testing.assert_close(dx, x_autograd.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dweight, weight_autograd.grad, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape", [(4, 16), (2, 3, 32), (1, 2, 4, 16)])
def test_fused_add_rmsnorm_backward_reference_matches_autograd(shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(*shape, device="cuda", dtype=torch.float32)
    residual = torch.randn(*shape, device="cuda", dtype=torch.float32)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.float32)
    grad_out = torch.randn(*shape, device="cuda", dtype=torch.float32)

    x_autograd = clone_for_autograd(x)
    residual_autograd = clone_for_autograd(residual)
    weight_autograd = clone_for_autograd(weight)
    y = fused_add_rmsnorm_reference(x_autograd, residual_autograd, weight_autograd, eps)
    y.backward(grad_out)

    dx, dresidual, dweight = fused_add_rmsnorm_backward_reference(
        grad_out,
        x,
        residual,
        weight,
        eps,
    )

    torch.testing.assert_close(dx, x_autograd.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dresidual, residual_autograd.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dweight, weight_autograd.grad, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rmsnorm_reference_gradcheck() -> None:
    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(2, 5, device="cuda", dtype=torch.double, requires_grad=True)
    weight = torch.randn(5, device="cuda", dtype=torch.double, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda x_arg, weight_arg: rmsnorm_reference(x_arg, weight_arg, eps).double(),
        (x, weight),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape", [(4, 16), (2, 3, 32), (1, 2, 4, 16)])
def test_rmsnorm_cuda_backward_matches_reference(shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(*shape, device="cuda", dtype=torch.float32)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.float32)
    grad_out = torch.randn(*shape, device="cuda", dtype=torch.float32)

    actual_dx, actual_dweight = rmsnorm_cuda.rmsnorm_backward(grad_out, x, weight, eps)
    expected_dx, expected_dweight = rmsnorm_backward_reference(grad_out, x, weight, eps)

    torch.testing.assert_close(actual_dx, expected_dx, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_dweight, expected_dweight, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rmsnorm_cuda_backward_requires_float32() -> None:
    x = torch.randn(2, 16, device="cuda", dtype=torch.float16)
    weight = torch.randn(16, device="cuda", dtype=torch.float16)
    grad_out = torch.randn(2, 16, device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="float32"):
        rmsnorm_cuda.rmsnorm_backward(grad_out, x, weight)
