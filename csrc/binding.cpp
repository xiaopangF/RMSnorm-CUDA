#include <torch/extension.h>


at::Tensor rmsnorm_forward(at::Tensor x, at::Tensor weight, double eps);
at::Tensor fused_add_rmsnorm_forward(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor weight,
    double eps);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward (CUDA)");
  m.def(
      "fused_add_rmsnorm_forward",
      &fused_add_rmsnorm_forward,
      "Fused residual add and RMSNorm forward (CUDA)");
}
