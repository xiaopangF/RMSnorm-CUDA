#include <torch/extension.h>


at::Tensor rmsnorm_forward(at::Tensor x, at::Tensor weight, double eps);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward (CUDA)");
}
