#include <ATen/ATen.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>


namespace {

__global__ void rmsnorm_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    int64_t rows,
    int64_t hidden_size,
    float eps) {
  extern __shared__ float shared_sum[];

  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  if (row >= rows) {
    return;
  }

  const int64_t base = static_cast<int64_t>(row) * hidden_size;

  float local_sum = 0.0f;
  for (int64_t col = tid; col < hidden_size; col += blockDim.x) {
    const float value = x[base + col];
    local_sum += value * value;
  }

  shared_sum[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    __syncthreads();
  }

  const float inv_rms = rsqrtf(shared_sum[0] / static_cast<float>(hidden_size) + eps);

  for (int64_t col = tid; col < hidden_size; col += blockDim.x) {
    y[base + col] = x[base + col] * inv_rms * weight[col];
  }
}

}  // namespace


at::Tensor rmsnorm_forward(at::Tensor x, at::Tensor weight, double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(x.dtype() == at::kFloat, "x must be float32");
  TORCH_CHECK(weight.dtype() == at::kFloat, "weight must be float32");
  TORCH_CHECK(x.dim() == 2, "x must have shape [batch, hidden_size]");
  TORCH_CHECK(weight.dim() == 1, "weight must have shape [hidden_size]");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

  const int64_t rows = x.size(0);
  const int64_t hidden_size = x.size(1);
  TORCH_CHECK(weight.size(0) == hidden_size, "weight size must match x hidden_size");
  TORCH_CHECK(hidden_size > 0, "hidden_size must be greater than 0");

  auto y = at::empty_like(x);
  if (rows == 0) {
    return y;
  }

  constexpr int threads = 256;
  const dim3 blocks(static_cast<unsigned int>(rows));
  const size_t shared_bytes = threads * sizeof(float);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  rmsnorm_f32_kernel<<<blocks, threads, shared_bytes, stream>>>(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      y.data_ptr<float>(),
      rows,
      hidden_size,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return y;
}
