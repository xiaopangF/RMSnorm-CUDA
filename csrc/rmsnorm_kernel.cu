#include <ATen/ATen.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>


namespace {

bool is_supported_dtype(at::ScalarType dtype) {
  return dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16;
}

template <typename scalar_t>
__global__ void rmsnorm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ y,
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
    const float value = static_cast<float>(x[base + col]);
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
    const float value = static_cast<float>(x[base + col]);
    const float scale = static_cast<float>(weight[col]);
    y[base + col] = static_cast<scalar_t>(value * inv_rms * scale);
  }
}

template <typename scalar_t>
__global__ void fused_add_rmsnorm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ y,
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
    const float value =
        static_cast<float>(x[base + col]) + static_cast<float>(residual[base + col]);
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
    const float value =
        static_cast<float>(x[base + col]) + static_cast<float>(residual[base + col]);
    const float scale = static_cast<float>(weight[col]);
    y[base + col] = static_cast<scalar_t>(value * inv_rms * scale);
  }
}

template <typename scalar_t>
void launch_rmsnorm_kernel(
    at::Tensor x,
    at::Tensor weight,
    at::Tensor y,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    cudaStream_t stream) {
  constexpr int threads = 256;
  const dim3 blocks(static_cast<unsigned int>(rows));
  const size_t shared_bytes = threads * sizeof(float);

  rmsnorm_kernel<scalar_t><<<blocks, threads, shared_bytes, stream>>>(
      x.data_ptr<scalar_t>(),
      weight.data_ptr<scalar_t>(),
      y.data_ptr<scalar_t>(),
      rows,
      hidden_size,
      eps);
}

template <typename scalar_t>
void launch_fused_add_rmsnorm_kernel(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor weight,
    at::Tensor y,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    cudaStream_t stream) {
  constexpr int threads = 256;
  const dim3 blocks(static_cast<unsigned int>(rows));
  const size_t shared_bytes = threads * sizeof(float);

  fused_add_rmsnorm_kernel<scalar_t><<<blocks, threads, shared_bytes, stream>>>(
      x.data_ptr<scalar_t>(),
      residual.data_ptr<scalar_t>(),
      weight.data_ptr<scalar_t>(),
      y.data_ptr<scalar_t>(),
      rows,
      hidden_size,
      eps);
}

}  // namespace


at::Tensor rmsnorm_forward(at::Tensor x, at::Tensor weight, double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(is_supported_dtype(x.scalar_type()), "x must be float32, float16, or bfloat16");
  TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "weight dtype must match x dtype");
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

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  switch (x.scalar_type()) {
    case at::kFloat:
      launch_rmsnorm_kernel<float>(x, weight, y, rows, hidden_size, static_cast<float>(eps), stream);
      break;
    case at::kHalf:
      launch_rmsnorm_kernel<at::Half>(x, weight, y, rows, hidden_size, static_cast<float>(eps), stream);
      break;
    case at::kBFloat16:
      launch_rmsnorm_kernel<at::BFloat16>(
          x, weight, y, rows, hidden_size, static_cast<float>(eps), stream);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return y;
}


at::Tensor fused_add_rmsnorm_forward(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor weight,
    double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(is_supported_dtype(x.scalar_type()), "x must be float32, float16, or bfloat16");
  TORCH_CHECK(residual.scalar_type() == x.scalar_type(), "residual dtype must match x dtype");
  TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "weight dtype must match x dtype");
  TORCH_CHECK(x.dim() == 2, "x must have shape [batch, hidden_size]");
  TORCH_CHECK(residual.dim() == 2, "residual must have shape [batch, hidden_size]");
  TORCH_CHECK(weight.dim() == 1, "weight must have shape [hidden_size]");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(residual.sizes() == x.sizes(), "residual shape must match x shape");

  const int64_t rows = x.size(0);
  const int64_t hidden_size = x.size(1);
  TORCH_CHECK(weight.size(0) == hidden_size, "weight size must match x hidden_size");
  TORCH_CHECK(hidden_size > 0, "hidden_size must be greater than 0");

  auto y = at::empty_like(x);
  if (rows == 0) {
    return y;
  }

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  switch (x.scalar_type()) {
    case at::kFloat:
      launch_fused_add_rmsnorm_kernel<float>(
          x, residual, weight, y, rows, hidden_size, static_cast<float>(eps), stream);
      break;
    case at::kHalf:
      launch_fused_add_rmsnorm_kernel<at::Half>(
          x, residual, weight, y, rows, hidden_size, static_cast<float>(eps), stream);
      break;
    case at::kBFloat16:
      launch_fused_add_rmsnorm_kernel<at::BFloat16>(
          x, residual, weight, y, rows, hidden_size, static_cast<float>(eps), stream);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return y;
}
