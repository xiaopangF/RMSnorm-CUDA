#include <ATen/ATen.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


namespace {

bool is_supported_dtype(at::ScalarType dtype) {
  return dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16;
}

int64_t get_hidden_size(at::Tensor x) {
  TORCH_CHECK(x.dim() >= 1, "x must have shape [..., hidden_size]");
  return x.size(x.dim() - 1);
}

int64_t get_row_count(at::Tensor x, int64_t hidden_size) {
  return x.numel() / hidden_size;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_reduce_sum(float value, float* shared_warp_sums) {
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp_id = threadIdx.x / warpSize;
  const int num_warps = (blockDim.x + warpSize - 1) / warpSize;

  value = warp_reduce_sum(value);
  if (lane == 0) {
    shared_warp_sums[warp_id] = value;
  }
  __syncthreads();

  value = threadIdx.x < num_warps ? shared_warp_sums[lane] : 0.0f;
  if (warp_id == 0) {
    value = warp_reduce_sum(value);
    if (lane == 0) {
      shared_warp_sums[0] = value;
    }
  }
  __syncthreads();

  return shared_warp_sums[0];
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
__global__ void rmsnorm_warp_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ y,
    int64_t rows,
    int64_t hidden_size,
    float eps) {
  extern __shared__ float shared_warp_sums[];

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

  const float row_sum = block_reduce_sum(local_sum, shared_warp_sums);
  const float inv_rms = rsqrtf(row_sum / static_cast<float>(hidden_size) + eps);

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
__global__ void fused_add_rmsnorm_warp_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ y,
    int64_t rows,
    int64_t hidden_size,
    float eps) {
  extern __shared__ float shared_warp_sums[];

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

  const float row_sum = block_reduce_sum(local_sum, shared_warp_sums);
  const float inv_rms = rsqrtf(row_sum / static_cast<float>(hidden_size) + eps);

  for (int64_t col = tid; col < hidden_size; col += blockDim.x) {
    const float value =
        static_cast<float>(x[base + col]) + static_cast<float>(residual[base + col]);
    const float scale = static_cast<float>(weight[col]);
    y[base + col] = static_cast<scalar_t>(value * inv_rms * scale);
  }
}

__global__ void rmsnorm_half2_warp_kernel(
    const half2* __restrict__ x,
    const half2* __restrict__ weight,
    half2* __restrict__ y,
    int64_t rows,
    int64_t pair_count,
    int64_t hidden_size,
    float eps) {
  extern __shared__ float shared_warp_sums[];

  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  if (row >= rows) {
    return;
  }

  const int64_t base_pair = static_cast<int64_t>(row) * pair_count;

  float local_sum = 0.0f;
  for (int64_t pair = tid; pair < pair_count; pair += blockDim.x) {
    const float2 value = __half22float2(x[base_pair + pair]);
    local_sum += value.x * value.x + value.y * value.y;
  }

  const float row_sum = block_reduce_sum(local_sum, shared_warp_sums);
  const float inv_rms = rsqrtf(row_sum / static_cast<float>(hidden_size) + eps);

  for (int64_t pair = tid; pair < pair_count; pair += blockDim.x) {
    const float2 value = __half22float2(x[base_pair + pair]);
    const float2 scale = __half22float2(weight[pair]);
    y[base_pair + pair] =
        __floats2half2_rn(value.x * inv_rms * scale.x, value.y * inv_rms * scale.y);
  }
}

__global__ void fused_add_rmsnorm_half2_warp_kernel(
    const half2* __restrict__ x,
    const half2* __restrict__ residual,
    const half2* __restrict__ weight,
    half2* __restrict__ y,
    int64_t rows,
    int64_t pair_count,
    int64_t hidden_size,
    float eps) {
  extern __shared__ float shared_warp_sums[];

  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  if (row >= rows) {
    return;
  }

  const int64_t base_pair = static_cast<int64_t>(row) * pair_count;

  float local_sum = 0.0f;
  for (int64_t pair = tid; pair < pair_count; pair += blockDim.x) {
    const float2 x_value = __half22float2(x[base_pair + pair]);
    const float2 residual_value = __half22float2(residual[base_pair + pair]);
    const float value0 = x_value.x + residual_value.x;
    const float value1 = x_value.y + residual_value.y;
    local_sum += value0 * value0 + value1 * value1;
  }

  const float row_sum = block_reduce_sum(local_sum, shared_warp_sums);
  const float inv_rms = rsqrtf(row_sum / static_cast<float>(hidden_size) + eps);

  for (int64_t pair = tid; pair < pair_count; pair += blockDim.x) {
    const float2 x_value = __half22float2(x[base_pair + pair]);
    const float2 residual_value = __half22float2(residual[base_pair + pair]);
    const float2 scale = __half22float2(weight[pair]);
    const float value0 = x_value.x + residual_value.x;
    const float value1 = x_value.y + residual_value.y;
    y[base_pair + pair] =
        __floats2half2_rn(value0 * inv_rms * scale.x, value1 * inv_rms * scale.y);
  }
}

__global__ void rmsnorm_backward_f32_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ dx,
    float* __restrict__ partial_dweight,
    int64_t rows,
    int64_t hidden_size,
    float eps) {
  extern __shared__ float shared_warp_sums[];

  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  if (row >= rows) {
    return;
  }

  const int64_t base = static_cast<int64_t>(row) * hidden_size;

  float local_square_sum = 0.0f;
  float local_dot = 0.0f;
  for (int64_t col = tid; col < hidden_size; col += blockDim.x) {
    const float x_value = x[base + col];
    const float scaled_grad = grad_out[base + col] * weight[col];
    local_square_sum += x_value * x_value;
    local_dot += scaled_grad * x_value;
  }

  const float square_sum = block_reduce_sum(local_square_sum, shared_warp_sums);
  const float dot = block_reduce_sum(local_dot, shared_warp_sums);
  const float inv_rms = rsqrtf(square_sum / static_cast<float>(hidden_size) + eps);
  const float inv_rms3 = inv_rms * inv_rms * inv_rms;
  const float dx_scale = inv_rms3 * dot / static_cast<float>(hidden_size);

  for (int64_t col = tid; col < hidden_size; col += blockDim.x) {
    const float x_value = x[base + col];
    const float grad_value = grad_out[base + col];
    const float scaled_grad = grad_value * weight[col];
    dx[base + col] = inv_rms * scaled_grad - x_value * dx_scale;
    partial_dweight[base + col] = grad_value * x_value * inv_rms;
  }
}

__global__ void reduce_dweight_f32_kernel(
    const float* __restrict__ partial_dweight,
    float* __restrict__ dweight,
    int64_t rows,
    int64_t hidden_size) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col >= hidden_size) {
    return;
  }

  float sum = 0.0f;
  for (int64_t row = 0; row < rows; ++row) {
    sum += partial_dweight[row * hidden_size + col];
  }
  dweight[col] = sum;
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
void launch_rmsnorm_warp_kernel(
    at::Tensor x,
    at::Tensor weight,
    at::Tensor y,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    cudaStream_t stream) {
  constexpr int threads = 256;
  constexpr int warps = threads / 32;
  const dim3 blocks(static_cast<unsigned int>(rows));
  const size_t shared_bytes = warps * sizeof(float);

  rmsnorm_warp_kernel<scalar_t><<<blocks, threads, shared_bytes, stream>>>(
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

template <typename scalar_t>
void launch_fused_add_rmsnorm_warp_kernel(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor weight,
    at::Tensor y,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    cudaStream_t stream) {
  constexpr int threads = 256;
  constexpr int warps = threads / 32;
  const dim3 blocks(static_cast<unsigned int>(rows));
  const size_t shared_bytes = warps * sizeof(float);

  fused_add_rmsnorm_warp_kernel<scalar_t><<<blocks, threads, shared_bytes, stream>>>(
      x.data_ptr<scalar_t>(),
      residual.data_ptr<scalar_t>(),
      weight.data_ptr<scalar_t>(),
      y.data_ptr<scalar_t>(),
      rows,
      hidden_size,
      eps);
}

void launch_rmsnorm_half2_warp_kernel(
    at::Tensor x,
    at::Tensor weight,
    at::Tensor y,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    cudaStream_t stream) {
  constexpr int threads = 256;
  constexpr int warps = threads / 32;
  const dim3 blocks(static_cast<unsigned int>(rows));
  const size_t shared_bytes = warps * sizeof(float);
  const int64_t pair_count = hidden_size / 2;

  rmsnorm_half2_warp_kernel<<<blocks, threads, shared_bytes, stream>>>(
      reinterpret_cast<const half2*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half2*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<half2*>(y.data_ptr<at::Half>()),
      rows,
      pair_count,
      hidden_size,
      eps);
}

void launch_fused_add_rmsnorm_half2_warp_kernel(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor weight,
    at::Tensor y,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    cudaStream_t stream) {
  constexpr int threads = 256;
  constexpr int warps = threads / 32;
  const dim3 blocks(static_cast<unsigned int>(rows));
  const size_t shared_bytes = warps * sizeof(float);
  const int64_t pair_count = hidden_size / 2;

  fused_add_rmsnorm_half2_warp_kernel<<<blocks, threads, shared_bytes, stream>>>(
      reinterpret_cast<const half2*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half2*>(residual.data_ptr<at::Half>()),
      reinterpret_cast<const half2*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<half2*>(y.data_ptr<at::Half>()),
      rows,
      pair_count,
      hidden_size,
      eps);
}

void launch_rmsnorm_backward_f32_kernel(
    at::Tensor grad_out,
    at::Tensor x,
    at::Tensor weight,
    at::Tensor dx,
    at::Tensor partial_dweight,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    cudaStream_t stream) {
  constexpr int threads = 256;
  constexpr int warps = threads / 32;
  const dim3 blocks(static_cast<unsigned int>(rows));
  const size_t shared_bytes = warps * sizeof(float);

  rmsnorm_backward_f32_kernel<<<blocks, threads, shared_bytes, stream>>>(
      grad_out.data_ptr<float>(),
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      dx.data_ptr<float>(),
      partial_dweight.data_ptr<float>(),
      rows,
      hidden_size,
      eps);
}

void launch_reduce_dweight_f32_kernel(
    at::Tensor partial_dweight,
    at::Tensor dweight,
    int64_t rows,
    int64_t hidden_size,
    cudaStream_t stream) {
  constexpr int threads = 256;
  const dim3 blocks(static_cast<unsigned int>((hidden_size + threads - 1) / threads));

  reduce_dweight_f32_kernel<<<blocks, threads, 0, stream>>>(
      partial_dweight.data_ptr<float>(),
      dweight.data_ptr<float>(),
      rows,
      hidden_size);
}

}  // namespace


at::Tensor rmsnorm_forward(at::Tensor x, at::Tensor weight, double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(is_supported_dtype(x.scalar_type()), "x must be float32, float16, or bfloat16");
  TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "weight dtype must match x dtype");
  TORCH_CHECK(x.dim() >= 1, "x must have shape [..., hidden_size]");
  TORCH_CHECK(weight.dim() == 1, "weight must have shape [hidden_size]");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

  const int64_t hidden_size = get_hidden_size(x);
  TORCH_CHECK(weight.size(0) == hidden_size, "weight size must match x hidden_size");
  TORCH_CHECK(hidden_size > 0, "hidden_size must be greater than 0");
  const int64_t rows = get_row_count(x, hidden_size);

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


at::Tensor rmsnorm_warp_forward(at::Tensor x, at::Tensor weight, double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(is_supported_dtype(x.scalar_type()), "x must be float32, float16, or bfloat16");
  TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "weight dtype must match x dtype");
  TORCH_CHECK(x.dim() >= 1, "x must have shape [..., hidden_size]");
  TORCH_CHECK(weight.dim() == 1, "weight must have shape [hidden_size]");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

  const int64_t hidden_size = get_hidden_size(x);
  TORCH_CHECK(weight.size(0) == hidden_size, "weight size must match x hidden_size");
  TORCH_CHECK(hidden_size > 0, "hidden_size must be greater than 0");
  const int64_t rows = get_row_count(x, hidden_size);

  auto y = at::empty_like(x);
  if (rows == 0) {
    return y;
  }

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  switch (x.scalar_type()) {
    case at::kFloat:
      launch_rmsnorm_warp_kernel<float>(
          x, weight, y, rows, hidden_size, static_cast<float>(eps), stream);
      break;
    case at::kHalf:
      launch_rmsnorm_warp_kernel<at::Half>(
          x, weight, y, rows, hidden_size, static_cast<float>(eps), stream);
      break;
    case at::kBFloat16:
      launch_rmsnorm_warp_kernel<at::BFloat16>(
          x, weight, y, rows, hidden_size, static_cast<float>(eps), stream);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return y;
}


at::Tensor rmsnorm_half2_forward(at::Tensor x, at::Tensor weight, double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be float16");
  TORCH_CHECK(weight.scalar_type() == at::kHalf, "weight must be float16");
  TORCH_CHECK(x.dim() >= 1, "x must have shape [..., hidden_size]");
  TORCH_CHECK(weight.dim() == 1, "weight must have shape [hidden_size]");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

  const int64_t hidden_size = get_hidden_size(x);
  TORCH_CHECK(weight.size(0) == hidden_size, "weight size must match x hidden_size");
  TORCH_CHECK(hidden_size > 0, "hidden_size must be greater than 0");
  TORCH_CHECK(hidden_size % 2 == 0, "hidden_size must be even for half2");
  const int64_t rows = get_row_count(x, hidden_size);

  auto y = at::empty_like(x);
  if (rows == 0) {
    return y;
  }

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  launch_rmsnorm_half2_warp_kernel(x, weight, y, rows, hidden_size, static_cast<float>(eps), stream);
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
  TORCH_CHECK(x.dim() >= 1, "x must have shape [..., hidden_size]");
  TORCH_CHECK(weight.dim() == 1, "weight must have shape [hidden_size]");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(residual.sizes() == x.sizes(), "residual shape must match x shape");

  const int64_t hidden_size = get_hidden_size(x);
  TORCH_CHECK(weight.size(0) == hidden_size, "weight size must match x hidden_size");
  TORCH_CHECK(hidden_size > 0, "hidden_size must be greater than 0");
  const int64_t rows = get_row_count(x, hidden_size);

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


at::Tensor fused_add_rmsnorm_warp_forward(
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
  TORCH_CHECK(x.dim() >= 1, "x must have shape [..., hidden_size]");
  TORCH_CHECK(weight.dim() == 1, "weight must have shape [hidden_size]");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(residual.sizes() == x.sizes(), "residual shape must match x shape");

  const int64_t hidden_size = get_hidden_size(x);
  TORCH_CHECK(weight.size(0) == hidden_size, "weight size must match x hidden_size");
  TORCH_CHECK(hidden_size > 0, "hidden_size must be greater than 0");
  const int64_t rows = get_row_count(x, hidden_size);

  auto y = at::empty_like(x);
  if (rows == 0) {
    return y;
  }

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  switch (x.scalar_type()) {
    case at::kFloat:
      launch_fused_add_rmsnorm_warp_kernel<float>(
          x, residual, weight, y, rows, hidden_size, static_cast<float>(eps), stream);
      break;
    case at::kHalf:
      launch_fused_add_rmsnorm_warp_kernel<at::Half>(
          x, residual, weight, y, rows, hidden_size, static_cast<float>(eps), stream);
      break;
    case at::kBFloat16:
      launch_fused_add_rmsnorm_warp_kernel<at::BFloat16>(
          x, residual, weight, y, rows, hidden_size, static_cast<float>(eps), stream);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return y;
}


at::Tensor fused_add_rmsnorm_half2_forward(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor weight,
    double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be float16");
  TORCH_CHECK(residual.scalar_type() == at::kHalf, "residual must be float16");
  TORCH_CHECK(weight.scalar_type() == at::kHalf, "weight must be float16");
  TORCH_CHECK(x.dim() >= 1, "x must have shape [..., hidden_size]");
  TORCH_CHECK(weight.dim() == 1, "weight must have shape [hidden_size]");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(residual.sizes() == x.sizes(), "residual shape must match x shape");

  const int64_t hidden_size = get_hidden_size(x);
  TORCH_CHECK(weight.size(0) == hidden_size, "weight size must match x hidden_size");
  TORCH_CHECK(hidden_size > 0, "hidden_size must be greater than 0");
  TORCH_CHECK(hidden_size % 2 == 0, "hidden_size must be even for half2");
  const int64_t rows = get_row_count(x, hidden_size);

  auto y = at::empty_like(x);
  if (rows == 0) {
    return y;
  }

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  launch_fused_add_rmsnorm_half2_warp_kernel(
      x, residual, weight, y, rows, hidden_size, static_cast<float>(eps), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return y;
}


std::vector<at::Tensor> rmsnorm_backward_forward(
    at::Tensor grad_out,
    at::Tensor x,
    at::Tensor weight,
    double eps) {
  TORCH_CHECK(grad_out.is_cuda(), "grad_out must be a CUDA tensor");
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
  TORCH_CHECK(grad_out.scalar_type() == at::kFloat, "grad_out must be float32");
  TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
  TORCH_CHECK(x.dim() >= 1, "x must have shape [..., hidden_size]");
  TORCH_CHECK(weight.dim() == 1, "weight must have shape [hidden_size]");
  TORCH_CHECK(grad_out.sizes() == x.sizes(), "grad_out shape must match x shape");
  TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

  const int64_t hidden_size = get_hidden_size(x);
  TORCH_CHECK(weight.size(0) == hidden_size, "weight size must match x hidden_size");
  TORCH_CHECK(hidden_size > 0, "hidden_size must be greater than 0");
  const int64_t rows = get_row_count(x, hidden_size);

  auto dx = at::empty_like(x);
  auto dweight = at::empty_like(weight);
  if (rows == 0) {
    dweight.zero_();
    return {dx, dweight};
  }

  auto partial_dweight = at::empty_like(x);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  launch_rmsnorm_backward_f32_kernel(
      grad_out,
      x,
      weight,
      dx,
      partial_dweight,
      rows,
      hidden_size,
      static_cast<float>(eps),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  launch_reduce_dweight_f32_kernel(partial_dweight, dweight, rows, hidden_size, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {dx, dweight};
}
