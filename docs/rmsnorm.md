# RMSNorm CUDA Kernel

这份文档解释当前项目里的 RMSNorm CUDA kernel 在做什么、为什么这样写，以及 benchmark 结果应该怎么看。

对应源码：

- `csrc/rmsnorm_kernel.cu`
- `csrc/binding.cpp`
- `rmsnorm_cuda/__init__.py`
- `benchmarks/bench_rmsnorm.py`

## 1. RMSNorm 是什么

RMSNorm 是大模型里常见的归一化算子。它处理一行数字，把这一行数字按整体大小缩放一下。

假设一行输入是：

```text
x = [x0, x1, x2, ..., xN-1]
```

RMSNorm 先算这一行的平方平均值：

```text
mean_square = (x0^2 + x1^2 + ... + xN-1^2) / N
```

然后取倒数平方根：

```text
inv_rms = 1 / sqrt(mean_square + eps)
```

最后每个元素都乘这个缩放系数和对应的 `weight`：

```text
y[i] = x[i] * inv_rms * weight[i]
```

`eps` 是一个很小的数，用来避免除以 0。

## 2. 输入输出

当前项目只支持二维 `float32` CUDA tensor：

```text
x:      [batch, hidden_size]
weight: [hidden_size]
y:      [batch, hidden_size]
```

比如：

```text
x shape = [32, 4096]
```

可以理解成：

```text
32 行
每行 4096 个数字
```

RMSNorm 按行处理。每一行有自己的 `mean_square` 和 `inv_rms`。

## 3. GPU 怎么分工

当前 kernel 的分工方式是：

```text
一个 CUDA block 处理 x 的一行
一个 block 里有 256 个 thread
```

如果 `hidden_size = 4096`，那么一行有 4096 个数字。256 个 thread 会一起处理这一行。

每个 thread 隔着 `blockDim.x` 处理多个元素：

```cpp
for (int64_t col = tid; col < hidden_size; col += blockDim.x) {
  const float value = x[base + col];
  local_sum += value * value;
}
```

如果 `blockDim.x = 256`：

```text
thread 0 处理 col 0, 256, 512, ...
thread 1 处理 col 1, 257, 513, ...
thread 2 处理 col 2, 258, 514, ...
```

这样 256 个 thread 能覆盖整行。

## 4. shared memory 和 reduction

每个 thread 先算自己负责部分的平方和：

```text
local_sum
```

但 RMSNorm 需要整行平方和，所以所有 thread 的 `local_sum` 要加起来。

当前做法是把每个 thread 的结果放进 shared memory：

```cpp
shared_sum[tid] = local_sum;
__syncthreads();
```

shared memory 可以理解成一个 block 内部 thread 都能访问的小黑板。

然后做并行归约：

```cpp
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
  if (tid < stride) {
    shared_sum[tid] += shared_sum[tid + stride];
  }
  __syncthreads();
}
```

过程是：

```text
256 个数 -> 128 个数 -> 64 个数 -> 32 个数 -> ... -> 1 个数
```

最后 `shared_sum[0]` 就是整行平方和。

`__syncthreads()` 用来保证 block 内所有 thread 都走到同一个阶段。没有同步的话，某些 thread 可能会读到还没写完的 shared memory。

## 5. 写回输出

算出整行的 `inv_rms` 后，每个 thread 再处理自己负责的列：

```cpp
for (int64_t col = tid; col < hidden_size; col += blockDim.x) {
  y[base + col] = x[base + col] * inv_rms * weight[col];
}
```

注意：

```text
x 的每一行都有自己的 inv_rms
weight 只按 hidden 维度变化
```

## 6. 为什么能比 PyTorch reference 快

benchmark 里的 PyTorch reference 是：

```python
y_ref = x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight
```

这行 Python 看起来简单，但底层可能拆成多个操作：

```text
x * x
mean
sqrt
divide
multiply weight
```

这可能产生多个临时 tensor 和多次 kernel launch。

我们的自定义 CUDA kernel 把这些步骤放进一个 kernel：

```text
读 x
算平方和
算 inv_rms
写 y
```

所以它减少了中间步骤和 kernel launch 开销。

## 7. 为什么要看 GB/s

RMSNorm 主要不是复杂计算，而是读写显存。

这种算子通常叫 memory-bound kernel。性能瓶颈更可能是：

```text
显存读写速度
访存是否连续
是否重复读写
```

所以只看 `ms` 不够，还要估算带宽：

```text
GB/s = 读写字节数 / kernel 时间
```

当前 custom kernel 的粗略读写量：

```text
读 x 两次
读 weight 一次
写 y 一次
```

对于 `float32`，每个元素 4 bytes，所以估算：

```text
bytes = batch * hidden_size * 4 * 4
```

这是粗略估算，不代表真实硬件事务数，但足够帮助我们判断优化方向。

## 8. benchmark 输出怎么看

运行：

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_rmsnorm.py
```

输出列含义：

```text
batch       batch size
hidden      hidden_size
torch med   PyTorch reference 的 median latency，单位 ms
custom med  自定义 CUDA kernel 的 median latency，单位 ms
custom p90  自定义 CUDA kernel 的 p90 latency，单位 ms
custom GB/s 自定义 CUDA kernel 的估算带宽
speedup     torch med / custom med
```

为什么用 median：

```text
mean 容易被偶发抖动影响
median 更能代表常见情况
p90 能看尾延迟是否稳定
```

更宽的 shape sweep：

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_rmsnorm.py --extended
```

## 9. 当前 kernel 的限制

当前版本限制：

```text
只支持 float32
只支持 forward
只支持 contiguous tensor
每行固定使用 256 个 thread
reduction 使用 shared memory
```

这些限制是刻意保留的。第一版目标是把 CUDA kernel、PyTorch binding、测试和 benchmark 全链路跑通。

## 10. 下一步优化

建议顺序：

```text
1. 用 benchmark 观察不同 batch / hidden_size 的 GB/s
2. 用 warp shuffle 优化 reduction
3. 支持 half / bfloat16
4. 实现 fused residual + RMSNorm
5. 支持 backward
```

最值得先做的是 fused residual + RMSNorm。

真实大模型里经常有：

```python
tmp = x + residual
out = rmsnorm(tmp, weight)
```

融合成一个 kernel 后，可以减少一次中间 tensor 读写。
