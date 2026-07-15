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

当前项目支持二维 CUDA tensor，dtype 可以是 `float32`、`float16` 或 `bfloat16`：

```text
x:      [batch, hidden_size]
weight: [hidden_size]
y:      [batch, hidden_size]
```

`x`、`weight` 和输出 `y` 的 dtype 相同。比如输入是 `float16`，输出也是 `float16`。

不过平方和不是用 `float16` 累加，而是转成 `float` 累加：

```text
低精度输入/输出
float32 累加平方和
低精度写回输出
```

这样做是因为 RMSNorm 要把一整行很多数字加起来。如果全程用 `float16` 加，误差会更明显。

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

## 4. shared memory reduction

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

这个版本保留在项目里，作为 baseline：

```python
rmsnorm_cuda.rmsnorm(x, weight)
rmsnorm_cuda.fused_add_rmsnorm(x, residual, weight)
```

## 5. warp shuffle reduction

项目里还实现了 warp shuffle 版本：

```python
rmsnorm_cuda.rmsnorm_warp(x, weight)
rmsnorm_cuda.fused_add_rmsnorm_warp(x, residual, weight)
```

先理解两个词：

```text
warp    GPU 里一组一起执行的 thread，NVIDIA GPU 上通常是 32 个 thread
shuffle warp 内 thread 之间直接交换寄存器里的值
```

shared memory reduction 的思路是：

```text
每个 thread 把 local_sum 写到 shared memory
一轮一轮读 shared memory 加起来
每一轮都要 __syncthreads()
```

warp shuffle reduction 的思路是：

```text
先在每个 warp 内直接用寄存器交换求和
每个 warp 只写一个结果到 shared memory
最后再把几个 warp 的结果加起来
```

当前 block 有 256 个 thread，也就是 8 个 warp。所以 shared memory 版本大致要存 256 个 float，warp 版本只需要存 8 个 warp sum。

核心代码是：

```cpp
for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
  value += __shfl_down_sync(0xffffffff, value, offset);
}
```

通俗理解：一个 warp 里的 32 个 thread 不用把数字写到 shared memory 小黑板上，而是直接互相传数字并加起来。

这不一定会带来巨大加速，因为当前 RMSNorm 很多时候受显存读写和 kernel launch 开销影响。但它减少了 shared memory 使用和同步次数，是 CUDA reduction 优化里非常基础、非常常见的一步。

## 6. 写回输出

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

## 7. 为什么能比 PyTorch reference 快

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

## 8. 为什么要看 GB/s

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

不同 dtype 的元素大小不同：

```text
float32  = 4 bytes
float16  = 2 bytes
bfloat16 = 2 bytes
```

所以估算：

```text
bytes = batch * hidden_size * element_size * 4
```

这是粗略估算，不代表真实硬件事务数，但足够帮助我们判断优化方向。

## 9. benchmark 输出怎么看

运行：

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_rmsnorm.py
```

输出列含义：

```text
batch       batch size
hidden      hidden_size
torch       PyTorch reference 的 median latency，单位 ms
shared      shared memory reduction 版本的 median latency，单位 ms
warp        warp shuffle reduction 版本的 median latency，单位 ms
warp p90    warp 版本的 p90 latency，单位 ms
warp GB/s   warp 版本的估算显存带宽
warp/shared shared med / warp med
torch/warp  torch med / warp med
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

## 10. fused add + RMSNorm 是什么

真实大模型里经常会看到这种模式：

```python
tmp = x + residual
out = rmsnorm(tmp, weight)
```

通俗理解：

```text
x        是当前层新算出来的结果
residual 是之前绕过来的残差
tmp      是两者相加后的结果
out      是归一化后的结果
```

普通 PyTorch 写法通常会先生成 `tmp` 这个中间 tensor，然后 RMSNorm 再读取它。

融合版 kernel 做的是：

```text
不把 tmp 单独写进显存
需要 tmp[i] 的时候，现场计算 x[i] + residual[i]
然后直接参与 RMSNorm
```

数学结果不变：

```text
out = RMSNorm(x + residual)
```

但显存读写更少。对于这种 memory-bound 算子，少读写显存通常比少做几次加法更重要。

当前 fused kernel 的粗略读写量：

```text
读 x 两次
读 residual 两次
读 weight 一次
写 y 一次
```

它仍然会重复读取 `x` 和 `residual`，因为第一遍要算平方和，第二遍要写输出。后续优化可以尝试让更小的 hidden size 复用中间值，或者用更高级的 reduction 写法。

## 11. 当前 kernel 的限制

当前版本限制：

```text
只支持 forward
只支持 contiguous tensor
每行固定使用 256 个 thread
还没有 backward
```

这些限制是刻意保留的。第一版目标是把 CUDA kernel、PyTorch binding、测试和 benchmark 全链路跑通。

## 12. 下一步优化

建议顺序：

```text
1. 用 benchmark 观察不同 batch / hidden_size 的 GB/s
2. 增加向量化读写，比如 half2 / bf162
3. 支持任意前缀维度，比如 [batch, seq_len, hidden_size]
4. 支持 backward
5. 扩展更多 LLM 常见算子
```

fused residual + RMSNorm、`float16`、`bfloat16` 和 warp shuffle reduction 都已经实现。下一步最值得做的是向量化读写，因为真实推理里低精度 tensor 通常需要一次搬多个元素，才能更好地吃满显存带宽。
