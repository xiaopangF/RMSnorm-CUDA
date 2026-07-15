# rmsnorm-cuda

一个面向新手的 CUDA LLM 算子项目。第一阶段目标：

> 实现可以从 Python 调用的 RMSNorm CUDA forward 算子，并和 PyTorch reference 对比正确性和速度。

当前已经包含两个 CUDA 算子：

- `rmsnorm(x, weight, eps)`: 普通 RMSNorm。
- `fused_add_rmsnorm(x, residual, weight, eps)`: 先算 `x + residual`，再做 RMSNorm，但不生成中间 tensor。

## RMSNorm 在做什么

RMSNorm 会把每一行数字按整体大小缩放一下：

```text
output = input / sqrt(mean(input^2) + eps) * weight
```

如果输入是 `[batch, hidden_size]`，当前 CUDA 分工是：

```text
一个 CUDA block 处理一行
一个 block 里的多个 thread 一起处理这一行的 hidden_size 个数字
```

每一行做三步：

```text
1. 每个 thread 计算自己负责数字的平方和
2. block 内所有 thread 汇总平方和
3. 每个 thread 写回自己负责的输出元素
```

## 当前范围

- 支持 `float32` / `float16` / `bfloat16`
- 支持二维输入 `[batch, hidden_size]`
- 支持 CUDA tensor
- 支持 fused residual add + RMSNorm forward
- 实现 forward，不实现 backward
- 提供 correctness test 和 benchmark

后续再扩展：

- Softmax
- RoPE
- FlashAttention mini

## 本机环境

当前本机已经跑通：

```text
Python: 3.12
PyTorch: 2.11.0+cu128
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
CUDA compiler: nvcc 12.8.93
MSVC toolset: 14.44.35207
```

项目内关键环境目录：

```text
.venv/          Python 虚拟环境
.cuda/toolkit/  本地 CUDA 12.8 redistributable
```

这两个目录是本地环境，不应该提交到 Git。

## 构建

本机 Windows 重新构建：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
```

这个脚本会设置 CUDA、MSVC、Windows SDK 和 PyTorch extension 需要的环境变量，然后执行：

```powershell
pip install -e . --no-build-isolation
```

## 正确性测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_rmsnorm.py -q
```

测试会把自定义 CUDA 算子和 PyTorch reference 对比：

```python
y_ref = x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight
```

当前结果：

```text
20 passed
```

## 性能测试

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_rmsnorm.py
```

benchmark 使用 CUDA event 计时，输出：

```text
torch med    PyTorch reference 的 median latency
custom med   自定义 CUDA kernel 的 median latency
custom p90   自定义 CUDA kernel 的 p90 latency
custom GB/s  自定义 CUDA kernel 的估算显存带宽
speedup      torch med / custom med
```

脚本会输出两张表：

- `RMSNorm`: 普通 RMSNorm。
- `Fused add + RMSNorm`: 对比 PyTorch 写法 `rmsnorm_reference(x + residual, weight, eps)` 和自定义融合 kernel。

更宽的 shape sweep：

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_rmsnorm.py --extended
```

测试低精度 dtype：

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_rmsnorm.py --dtype float16
.\.venv\Scripts\python.exe benchmarks\bench_rmsnorm.py --dtype bfloat16
```

## 项目目标

这个项目不是为了第一版就超过 PyTorch，而是为了建立完整闭环：

```text
写 CUDA kernel
接入 PyTorch
验证正确性
benchmark 性能
分析瓶颈
逐步优化
```

## 文档

- `docs/rmsnorm.md`: 解释 RMSNorm 公式、CUDA 线程分工、shared memory reduction、GB/s 指标和后续优化方向。
- `docs/build_windows.md`: 解释本机 Windows + PyTorch CUDA extension 的构建环境和常见坑。
