# Profiling Guide

这份文档解释怎么给当前 RMSNorm CUDA 算子做 profiling。

先记住一句话：

```text
benchmark 告诉我们快不快
profiling 告诉我们为什么快、为什么慢
```

## 1. 当前 profiling 目标

当前项目已经有几种 kernel：

```text
rmsnorm_shared
rmsnorm_warp
rmsnorm_half2
fused_shared
fused_warp
fused_half2
```

profiling 的目标不是一次看所有东西，而是一次只看一个 op、一个 shape、一个 dtype。

推荐先看：

```text
op: fused_warp
dtype: float16
shape: [32, 4096] 或 [4, 128, 4096]
```

原因：

```text
float16 更接近真实 LLM 推理
fused_warp 是当前比较实用的版本
[32, 4096] 是 benchmark 里常用二维 shape
[4, 128, 4096] 更接近 batch + seq_len + hidden_size
```

## 2. 纯 Python profile target

先确认 profile target 本身能跑：

```powershell
.\.venv\Scripts\python.exe benchmarks\profile_rmsnorm.py --op fused_warp --dtype float16 --batch 32 --hidden-size 4096 --warmup 10 --repeat 50
```

测试三维输入：

```powershell
.\.venv\Scripts\python.exe benchmarks\profile_rmsnorm.py --op fused_warp --dtype float16 --batch 4 --seq-len 128 --hidden-size 4096 --warmup 10 --repeat 50
```

它会输出：

```text
device
op
dtype
shape
repeat
total_ms
avg_ms
```

这个脚本还会打一个 NVTX range。后面 Nsight Systems 可以用这个 range 在时间线上更容易找到我们关心的区域。

## 3. Nsight Systems 看什么

Nsight Systems 对应命令是 `nsys`。

它主要看时间线：

```text
Python 代码什么时候发起 kernel
CUDA kernel 什么时候运行
有没有很多 kernel launch
CPU 和 GPU 中间有没有空档
```

运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\profile_nsys.ps1 -Op fused_warp -DType float16 -Batch 32 -HiddenSize 4096 -Repeat 200
```

三维输入可以加：

```powershell
-SeqLen 128
```

输出文件默认在：

```text
profiles\nsys_rmsnorm.nsys-rep
```

如果命令报：

```text
nsys was not found in PATH
```

说明还没有安装 Nsight Systems，或者安装了但没有加入 PATH。

## 4. Nsight Compute 看什么

Nsight Compute 对应命令是 `ncu`。

它主要看单个 CUDA kernel 内部指标：

```text
显存带宽用了多少
SM 忙不忙
warp 有没有 stall
访存是否合并
shared memory 使用情况
理论 roofline 离峰值有多远
```

运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\profile_ncu.ps1 -Op fused_warp -DType float16 -Batch 32 -HiddenSize 4096 -Repeat 20
```

三维输入同样可以加：

```powershell
-SeqLen 128
```

这个 wrapper 默认会：

```text
先让 Python 脚本跑 warmup
ncu 跳过 warmup kernel
只抓第一个正式 custom kernel
```

这样报告会更干净，不会一次塞进几十个重复 kernel。

输出文件默认在：

```text
profiles\ncu_rmsnorm.ncu-rep
```

如果命令报：

```text
ncu was not found in PATH
```

说明还没有安装 Nsight Compute，或者安装了但没有加入 PATH。

如果命令报：

```text
Cuda driver is not compatible with Nsight Compute
```

说明 Nsight Compute 版本和当前 NVIDIA 驱动不匹配。当前项目机器上观察到的情况是：

```text
Driver Version: 573.24
CUDA Version: 12.8
Nsight Compute: 2025.4.1
```

这种情况下不要继续猜 kernel 问题。要么升级 NVIDIA 驱动，要么安装和当前驱动更匹配的 Nsight Compute 版本。

## 5. PyTorch Profiler 兜底方案

如果暂时跑不了 `ncu`，可以先用 PyTorch Profiler 导出 Chrome trace：

```powershell
.\.venv\Scripts\python.exe benchmarks\torch_profile_rmsnorm.py --op fused_warp --dtype float16 --batch 32 --hidden-size 4096 --warmup 10 --repeat 50
```

也可以 profile 三维输入：

```powershell
.\.venv\Scripts\python.exe benchmarks\torch_profile_rmsnorm.py --op fused_warp --dtype float16 --batch 4 --seq-len 128 --hidden-size 4096 --warmup 10 --repeat 50
```

它会输出两类东西：

```text
终端里的 operator / CUDA 时间表
profiles\ 下的 Chrome trace JSON
```

打开 trace 的方法：

```text
Chrome 浏览器地址栏输入 chrome://tracing
点 Load
选择 profiles\ 里的 torch_trace_*.json
```

PyTorch Profiler 不如 Nsight Compute 细，但它能先回答：

```text
哪些 CUDA kernel 被调用了
每个 kernel 大概用了多久
Python / PyTorch 调用和 CUDA kernel 之间有没有明显空档
```

如果输出里看到：

```text
warning: no CUDA activity was captured
CUPTI initialization failed
```

说明 PyTorch Profiler 也没拿到 CUDA 活动。当前机器上也观察到了这个情况。此时 CPU trace 还能看 Python 调用，但不能用它分析 CUDA kernel 内部性能。

这和 `ncu` 报驱动不兼容是同一类问题：CUPTI / profiler 工具链和当前 NVIDIA 驱动没有配好。

## 6. 先看哪些指标

第一次看 Nsight Compute，不要被所有指标淹没。先看这几类：

```text
Duration
Memory Throughput
L2 Throughput
DRAM Throughput
SM Throughput
Achieved Occupancy
Warp Stall Reasons
```

通俗理解：

```text
DRAM Throughput 高，说明主要在等显存
SM Throughput 高，说明计算单元比较忙
Occupancy 太低，说明并发不够
Warp stall 很高，说明 thread 经常在等某些资源
```

## 7. 怎么比较两个实现

不要只 profile 一个版本。建议成对比较：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\profile_ncu.ps1 -Op fused_warp -DType float16 -Batch 32 -HiddenSize 4096 -Output profiles\ncu_fused_warp
powershell -ExecutionPolicy Bypass -File .\scripts\profile_ncu.ps1 -Op fused_half2 -DType float16 -Batch 32 -HiddenSize 4096 -Output profiles\ncu_fused_half2
```

然后比较：

```text
哪个 Duration 更低
哪个 DRAM Throughput 更高
哪个 Warp Stall 更少
half2 有没有真的减少访存瓶颈
```

这能回答一个很关键的问题：

```text
为什么 half2 看起来更高级，但 benchmark 没有稳定更快？
```

## 8. 当前结论

当前 benchmark 已经说明：

```text
warp reduction 有时比 shared reduction 快一点
half2 没有稳定超过普通 warp 版本
```

当前机器上 profiling 工具链状态：

```text
Nsight Compute 2025.4.1 已安装
ncu.exe 可以启动
ncu 采集时报 Cuda driver is not compatible with Nsight Compute
PyTorch Profiler 可以导出 trace
PyTorch Profiler 的 CUDA 活动因 CUPTI 初始化失败而缺失
```

所以后续优化应该先修好 profiling 工具链，而不是继续凭感觉改 kernel。

下一步建议：

```text
1. 更新 NVIDIA 驱动，或安装和当前驱动匹配的 Nsight Compute
2. 确认 ncu 能成功采集 fused_warp
3. 跑 fused_warp 和 fused_half2 的 ncu profile
4. 记录 Duration、DRAM Throughput、SM Throughput、Warp Stall
5. 决定下一步是优化访存、减少重复读取，还是改线程分工
```
