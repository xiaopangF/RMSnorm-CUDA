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
shape: [32, 4096]
```

原因：

```text
float16 更接近真实 LLM 推理
fused_warp 是当前比较实用的版本
[32, 4096] 是 benchmark 里常用 shape
```

## 2. 纯 Python profile target

先确认 profile target 本身能跑：

```powershell
.\.venv\Scripts\python.exe benchmarks\profile_rmsnorm.py --op fused_warp --dtype float16 --batch 32 --hidden-size 4096 --warmup 10 --repeat 50
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

## 5. 先看哪些指标

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

## 6. 怎么比较两个实现

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

## 7. 当前结论

当前 benchmark 已经说明：

```text
warp reduction 有时比 shared reduction 快一点
half2 没有稳定超过普通 warp 版本
```

所以后续优化应该先靠 profiling 判断瓶颈，而不是继续凭感觉改 kernel。

下一步建议：

```text
1. 安装并配置 Nsight Systems / Nsight Compute
2. 跑 fused_warp 和 fused_half2 的 ncu profile
3. 记录 Duration、DRAM Throughput、SM Throughput、Warp Stall
4. 决定下一步是优化访存、减少重复读取，还是改线程分工
```
