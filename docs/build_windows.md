# Windows Build Notes

这份文档解释本机 Windows 构建环境为什么这样配置。

对应脚本：

```text
scripts/build_windows.ps1
```

## 1. 当前本机环境

当前项目已经在本机跑通：

```text
Python: 3.12
PyTorch: 2.11.0+cu128
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
CUDA compiler: nvcc 12.8.93
MSVC toolset: 14.44.35207
```

测试结果：

```text
pytest tests/test_rmsnorm.py -q
4 passed
```

## 2. 为什么不用默认 python

系统默认 `python` 是 3.14。

PyTorch CUDA 生态通常不会第一时间支持最新 Python，所以项目用 Python 3.12 创建虚拟环境：

```text
.venv/
```

以后运行项目内命令时，优先使用：

```powershell
.\.venv\Scripts\python.exe
```

不要直接用：

```powershell
python
```

否则可能切到 Python 3.14，导致找不到 PyTorch 或扩展模块。

## 3. 为什么有 .cuda 目录

这台机器最开始只有 NVIDIA 驱动，没有完整 CUDA Toolkit：

```text
nvidia-smi 能看到 GPU
nvcc --version 找不到
```

PyTorch wheel 自带运行时库，但编译自定义 CUDA extension 还需要 `nvcc`。

为了避免系统级安装 CUDA Toolkit，我们下载了 NVIDIA 官方 CUDA 12.8 redistributable 组件，并合并到：

```text
.cuda/toolkit/
```

里面关键文件包括：

```text
.cuda/toolkit/bin/nvcc.exe
.cuda/toolkit/include/cuda_runtime.h
.cuda/toolkit/lib/x64/cudart.lib
```

构建时脚本会设置：

```text
CUDA_HOME=.cuda/toolkit
CUDA_PATH=.cuda/toolkit
```

让 PyTorch extension 找到本地 CUDA 编译工具链。

## 4. 为什么不用默认 MSVC

机器上有 Visual Studio 2026 Insiders。默认 toolset 是：

```text
MSVC 14.51
```

CUDA 12.8 默认只支持到 VS 2022 范围。直接用 MSVC 14.51 会遇到两个问题：

```text
unsupported Microsoft Visual Studio version
cudafe++ ACCESS_VIOLATION
```

机器上还装有较旧的 MSVC toolset：

```text
MSVC 14.44.35207
```

这个版本更接近 VS 2022 工具链，能和 CUDA 12.8 正常配合。所以脚本固定使用：

```text
-vcvars_ver=14.44.35207
```

## 5. build_windows.ps1 在做什么

脚本核心做 4 件事：

```text
1. 进入 Visual Studio x64 开发环境
2. 固定 MSVC 14.44 toolset
3. 设置 CUDA_HOME / CUDA_PATH 指向项目内 .cuda/toolkit
4. 执行 pip install -e . --no-build-isolation
```

运行方式：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
```

## 6. 为什么有这么多环境变量

Windows 下构建 PyTorch CUDA extension 需要很多工具互相找到：

```text
Python 要找到 PyTorch
PyTorch 要找到 nvcc
nvcc 要找到 cl.exe
link.exe 要找到 rc.exe
link.exe 要找到 cudart.lib 和 torch libraries
```

脚本里的关键变量：

```text
CUDA_HOME
CUDA_PATH
TORCH_CUDA_ARCH_LIST
CL_EXE
LINK_EXE
LIB_EXE
NVCC_CCBIN
BUILD_PATH_PREFIX
TORCH_DONT_CHECK_COMPILER_ABI
```

其中：

```text
TORCH_CUDA_ARCH_LIST=12.0
```

是因为 RTX 5060 Laptop GPU 的 compute capability 是：

```text
12.0
```

## 7. 为什么 setup.py 里也有 Windows 兼容逻辑

当前 Codex/Windows 命令环境会让 Python 子进程看到的 `PATH` 不完全等于外层 shell 的 PATH。

这会导致：

```text
cmd 能 where cl
但 Python 子进程里找不到 cl.exe
```

所以 `setup.py` 做了几件兼容处理：

```text
用 CL_EXE 指定 cl.exe 绝对路径
用 LINK_EXE 指定 link.exe 绝对路径
用 LIB_EXE 指定 lib.exe 绝对路径
用 NVCC_CCBIN 给 nvcc 指定 host compiler 目录
把必要目录 prepend 到 Python 进程 PATH
```

这些逻辑只在相关环境变量存在时生效，普通环境不会受影响。

## 8. 为什么用 --no-build-isolation

第一次构建时，`pip` 会尝试创建隔离构建环境，并联网下载构建依赖。

当前环境网络访问受控，所以构建时使用：

```powershell
pip install -e . --no-build-isolation
```

意思是：

```text
直接用当前 .venv 里的 setuptools / wheel / torch / ninja
```

这更稳定，也避免重复联网。

## 9. 常用命令

重新构建：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
```

跑测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_rmsnorm.py -q
```

跑 benchmark：

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_rmsnorm.py
```

检查 PyTorch CUDA：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

检查本地 nvcc：

```powershell
.\.cuda\toolkit\bin\nvcc.exe --version
```

