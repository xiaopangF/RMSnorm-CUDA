$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$CudaHome = Join-Path $ProjectRoot ".cuda\toolkit"
$VsDevCmd = "C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat"
$MsvcRoot = "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Tools\MSVC\14.44.35207"
$MsvcBin = Join-Path $MsvcRoot "bin\Hostx64\x64"
$SdkBin = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64"
$NinjaExe = Join-Path $ProjectRoot ".venv\Scripts\ninja.exe"

if (-not (Test-Path $VenvPython)) {
    throw "Missing virtual environment Python: $VenvPython"
}

if (-not (Test-Path (Join-Path $CudaHome "bin\nvcc.exe"))) {
    throw "Missing local CUDA nvcc: $CudaHome\bin\nvcc.exe"
}

$commands = @(
    "chcp 65001 > nul",
    "set `"VSLANG=1033`"",
    "`"$VsDevCmd`" -arch=x64 -vcvars_ver=14.44.35207",
    "set `"DISTUTILS_USE_SDK=1`"",
    "set `"MSSdk=1`"",
    "set `"TORCH_DONT_CHECK_COMPILER_ABI=1`"",
    "set `"CL_EXE=$MsvcBin\cl.exe`"",
    "set `"LINK_EXE=$MsvcBin\link.exe`"",
    "set `"LIB_EXE=$MsvcBin\lib.exe`"",
    "set `"CC=$MsvcBin\cl.exe`"",
    "set `"CXX=$MsvcBin\cl.exe`"",
    "set `"NVCC_CCBIN=$MsvcBin`"",
    "set `"NINJA_EXE=$NinjaExe`"",
    "set `"BUILD_PATH_PREFIX=$SdkBin`"",
    "set `"CUDA_HOME=$CudaHome`"",
    "set `"CUDA_PATH=$CudaHome`"",
    "set `"TORCH_CUDA_ARCH_LIST=12.0`"",
    "`"$VenvPython`" -m pip install -e . --no-build-isolation"
)
$command = $commands -join " && "

Push-Location $ProjectRoot
try {
    cmd /v:on /c $command
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
